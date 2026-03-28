#include "common.hpp"
#ifdef USE_ORT

    #include "armor_detector_onnxruntime.hpp"
    #include "tasks/auto_aim/armor_detect/armor_detector_common.hpp"
    #include "tasks/auto_aim/armor_detect/armor_infer.hpp"
    #include "utils/utils.hpp"
    #include "wust_vl/ml_net/onnxruntime/onnxruntime_net.hpp"
namespace awakening {
namespace auto_aim {
    struct ArmorDetectorOnnxRuntime::Impl {
    public:
        Impl(const YAML::Node& config, bool use_armor_detect_common) {
            if (use_armor_detect_common) {
                armor_detect_common_ = std::make_unique<ArmorDetectorCommon>(config);
            }
            std::string model_type = config["onnxruntime"]["model_type"].as<std::string>();
            auto model = armor_infer::modeFromString(model_type);
            float conf_threshold = config["onnxruntime"]["conf_threshold"].as<float>();
            int top_k = config["onnxruntime"]["top_k"].as<int>();
            float nms_threshold = config["onnxruntime"]["nms_threshold"].as<float>();
            armor_infer_ = std::make_unique<armor_infer::ArmorInfer>(
                model,
                conf_threshold,
                nms_threshold,
                top_k
            );
            std::string provider = config["onnxruntime"]["provider"].as<std::string>("CPU");
            provider_ = wust_vl::ml_net::string2OrtProvider(provider);
            onnxruntime_net_ = std::make_unique<wust_vl::ml_net::OnnxRuntimeNet>();
            wust_vl::ml_net::OnnxRuntimeNet::Params params;
            std::string model_path =
                utils::expandEnv(config["onnxruntime"]["model_path"].as<std::string>());
            params.model_path = model_path;
            params.provider = provider_;
            onnxruntime_net_->init(params);
        }

        ~Impl() {
            onnxruntime_net_.reset();
            armor_detect_common_.reset();
        }
        void setCallback(DetectorCallback callback) {
            infer_callback_ = callback;
        }
        void
        processCallback(const CommonFrame& frame, const std::optional<ArmorNumber>& target_number) {
            Eigen::Matrix3f transform_matrix;
            auto roi = frame.img_frame.src_img(frame.expanded);
            cv::Mat resized_img = utils::letterbox(
                roi,
                transform_matrix,
                armor_infer_->inputW(),
                armor_infer_->inputH()
            );
            const bool swap_rb = armor_infer_->inputRGB()
                != (frame.img_frame.pixel_format == wust_vl::video::PixelFormat::RGB);
            const float scale = armor_infer_->useNorm() ? 1.0f / 255.0f : 1.0f;
            cv::Mat blob = cv::dnn::blobFromImage(
                resized_img,
                scale,
                cv::Size(armor_infer_->inputW(), armor_infer_->inputH()),
                cv::Scalar(0, 0, 0),
                swap_rb
            );

            auto output_data = onnxruntime_net_->infer(blob.ptr<float>(), blob.total());

            auto output_shape = onnxruntime_net_->getOutputShape();
            int rows = static_cast<int>(output_shape[1]);
            int cols = static_cast<int>(output_shape[2]);
            cv::Mat output_buffer(rows, cols, CV_32F, output_data);

            // Parsed variable
            std::vector<ArmorObject> objs_result;
            objs_result = armor_infer_->postProcess(output_buffer);
            std::vector<ArmorObject> armors;
            if (armor_detect_common_) {
                armors = armor_detect_common_->detectNet(
                    resized_img,
                    objs_result,
                    transform_matrix,
                    frame.detect_color,
                    target_number
                );
                // Call callback function
                if (this->infer_callback_) {
                    this->infer_callback_(armors, frame);
                    return;
                }
            } else {
                for (auto obj: objs_result) {
                    auto detect_color = frame.detect_color;
                    if (detect_color == EnemyColor::RED && obj.color == ArmorColor::BLUE) {
                        continue;
                    } else if (detect_color == EnemyColor::BLUE && obj.color == ArmorColor::RED) {
                        continue;
                    }
                    obj.transform(transform_matrix);
                    armors.push_back(obj);
                }
                if (this->infer_callback_) {
                    this->infer_callback_(armors, frame);
                    return;
                }
            }

            return;
        }
        void pushInput(CommonFrame& frame, const std::optional<ArmorNumber>& target_number) {
            frame.id = current_id_++;
            processCallback(frame, target_number);
        }

        wust_vl::ml_net::OrtProvider provider_ = wust_vl::ml_net::OrtProvider::CPU;
        DetectorCallback infer_callback_;
        std::unique_ptr<ArmorDetectorCommon> armor_detect_common_;
        std::unique_ptr<armor_infer::ArmorInfer> armor_infer_;
        int current_id_ = 0;
        std::unique_ptr<wust_vl::ml_net::OnnxRuntimeNet> onnxruntime_net_;
    };
    ArmorDetectorOnnxRuntime::ArmorDetectorOnnxRuntime(
        const YAML::Node& config,
        bool use_armor_detect_common
    ) {
        _impl = std::make_unique<Impl>(config, use_armor_detect_common);
    }
    ArmorDetectorOnnxRuntime::~ArmorDetectorOnnxRuntime() {
        _impl.reset();
    }
    void ArmorDetectorOnnxRuntime::setCallback(DetectorCallback callback) {
        _impl->setCallback(callback);
    }
    void ArmorDetectorOnnxRuntime::pushInput(
        CommonFrame& frame,
        const std::optional<ArmorNumber>& target_number
    ) {
        _impl->pushInput(frame, target_number);
    }
} // namespace auto_aim
} // namespace awakening
#endif