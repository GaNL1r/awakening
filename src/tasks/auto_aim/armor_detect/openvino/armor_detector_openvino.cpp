// Copyright 2025 Zikang Xie
// Copyright 2025 Xiaojian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifdef USE_OPENVINO
    #include "tasks/auto_aim/armor_detect/openvino/armor_detector_openvino.hpp"
    #include "tasks/auto_aim/armor_detect/armor_detector_common.hpp"
    #include "tasks/auto_aim/armor_detect/armor_infer.hpp"
    #include "utils/utils.hpp"
    #include "wust_vl/ml_net/openvino/openvino_net.hpp"
namespace awakening {
namespace auto_aim {
    struct ArmorDetectorOpenVino::Impl {
    public:
        Impl(const YAML::Node& config, bool use_armor_detect_common) {
            if (use_armor_detect_common) {
                armor_detect_common_ = std::make_unique<ArmorDetectorCommon>(config);
            }
            std::string model_type = config["openvino"]["model_type"].as<std::string>();
            auto model = armor_infer::modeFromString(model_type);
            float conf_threshold = config["openvino"]["conf_threshold"].as<float>();
            int top_k = config["openvino"]["top_k"].as<int>();
            float nms_threshold = config["openvino"]["nms_threshold"].as<float>();
            bool use_throughputmode = config["openvino"]["use_throughputmode"].as<bool>();
            armor_infer_ = std::make_unique<armor_infer::ArmorInfer>(
                model,
                conf_threshold,
                nms_threshold,
                top_k
            );
            std::string model_path =
                utils::expandEnv(config["openvino"]["model_path"].as<std::string>());

            auto device_name = config["openvino"]["device_name"].as<std::string>();
            ov_params_.model_path = model_path;
            ov_params_.device_name = device_name;
            ov_params_.mode = use_throughputmode ? ov::hint::PerformanceMode::THROUGHPUT
                                                 : ov::hint::PerformanceMode::LATENCY;
            initOpenVINO();
        }
        void
        initOpenVINO(wust_vl::video::PixelFormat pixel_format = wust_vl::video::PixelFormat::BGR) {
            openvino_net_.reset();
            openvino_net_ = std::make_unique<wust_vl::ml_net::OpenvinoNet>();
            const auto ppp_init_fun = [this, pixel_format](ov::preprocess::PrePostProcessor& ppp) {
                if (pixel_format == wust_vl::video::PixelFormat::RGB) {
                    ppp.input()
                        .tensor()
                        .set_element_type(ov::element::u8)
                        .set_layout("NHWC")
                        .set_color_format(ov::preprocess::ColorFormat::RGB);
                } else {
                    ppp.input()
                        .tensor()
                        .set_element_type(ov::element::u8)
                        .set_layout("NHWC")
                        .set_color_format(ov::preprocess::ColorFormat::BGR);
                }
                pixel_format_ = pixel_format;
                const bool RGB = armor_infer_->inputRGB();
                const float scale = armor_infer_->useNorm() ? 255.0f : 1.0f;
                if (RGB) {
                    ppp.input()
                        .preprocess()
                        .convert_element_type(ov::element::f32)
                        .convert_color(ov::preprocess::ColorFormat::RGB)
                        .scale(scale);
                } else {
                    ppp.input()
                        .preprocess()
                        .convert_element_type(ov::element::f32)
                        .convert_color(ov::preprocess::ColorFormat::BGR)
                        .scale(scale);
                }

                ppp.input().model().set_layout("NCHW");

                ppp.output().tensor().set_element_type(ov::element::f32);
            };

            openvino_net_->init(ov_params_, ppp_init_fun);
        }

        ~Impl() {
            openvino_net_.reset();
            armor_detect_common_.reset();
        }

        void setCallback(DetectorCallback callback) {
            infer_callback_ = callback;
        }
        bool processCallback(
            const CommonFrame& frame,
            const std::optional<ArmorNumber>& target_number
        ) const {
            const auto start = std::chrono::steady_clock::now();
            Eigen::Matrix3f transform_matrix;
            const auto roi = frame.img_frame.src_img(frame.expanded);
            cv::Mat resized_img = utils::letterbox(
                roi,
                transform_matrix,
                armor_infer_->inputW(),
                armor_infer_->inputH()
            );
            const auto input_info = openvino_net_->getInputInfo();
            const auto input_tensor =
                ov::Tensor(input_info.first, input_info.second, resized_img.data);
            const auto output = openvino_net_->infer_thread_local(input_tensor);

            // Process output data
            const auto output_shape = output.get_shape();

            const float* ptr = output.data<const float>();
            cv::Mat
                output_buffer(output_shape[1], output_shape[2], CV_32F, const_cast<float*>(ptr));

            // Parsed variable
            auto objs_result = armor_infer_->postProcess(output_buffer);

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
                    return true;
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
                    return true;
                }
            }

            return false;
        }
        void pushInput(CommonFrame& frame, const std::optional<ArmorNumber>& target_number) {
            if (resetting_) {
                return;
            }
            frame.id = current_id_++;
            if (frame.img_frame.pixel_format != pixel_format_) {
                resetting_ = true;
                initOpenVINO(frame.img_frame.pixel_format);
                resetting_ = false;
            }
            processCallback(frame, target_number);
        }

    private:
        wust_vl::video::PixelFormat pixel_format_ = wust_vl::video::PixelFormat::BGR;
        std::unique_ptr<wust_vl::ml_net::OpenvinoNet> openvino_net_;
        DetectorCallback infer_callback_;
        std::unique_ptr<ArmorDetectorCommon> armor_detect_common_;
        std::unique_ptr<armor_infer::ArmorInfer> armor_infer_;
        int current_id_ = 0;
        wust_vl::ml_net::OpenvinoNet::Params ov_params_;
        bool resetting_ = false;
    };
    ArmorDetectorOpenVino::ArmorDetectorOpenVino(
        const YAML::Node& config,
        bool use_armor_detect_common
    ) {
        _impl = std::make_unique<Impl>(config, use_armor_detect_common);
    }
    ArmorDetectorOpenVino::~ArmorDetectorOpenVino() {
        _impl.reset();
    }
    void ArmorDetectorOpenVino::setCallback(DetectorCallback callback) {
        _impl->setCallback(callback);
    }
    void ArmorDetectorOpenVino::pushInput(
        CommonFrame& frame,
        const std::optional<ArmorNumber>& target_number
    ) {
        _impl->pushInput(frame, target_number);
    }
} // namespace auto_aim
} // namespace awakening
#endif