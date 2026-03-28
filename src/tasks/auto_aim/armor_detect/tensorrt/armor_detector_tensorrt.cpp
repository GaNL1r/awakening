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
#ifdef USE_TRT
    #include "tasks/auto_aim/armor_detect/tensorrt/armor_detector_tensorrt.hpp"
    #include "cuda_infer/armor_infer.hpp"
    #include "tasks/auto_aim/armor_detect/armor_detector_common.hpp"
    #include "tasks/auto_aim/armor_detect/armor_infer.hpp"
    #include "utils/utils.hpp"
    #include "wust_vl/common/concurrency/adaptive_resource_pool.hpp"
    #include "wust_vl/common/utils/logger.hpp"
    #include "wust_vl/common/utils/timer.hpp"
    #include "wust_vl/ml_net/tensorrt/tensorrt_net.hpp"
namespace awakening {
namespace auto_aim {
    static constexpr int MAX_SRC_IMG_W = 1920;
    static constexpr int MAX_SRC_IMG_H = 1440;
    struct ArmorDetectorTrt::Impl {
    public:
        struct Infer {
            std::unique_ptr<nvinfer1::IExecutionContext> context;
            std::unique_ptr<armor_cuda_infer::CudaInfer> cuda_infer;
        };

        Impl(const YAML::Node& config, bool use_armor_detect_common) {
            if (use_armor_detect_common) {
                armor_detect_common_ = std::make_unique<ArmorDetectorCommon>(config);
            }
            const double conf_threshold = config["tensorrt"]["conf_threshold"].as<float>();
            const double nms_threshold = config["tensorrt"]["nms_threshold"].as<float>();
            const int top_k = config["tensorrt"]["top_k"].as<int>();
            const int max_infer_running = config["tensorrt"]["max_infer_running"].as<int>();
            const double min_free_mem_ratio = config["tensorrt"]["min_free_mem_ratio"].as<double>();
            use_cuda_pre_ = config["tensorrt"]["use_cuda_pre"].as<bool>();
            log_time_ = config["tensorrt"]["log_time"].as<bool>();
            const std::string model_type = config["tensorrt"]["model_type"].as<std::string>();
            const std::string model_path =
                utils::expandEnv(config["tensorrt"]["model_path"].as<std::string>());
            const int device_id = config["tensorrt"]["device_id"].as<int>();
            cudaSetDevice(device_id);
            const auto model = armor_infer::modeFromString(model_type);
            armor_infer_ = std::make_unique<armor_infer::ArmorInfer>(
                model,
                conf_threshold,
                nms_threshold,
                top_k
            );
            trt_net_ = std::make_unique<wust_vl::ml_net::TensorRTNet>();
            wust_vl::ml_net::TensorRTNet::Params trt_params;
            trt_params.model_path = model_path;
            trt_params.input_dims =
                nvinfer1::Dims4 { 1, 3, armor_infer_->inputH(), armor_infer_->inputW() };
            trt_net_->init(trt_params);
            const auto input_output_dims = trt_net_->getInputOutputDims();
            input_dims_ = std::get<0>(input_output_dims);
            output_dims_ = std::get<1>(input_output_dims);

            wust_vl::common::concurrency::AdaptiveResourcePool<Infer>::Params pool_params;
            pool_params.resource_initializer = [&]() {
                std::vector<std::unique_ptr<Infer>> infers;
                for (int i = 0; i < max_infer_running; ++i) {
                    auto infer = std::make_unique<Infer>();
                    auto ctx = trt_net_->getAContext();
                    infer->context = std::unique_ptr<nvinfer1::IExecutionContext>(ctx);
                    if (use_cuda_pre_) {
                        infer->cuda_infer = std::make_unique<armor_cuda_infer::CudaInfer>();
                        infer->cuda_infer->init(
                            MAX_SRC_IMG_W,
                            MAX_SRC_IMG_H,
                            armor_infer_->inputW(),
                            armor_infer_->inputH()
                        );
                    }
                    if (!infer->context) {
                        WUST_ERROR("TRT") << "create infer failed, missing context"
                                          << " index:" << i;
                        continue;
                    }
                    if (use_cuda_pre_ && !infer->cuda_infer) {
                        WUST_ERROR("TRT") << "create infer failed, missing cuda_infer"
                                          << " index:" << i;
                        continue;
                    }

                    size_t free_mem, total_mem;
                    cudaMemGetInfo(&free_mem, &total_mem);
                    WUST_DEBUG("TRT") << "Free GPU memory:" << free_mem / 1024.0 / 1024.0 << "MB"
                                      << "Total GPU memory:" << total_mem / 1024.0 / 1024.0 << "MB";
                    double free_mem_ratio =
                        static_cast<double>(free_mem) / static_cast<double>(total_mem);
                    if (free_mem_ratio < min_free_mem_ratio && i > 0) {
                        WUST_WARN("TRT") << "GPU memory is not enough!"
                                         << "Free GPU memory:" << free_mem_ratio * 100 << "%";
                        WUST_INFO("TRT") << "Cut remaining infer";
                        break;
                    }
                    infers.emplace_back(std::move(infer));
                    WUST_INFO("TRT") << "create execution context success"
                                     << "index:" << i;
                }
                return infers;
            };
            auto release_func = [&](std::unique_ptr<Infer>& resource) {
                if (resource) {
                    if (resource->cuda_infer) {
                        resource->cuda_infer.reset();
                    }
                }
            };
            auto restore_func = [&](size_t idx) -> std::unique_ptr<Infer> {
                auto infer = std::make_unique<Infer>();
                auto ctx = trt_net_->getAContext();
                infer->context = std::unique_ptr<nvinfer1::IExecutionContext>(ctx);
                if (use_cuda_pre_) {
                    infer->cuda_infer = std::make_unique<armor_cuda_infer::CudaInfer>();
                    infer->cuda_infer->init(
                        MAX_SRC_IMG_W,
                        MAX_SRC_IMG_H,
                        armor_infer_->inputW(),
                        armor_infer_->inputH()
                    );
                }
                if (!infer->context) {
                    WUST_ERROR("TRT") << "create infer failed, missing context";
                    return nullptr;
                }
                if ((use_cuda_pre_) && !infer->cuda_infer) {
                    WUST_ERROR("TRT") << "create infer failed, missing cuda_infer";
                    return nullptr;
                }
                return infer;
            };
            pool_params.restore_func = restore_func;

            pool_params.release_func = release_func;

            pool_params.can_restore = [&](size_t active_count) { return false; };

            pool_params.should_release = [&](size_t active_count) { return false; };

            pool_params.logger = [](const std::string& msg) {
                WUST_INFO("ArmorDetectorTrt:infer pool") << msg;
            };
            infer_pool_ =
                std::make_unique<wust_vl::common::concurrency::AdaptiveResourcePool<Infer>>(
                    pool_params
                );
        }

        ~Impl() {
            if (infer_pool_) {
                infer_pool_.reset();
            }
            trt_net_.reset();
            armor_detect_common_.reset();
        }

        void setCallback(DetectorCallback callback) {
            infer_callback_ = callback;
        }
        struct Tag {};
        void processCallback(
            const CommonFrame& frame,
            Infer* infer,
            const std::optional<ArmorNumber>& target_number
        ) const {
            std::vector<ArmorObject> armors;
            const auto t0 = wust_vl::common::utils::time_utils::now();
            Eigen::Matrix3f transform_matrix;
            std::vector<ArmorObject> objs_result;
            void* input_tensor_ptr;
            const cv::Mat roi = frame.img_frame.src_img(frame.expanded);

            cv::Mat resized_img;
            const float scale = armor_infer_->useNorm() ? 1.0f / 255.0f : 1.0f;
            const bool swap_rb = armor_infer_->inputRGB()
                != (frame.img_frame.pixel_format == wust_vl::video::PixelFormat::RGB);
            if (infer->cuda_infer && use_cuda_pre_) {
                input_tensor_ptr =
                    infer->cuda_infer
                        ->preprocess_pitched( //支持不连续内存,无需拷贝后输入可直接传roi的指针
                            roi.data,
                            roi.cols,
                            roi.rows,
                            roi.step,
                            scale,
                            swap_rb,
                            transform_matrix,
                            trt_net_->getStream()
                        );
                resized_img = infer->cuda_infer->tensorToMat( //nchw_float_to_hwc_uchar
                    static_cast<float*>(input_tensor_ptr),
                    armor_infer_->inputW(),
                    armor_infer_->inputH(),
                    scale,
                    trt_net_->getStream()
                );
            } else {
                resized_img = utils::letterbox(
                    roi,
                    transform_matrix,
                    armor_infer_->inputW(),
                    armor_infer_->inputH()
                );
                const cv::Mat blob = cv::dnn::blobFromImage(
                    resized_img,
                    scale,
                    cv::Size(armor_infer_->inputW(), armor_infer_->inputH()),
                    cv::Scalar(0, 0, 0),
                    swap_rb
                );
                trt_net_->input2Device(blob.ptr<float>());
                input_tensor_ptr = trt_net_->getInputTensorPtr();
            }
            const auto t1 = wust_vl::common::utils::time_utils::now();
            if (infer->context && input_tensor_ptr) {
                trt_net_->infer(input_tensor_ptr, infer->context.get());
            }
            const auto t2 = wust_vl::common::utils::time_utils::now();
            const cv::Mat
                output_mat(output_dims_.d[1], output_dims_.d[2], CV_32F, trt_net_->output2Host());
            cudaStreamSynchronize(trt_net_->getStream());
            objs_result = armor_infer_->postProcess(output_mat);
            const auto t3 = wust_vl::common::utils::time_utils::now();
            if (log_time_) {
                WUST_INFO("TRT") << std::fixed << std::setprecision(3) << "pre "
                                 << wust_vl::common::utils::time_utils::durationMs(t0, t1) << " "
                                 << "infer "
                                 << wust_vl::common::utils::time_utils::durationMs(t1, t2) << " "
                                 << "post "
                                 << wust_vl::common::utils::time_utils::durationMs(t2, t3) << " "
                                 << "total "
                                 << wust_vl::common::utils::time_utils::durationMs(t0, t3);
            }
            infer_pool_->release(infer);

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
            if (infer_pool_) {
                auto infer_ptr = infer_pool_->acquire();
                if (infer_ptr != nullptr) {
                    frame.id = current_id_++;
                    this->processCallback(frame, infer_ptr, target_number);
                }
            }
        }

    private:
        bool use_cuda_pre_;
        bool log_time_;
        nvinfer1::Dims input_dims_;
        nvinfer1::Dims output_dims_;
        DetectorCallback infer_callback_;
        std::unique_ptr<ArmorDetectorCommon> armor_detect_common_;
        std::unique_ptr<wust_vl::common::concurrency::AdaptiveResourcePool<Infer>> infer_pool_;
        std::unique_ptr<armor_infer::ArmorInfer> armor_infer_;
        int current_id_ = 0;
        std::unique_ptr<wust_vl::ml_net::TensorRTNet> trt_net_;
    };
    ArmorDetectorTrt::ArmorDetectorTrt(const YAML::Node& config, bool use_armor_detect_common) {
        _impl = std::make_unique<Impl>(config, use_armor_detect_common);
    }
    ArmorDetectorTrt::~ArmorDetectorTrt() {
        _impl.reset();
    }
    void ArmorDetectorTrt::setCallback(DetectorCallback callback) {
        _impl->setCallback(callback);
    }
    void ArmorDetectorTrt::pushInput(
        CommonFrame& frame,
        const std::optional<ArmorNumber>& target_number
    ) {
        _impl->pushInput(frame, target_number);
    }
} // namespace auto_aim
} // namespace awakening
#endif