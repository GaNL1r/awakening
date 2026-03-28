// Copyright 2025 XiaoJian Wu
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
#pragma once
#include "armor_detector_base.hpp"
#include "config.hpp"
#include <string>
#include <yaml-cpp/yaml.h>

#ifdef USE_OPENVINO
    #include "tasks/auto_aim/armor_detect/openvino/armor_detector_openvino.hpp"
#endif
#ifdef USE_TRT
    #include "tasks/auto_aim/armor_detect/tensorrt/armor_detector_tensorrt.hpp"
#endif
#ifdef USE_NCNN
    #include "tasks/auto_aim/armor_detect/ncnn/armor_detector_ncnn.hpp"
#endif
#ifdef USE_ORT
    #include "tasks/auto_aim/armor_detect/onnxruntime/armor_detector_onnxruntime.hpp"
#endif
#include "tasks/auto_aim/armor_detect/opencv/armor_detector_opencv.hpp"

namespace awakening {
namespace auto_aim {

    class DetectorFactory {
    public:
        static ArmorDetectorBase::Ptr createArmorDetector(
            const std::string& backend,
            bool use_armor_detect_common,
            std::string cv_config_path = OPENCV_CONFIG,
            std::string ml_config_path = ML_CONFIG
        ) {
            // 检查编译时是否支持
            auto isBackendEnabled = [&backend]() -> bool {
#ifdef USE_OPENVINO
                if (backend == "openvino")
                    return true;
#endif
#ifdef USE_TRT
                if (backend == "tensorrt")
                    return true;
#endif
#ifdef USE_NCNN
                if (backend == "ncnn")
                    return true;
#endif
#ifdef USE_ORT
                if (backend == "onnxruntime")
                    return true;
#endif
                if (backend == "opencv")
                    return true;
                return false;
            };

            if (!isBackendEnabled()) {
                std::cout << "Backend " << backend << " is not enabled at compile time."
                          << std::endl;
                throw std::runtime_error("Backend " + backend + " is not enabled at compile time.");
            }

            auto getConfigPath = [&](const std::string& backend) -> std::string {
                if (backend == "opencv")
                    return cv_config_path;
                else
                    return ml_config_path;
            };

            std::string config_path = getConfigPath(backend);
            if (config_path.empty()) {
                std::cout << "No config path for backend: " << backend << std::endl;
                throw std::runtime_error("No config path for backend: " + backend);
            }

            YAML::Node armor_detect_config = YAML::LoadFile(config_path);

            // 创建对应后端实例
#if defined(USE_OPENVINO)
            if (backend == "openvino") {
                return ArmorDetectorOpenVino::create(
                    armor_detect_config["armor_detector"],
                    use_armor_detect_common
                );
            }
#endif
#if defined(USE_TRT)
            if (backend == "tensorrt") {
                return ArmorDetectorTrt::create(
                    armor_detect_config["armor_detector"],
                    use_armor_detect_common
                );
            }
#endif
#if defined(USE_NCNN)
            if (backend == "ncnn") {
                return ArmorDetectorNCNN::create(
                    armor_detect_config["armor_detector"],
                    use_armor_detect_common
                );
            }
#endif
#if defined(USE_ORT)
            if (backend == "onnxruntime") {
                return ArmorDetectorOnnxRuntime::create(
                    armor_detect_config["armor_detector"],
                    use_armor_detect_common
                );
            }
#endif
            if (backend == "opencv") {
                return ArmorDetectorOpenCV::create(armor_detect_config["armor_detector"]);
            }
            std::cout << "Unsupported armor detector backend (or not compiled): " << backend
                      << std::endl;
            throw std::runtime_error(
                "Unsupported armor detector backend (or not compiled): " + backend
            );
        }
    };

} // namespace auto_aim
} // namespace awakening
