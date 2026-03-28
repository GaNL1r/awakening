#pragma once
#include "number_classifier.hpp"
#ifdef USE_TRT
    #include "number_classifier_trt.hpp"
#endif
namespace awakening {
namespace auto_aim {
    class NumberClassifierFactory {
    public:
        static std::unique_ptr<NumberClassifierBase> createNumberClassifier(
            const std::string& backend,
            const std::string& classify_model_path,
            const std::string& classify_label_path
        ) {
#if defined(USE_TRT)
            if (backend == "tensorrt") {
                return std::make_unique<NumberClassifierTRT>(
                    classify_model_path,
                    classify_label_path
                );
            }
#endif

            if (backend == "opencv") {
                return std::make_unique<NumberClassifier>(classify_model_path, classify_label_path);
            }

            throw std::runtime_error(
                "Unsupported number classifier backend (or not compiled): " + backend
            );
        }
    };
} // namespace auto_aim
} // namespace awakening