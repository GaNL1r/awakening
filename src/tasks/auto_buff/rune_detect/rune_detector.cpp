#include "rune_detector.hpp"
#include "rune_infer.hpp"
#include <memory>
#include <opencv2/core/mat.hpp>
#include <vector>
#if USE_OPENVINO
    #include "utils/net_detector/openvino/net_detector_openvino.hpp"
#endif
#ifdef USE_TRT
    #include "utils/net_detector/tensorrt/net_detector_tensorrt.hpp"
#endif

namespace awakening::auto_buff {
struct RuneDetector::Impl {
    static constexpr const char* OPENVINO = "openvino";
    static constexpr const char* TENSORRT = "tensorrt";
    struct Params {
        void load(const YAML::Node& config) {}
    } params_;
    Impl(const YAML::Node& config) {
        params_.load(config);
        rune_infer_ = RuneInfer::create(config["rune_infer"]);
        auto backend = config["net_detector"]["backend"].as<std::string>();
        const double scale = rune_infer_->use_norm() ? 1.0 / 255.0f : 1.0f;
        auto format = rune_infer_->target_format();
        auto net_cfg = utils::NetDetectorBase::Config {
            .target_format = format,
            .preprocess_scale = scale,
            .target_w = rune_infer_->input_w(),
            .target_h = rune_infer_->input_h(),
        };
        bool backend_valid = false;
#ifdef USE_OPENVINO
        if (backend == OPENVINO) {
            backend_valid = true;
            net_detector_ = std::make_unique<utils::NetDetectorOpenVINO>(
                config["net_detector"][OPENVINO],
                net_cfg
            );
        }
#endif
#ifdef USE_TRT
        if (backend == TENSORRT) {
            backend_valid = true;
            net_detector_ = std::make_unique<utils::NetDetectorTensorrt>(
                config["net_detector"][TENSORRT],
                net_cfg
            );
        }
#endif
        if (backend == "opencv") {
            backend_valid = true;
            net_detector_ = nullptr;
        }
        if (!backend_valid) {
            throw std::runtime_error("Invalid backend");
        }
    }

    RuneDetection
    detect(const CommonFrame& frame, const cv::Rect& focus, EnemyColor enemy_color) const noexcept {
        RuneDetection result;
        cv::Mat roi = frame.img_frame.src_img(focus);

        utils::NetDetectorBase::OutPut net_output;
        net_output = net_detector_->detect(roi, frame.img_frame.format);
        result.fan_blades = rune_infer_->process(net_output.output);
        for (auto& fan_blade: result.fan_blades) {
            fan_blade.transform(net_output.transform_matrix);
            fan_blade.add_offset(focus.tl());
        }
        return result;
    }
    utils::NetDetectorBase::Ptr net_detector_;
    RuneInfer::Ptr rune_infer_;
};
RuneDetector::RuneDetector(const YAML::Node& config) {
    _impl = std::make_unique<Impl>(config);
}
RuneDetector::~RuneDetector() noexcept {
    _impl.reset();
}
RuneDetection
RuneDetector::detect(const CommonFrame& frame, const cv::Rect& focus, EnemyColor enemy_color) {
    return _impl->detect(frame, focus, enemy_color);
}
} // namespace awakening::auto_buff