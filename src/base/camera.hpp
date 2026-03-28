#pragma once
#include "utils/utils.hpp"
#include "wust_vl/video/video.hpp"
#include <memory>
#include <wust_vl/video/icamera.hpp>
namespace awakening {
template<typename CTX>
class Camera {
public:
    using Ptr = std::shared_ptr<Camera>;
    Camera() {}
    static Ptr create() {
        return std::make_shared<Camera>();
    }
    template<typename LOADCTX>
    void load(
        const YAML::Node& config,
        wust_vl::video::ICameraDevice::FrameCallback callback,
        const LOADCTX& load_ctx
    ) {
        load_ctx(config, ctx_);
        load(config, callback);
    }

    void load(const YAML::Node& config, wust_vl::video::ICameraDevice::FrameCallback callback) {
        device_ = std::make_shared<wust_vl::video::Camera>();
        device_->init(config);
        device_->setFrameCallback(callback);
        std::string camera_info_path =
            utils::expandEnv(config["camera_info_path"].as<std::string>());
        YAML::Node config_camera_info = YAML::LoadFile(camera_info_path);
        std::vector<double> camera_k =
            config_camera_info["camera_matrix"]["data"].as<std::vector<double>>();
        std::vector<double> camera_d =
            config_camera_info["distortion_coefficients"]["data"].as<std::vector<double>>();

        assert(camera_k.size() == 9);
        assert(camera_d.size() == 5);

        cv::Mat K(3, 3, CV_64F);
        std::memcpy(K.data, camera_k.data(), 9 * sizeof(double));

        cv::Mat D(1, 5, CV_64F);
        std::memcpy(D.data, camera_d.data(), 5 * sizeof(double));

        camera_info_ = std::make_pair(K.clone(), D.clone());
    }
    std::pair<cv::Mat, cv::Mat> camera_info_;
    std::shared_ptr<wust_vl::video::Camera> device_;
    CTX ctx_;
};
} // namespace awakening