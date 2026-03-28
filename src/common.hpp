#pragma once
#include "wust_vl/common/utils/logger.hpp"
#include "wust_vl/video/icamera.hpp"
#include <Eigen/Dense>
#include <any>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
namespace awakening {
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using ISO3 = Eigen::Isometry3d;
using Quaternion = Eigen::Quaterniond;
using AngleAxis = Eigen::AngleAxisd;
using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;
enum class Frame : int { ODOM, GIMBAL_ODOM, GIMBAL, CAMERA, SHOOT };

enum class EnemyColor : bool {
    RED = 0,
    BLUE = 1,
};
struct CommonFrame {
    wust_vl::video::ImageFrame img_frame;
    int id;
    EnemyColor detect_color;
    cv::Rect expanded;
    cv::Point2f offset = cv::Point2f(0, 0);
    std::any any_ctx;
};
} // namespace awakening