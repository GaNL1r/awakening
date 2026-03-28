#pragma once
#include "common.hpp"
#include "utils/utils.hpp"
#include <array>
#include <cstddef>
#include <opencv2/core/types.hpp>
#include <optional>
#include <utility>
namespace awakening::auto_aim {
constexpr double SIMPLE_SMALL_ARMOR_WIDTH = 133.0 / 1000.0; // 135
constexpr double SIMPLE_SMALL_ARMOR_HEIGHT = 50.0 / 1000.0; // 55
constexpr double LARGE_ARMOR_WIDTH = 225.0 / 1000.0;
constexpr double LARGE_ARMOR_HEIGHT = 50.0 / 1000.0; // 55
enum class ArmorColor : int { BLUE = 0, RED, NONE, PURPLE };
constexpr std::string getStringByArmorColor(const ArmorColor& armor_class) {
    constexpr std::array details { "blue", "red", "none", "purple" };
    return details[std::to_underlying(armor_class)];
}
enum class ArmorClass : int { SENTRY = 0, NO1, NO2, NO3, NO4, NO5, OUTPOST, BASE, UNKNOWN };
constexpr int getArmorNumByArmorClass(const ArmorClass& armor_class) {
    constexpr std::array details { 4, 4, 4, 4, 4, 4, 3, 4, 4 };
    return details[std::to_underlying(armor_class)];
}
constexpr std::string getStringByArmorClass(const ArmorClass& armor_class) {
    constexpr std::array details { "sentry", "no1",     "no2",  "no3",    "no4",
                                   "no5",    "optpost", "base", "unknown" };
    return details[std::to_underlying(armor_class)];
}
enum class ArmorType { SMALL, LARGE, INVALID };
struct Light: public cv::RotatedRect {
    Light() = default;

    explicit Light(const std::vector<cv::Point>& contour);

    void addOffset(const cv::Point2f& offset) noexcept {
        this->center += offset;
        top += offset;
        bottom += offset;
    }
    void transform(const Eigen::Matrix<float, 3, 3>& transform_matrix) noexcept {
        top = utils::transformPoint2D(transform_matrix, top);
        bottom = utils::transformPoint2D(transform_matrix, bottom);
        length = cv::norm(top - bottom);
        cv::Point2f p[4];
        this->points(p);

        width = cv::norm(
            utils::transformPoint2D(transform_matrix, p[0])
            - utils::transformPoint2D(transform_matrix, p[1])
        );
        const cv::Point2f p0 = center;
        const cv::Point2f p1 = center + axis;

        const cv::Point2f p0_t = utils::transformPoint2D(transform_matrix, p0);

        const cv::Point2f p1_t = utils::transformPoint2D(transform_matrix, p1);

        axis = p1_t - p0_t;
        axis /= cv::norm(axis);

        tilt_angle =
            std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y)) / CV_PI * 180.0f;
        center = utils::transformPoint2D(transform_matrix, center);
    }

    cv::Point2f top, bottom;
    int color = 0;
    cv::Point2f axis;
    double length = 0;
    double width = 0;
    float tilt_angle = 0;
};

enum class ArmorKeyPointsIndex : int {
    RIGHT_BOTTOM,
    RIGHT_MID,
    RIGHT_TOP,
    LEFT_TOP,
    LEFT_MID,
    LEFT_BOTTOM,
    N
};
namespace armor_keypoints {
    using I = ArmorKeyPointsIndex;
    constexpr std::array sys_pairs = {

        std::pair { std::to_underlying(I::RIGHT_BOTTOM), std::to_underlying(I::LEFT_BOTTOM) },
        std::pair { std::to_underlying(I::RIGHT_MID), std::to_underlying(I::LEFT_MID) },
        std::pair { std::to_underlying(I::RIGHT_MID), std::to_underlying(I::LEFT_TOP) }
    };
} // namespace armor_keypoints

template<typename PointT>
struct ArmorKeyPoints2D {
    using I = ArmorKeyPointsIndex;
    void addOffset(const PointT& offset) noexcept {
        for (auto& p: points) {
            if (p.has_value()) {
                p.value() += offset;
            }
        }
    }
    void transform(const Eigen::Matrix<float, 3, 3>& transform_matrix) noexcept {
        for (auto& p: points) {
            if (p.has_value()) {
                p.value() = utils::transformPoint2D(transform_matrix, p.value());
            }
        }
    }
    std::array<PointT, 6> getPoints() noexcept {
        if (!points[I::RIGHT_MID].has_value()) {
            points[I::RIGHT_MID] = (points[I::RIGHT_BOTTOM] + points[I::RIGHT_TOP]) / 2.0;
        }
        if (!points[I::LEFT_MID].has_value()) {
            points[I::LEFT_MID] = (points[I::LEFT_BOTTOM] + points[I::LEFT_TOP]) / 2.0;
        }
        for (const auto& p: points) {
            if (!p.has_value()) {
                throw std::runtime_error("ArmorKeyPoints2D::points(): one of the points is not set"
                );
            }
        }
        return points;
    }
    std::array<std::optional<PointT>, 6> points;
};
template<typename PointT, double W, double H>
struct ArmorKeyPoint3D {
    using I = ArmorKeyPointsIndex;
    static constexpr std::array points = {
        PointT(0, W / 2, -H / 2), // 右下
        PointT(0, W / 2, 0.0), // 右中
        PointT(0, W / 2, H / 2), // 右上

        PointT(0, -W / 2, H / 2), // 左上
        PointT(0, -W / 2, 0.0), // 左中
        PointT(0, -W / 2, -H / 2) // 左下
    };
};
struct Armor {
    ArmorColor color = ArmorColor::NONE;
    ArmorClass number = ArmorClass::UNKNOWN;
    ArmorKeyPoints2D<cv::Point2f> key_points;
    std::chrono::steady_clock::time_point timestamp;
    int id = -1;
    int frame_id = -1;
    ISO3 pose;
    struct NetCtx {
        double confidence = 0;
        ArmorColor color = ArmorColor::NONE;
        ArmorClass number = ArmorClass::UNKNOWN;
        ArmorKeyPoints2D<cv::Point2f> key_points;
    } net;
    struct ClassifierCtx {
        ArmorClass number = ArmorClass::UNKNOWN;
        cv::Mat number_img;
        double confidence = 0;
    } classifier;
    struct CvCtx {
        ArmorColor color = ArmorColor::NONE;
        ArmorType type = ArmorType::INVALID;
        cv::Mat whole_binary_img;
        cv::Mat whole_rgb_img;
        cv::Mat whole_gray_img;
        std::vector<Light> tmp_lights;
        Light left_light, right_light;
        bool is_valid = false;
    } cv;
    bool isBig() const noexcept {
        return number == ArmorClass::NO1;
    }
    Armor() = default;
};

} // namespace awakening::auto_aim