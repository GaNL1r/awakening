#pragma once
#include "utils/common/type_common.hpp"
#include "utils/utils.hpp"
#include <array>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <vector>
namespace awakening::auto_buff {
constexpr double RUNE_FAN_TARGET_BOX_DIS = 0.15;
constexpr double RUNE_FAN_TARGET_R = 0.115;
constexpr double RUNE_R2_FAN_TARGET_CENTER = 0.70;
constexpr double FUCK = 0.015;
constexpr int FAN_NUM = 5;
enum class RuneColor : int {
    RED = 0,
    BLUE = 1,
    NONE = -1,
};
enum RuneKeyPointsIndex { TOP, LEFT, BOTTOM, RIGHT, CENTER, R, N };

template<typename PointT>
struct RuneKeyPoint3D {
    inline static std::vector<PointT> build() {
        return {
            PointT(0, 0, RUNE_R2_FAN_TARGET_CENTER + RUNE_FAN_TARGET_R), // 上
            PointT(0, RUNE_FAN_TARGET_R, RUNE_R2_FAN_TARGET_CENTER), // 左
            PointT(0, 0, RUNE_R2_FAN_TARGET_CENTER - RUNE_FAN_TARGET_R), // 下
            PointT(0, -RUNE_FAN_TARGET_R, RUNE_R2_FAN_TARGET_CENTER), // 右
            PointT(0, 0, RUNE_R2_FAN_TARGET_CENTER), // 中
            PointT(0, 0, 0),
        };
    }
    inline static std::vector<PointT> build_no_r() {
        return {
            PointT(0, 0, RUNE_R2_FAN_TARGET_CENTER + RUNE_FAN_TARGET_R), // 上
            PointT(0, RUNE_FAN_TARGET_R, RUNE_R2_FAN_TARGET_CENTER), // 左
            PointT(0, 0, RUNE_R2_FAN_TARGET_CENTER - RUNE_FAN_TARGET_R), // 下
            PointT(0, -RUNE_FAN_TARGET_R, RUNE_R2_FAN_TARGET_CENTER), // 右
            PointT(0, 0, RUNE_R2_FAN_TARGET_CENTER), // 中
        };
    }
};

struct RuneFanBladeWithR {
    std::vector<cv::Point2f> points;
    std::vector<std::vector<cv::Point2f>> tmp_points;
    ISO3 pose;
    cv::Rect2f bbox;
    RuneColor color = RuneColor::NONE;
    double confidence = 0;
    void draw(cv::Mat& img) const {
        for (int i = 0; i < points.size(); ++i) {
            cv::circle(img, points[i], 3, cv::Scalar(0, 255, 0), cv::FILLED);
            // cv::putText(
            //     img,
            //     std::to_string(i),
            //     points[i],
            //     cv::FONT_HERSHEY_COMPLEX,
            //     0.5,
            //     cv::Scalar(0, 255, 0)
            // );
        }
        // cv::rectangle(img, bbox, cv::Scalar(0, 255, 0), 2);
    }
    void add_offset(const cv::Point2f& offset) {
        for (auto& point: points) {
            point += offset;
        }
        bbox += offset;
    }
    void transform(const Eigen::Matrix<float, 3, 3>& transform_matrix) noexcept {
        for (auto& point: points) {
            point = utils::transform_point2D(transform_matrix, point);
        }
        bbox = utils::transform_rect(transform_matrix, bbox);
    }
};
struct RuneR {
    cv::RotatedRect rr;
    bool laji = true;
    void add_offset(const cv::Point2f& offset) {
        rr.center += offset;
    }
    void draw(cv::Mat& img) const {
        cv::Point2f vertices[4];
        rr.points(vertices);
        for (int i = 0; i < 4; ++i) {
            cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        cv::circle(img, rr.center, 3, cv::Scalar(0, 0, 255), cv::FILLED);
    }
};
struct RuneFanTarget {
    std::array<cv::Point2f, 4> corners;
    cv::Point2f center;

    void add_offset(const cv::Point2f& offset) {
        center += offset;
        for (auto& corner: corners) {
            corner += offset;
        }
    }

    void draw(cv::Mat& img) const {
        for (int i = 0; i < 4; ++i) {
            cv::circle(img, corners[i], 3, cv::Scalar(255, 0, 0), cv::FILLED);
        }
        cv::circle(img, center, 3, cv::Scalar(0, 0, 255), cv::FILLED);
    }
};

struct RuneDetection {
    TimePoint timestamp;
    int id = -1;
    int frame_id = -1;

    std::vector<RuneFanBladeWithR> fan_blades;
    std::vector<RuneR> rune_rs;
    std::vector<RuneFanTarget> fan_targets;
    void draw(cv::Mat& img) const {
        for (const auto& fan_blade: fan_blades) {
            fan_blade.draw(img);
        }
        for (const auto& rune_r: rune_rs) {
            rune_r.draw(img);
        }
        for (const auto& fan_target: fan_targets) {
            fan_target.draw(img);
        }
    }
};
} // namespace awakening::auto_buff