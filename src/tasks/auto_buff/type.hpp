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
constexpr double RUNE_PAN_BOX_DIS = 0.15;
constexpr double RUNE_PAN_R = 0.115;
constexpr double RUNE_R2PANCENTER = 0.70;
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
            PointT(0, 0, RUNE_R2PANCENTER + RUNE_PAN_R), // 上
            PointT(0, RUNE_PAN_R, RUNE_R2PANCENTER), // 左
            PointT(0, 0, RUNE_R2PANCENTER - RUNE_PAN_R), // 下
            PointT(0, -RUNE_PAN_R, RUNE_R2PANCENTER), // 右
            PointT(0, 0, RUNE_R2PANCENTER), // 中
            PointT(0, 0, 0),
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
struct RuneDetection {
    TimePoint timestamp;
    int id = -1;
    int frame_id = -1;

    std::vector<RuneFanBladeWithR> fan_blades;
    void draw(cv::Mat& img) const {
        for (const auto& fan_blade: fan_blades) {
            fan_blade.draw(img);
        }
    }
};
} // namespace awakening::auto_buff