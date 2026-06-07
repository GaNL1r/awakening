#pragma once
#include "utils/common/type_common.hpp"
#include <array>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <vector>
namespace awakening::auto_buff {
constexpr double RUNE_PAN_BOX_DIS = 0.16;
constexpr double RUNE_R2PANCENTER = 0.75;
struct RuneR {
    cv::RotatedRect rr;

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

struct RunePan {
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
            cv::line(img, corners[i], corners[(i + 1) % 4], cv::Scalar(255, 0, 0), 2);
        }

        cv::circle(img, center, 3, cv::Scalar(0, 0, 255), cv::FILLED);
    }
};
struct RuneDetection {
    TimePoint timestamp;
    int id = -1;
    int frame_id = -1;
    std::vector<RuneR> r_tags;
    std::vector<RunePan> pans;
    void add_offset(const cv::Point2f& offset) {
        for (auto& r_tag: r_tags) {
            r_tag.add_offset(offset);
        }
        for (auto& pan: pans) {
            pan.add_offset(offset);
        }
    }
    void draw(cv::Mat& img) const {
        for (const auto& r_tag: r_tags) {
            r_tag.draw(img);
        }
        for (const auto& pan: pans) {
            pan.draw(img);
        }
    }
};
} // namespace awakening::auto_buff