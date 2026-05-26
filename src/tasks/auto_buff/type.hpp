#pragma once
#include <array>
#include <opencv2/core/types.hpp>
#include <optional>
#include <vector>
namespace awakening::auto_buff {
constexpr double RUNE_PAN_BOX_DIS = 0.16;
constexpr double RUNE_R2PANCENTER = 0.75;
struct RuneDetection {
    struct RunePan {
        std::array<cv::Point2f, 4> corners;
        cv::Point2f center;
    };
    std::optional<cv::Point2f> r_tag;
    std::vector<RunePan> pans;
};
} // namespace awakening::auto_buff