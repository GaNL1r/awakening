#pragma once

#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "utils/common/type_common.hpp"
#include <optional>
#include <vector>
namespace awakening::vslam {
struct Feature {
    struct Detected {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };

    std::optional<Detected> detected;
    cv::Mat src;
    TimePoint timestamp;
    int id;
    int frame_id;
};
struct Match {
    std::vector<cv::DMatch> matches;
    TimePoint timestamp;
    int seq;
    int frame_id;
};
} // namespace awakening::vslam