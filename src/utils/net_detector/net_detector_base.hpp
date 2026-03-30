#pragma once
#include "common.hpp"
#include <opencv2/core/mat.hpp>
#include <optional>
namespace awakening::utils {

class NetDetectorBase {
public:
    using Ptr = std::unique_ptr<NetDetectorBase>;
    enum class PixelFormat : int {
        BGR = 0,
        GRAY,
        RGB,
    };
    virtual cv::Mat detect(const cv::Mat& img, PixelFormat format) = 0;
    virtual ~NetDetectorBase() = default;
};
} // namespace awakening::utils