#include "rune_tracker.hpp"
namespace awakening::auto_buff {
struct RuneTracker::Impl {
    Impl(const YAML::Node& config) {}
    void track(
        RuneDetection& detection,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom,
        int frame_id
    ) {}
};
} // namespace awakening::auto_buff