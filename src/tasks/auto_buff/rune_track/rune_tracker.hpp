#pragma once
#include "tasks/auto_buff/type.hpp"
#include "tasks/base/common.hpp"
#include "utils/impl.hpp"
#include <yaml-cpp/node/node.h>
namespace awakening::auto_buff {
class RuneTracker {
    RuneTracker(const YAML::Node& config);
    AWAKENING_IMPL_DEFINITION(RuneTracker)
    void track(
        RuneDetection& detection,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom,
        int frame_id
    );
};
} // namespace awakening::auto_buff