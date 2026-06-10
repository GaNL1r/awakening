#pragma once

#include "motion_model.hpp"
#include "utils/common/type_common.hpp"
#include <optional>
namespace awakening::auto_buff {
struct RuneTrackerCfg {
    int esekf_iter_num;
    double lost_time_thres;
    int tracking_thres;
    Vec3 q_xyz;
    double q_yaw;
    double q_roll;
};
static inline int GOBAL_ID = 0; //全局状态标记，下游控制对同一id的不重复构建轨迹
class RuneTarget {
public:
    struct TrackState {
        enum State {
            LOST,
            DETECTING,
            TRACKING,
            TEMP_LOST,
        };
        State tracker_state = LOST;
        int detect_count = 0;
        int lost_count = 0;
        static inline std::string string_by_state(State state) {
            constexpr const char* details[] = { "LOST", "DETECTING", "TRACKING", "TEMP_LOST" };
            return std::string(details[state]);
        }
        bool is_tracking() const noexcept {
            return tracker_state == TRACKING || tracker_state == TEMP_LOST;
        }
        void reset() {
            tracker_state = LOST;
            detect_count = 0;
            lost_count = 0;
        }
    };
    RuneTarget() {}
    std::optional<motion_model::ESEKF> esekf;
    RuneTrackerCfg cfg;
    TimePoint last_update;
    TrackState track_state;
    int this_id = -1;
};
} // namespace awakening::auto_buff