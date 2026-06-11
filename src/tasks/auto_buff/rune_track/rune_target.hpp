#pragma once

#include "motion_model.hpp"
#include "tasks/auto_buff/type.hpp"
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
    void reset(
        RuneFanBladeWithR& r,
        const RuneTrackerCfg& c,
        const TimePoint& timestamp,
        int frame_id,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom
    );
    static void fan_pnp(
        RuneFanBladeWithR& a,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom
    ) noexcept;
    [[nodiscard]] Eigen::Matrix<double, motion_model::X_N, motion_model::X_N>
    process_noise(double dt) const noexcept;
    [[nodiscard]] Eigen::Matrix<double, motion_model::YPDZ_N, motion_model::YPDZ_N>
    ypdmeasurement_covariance(const Eigen::Matrix<double, motion_model::YPDZ_N, 1>& z
    ) const noexcept;
    [[nodiscard]] Eigen::Matrix<double, motion_model::YPDZ_N, 1>
    get_ypdmeasurement(RuneFanBladeWithR& fan) const noexcept;
    void predict_ekf(const TimePoint& timestamp);
    std::vector<std::pair<int, RuneFanBladeWithR>> match_fan(
        std::vector<RuneFanBladeWithR>& fans,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom
    ) const noexcept;
    int update(
        std::vector<std::pair<int, RuneFanBladeWithR>>& f,
        const TimePoint& timestamp,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom
    );
    std::optional<motion_model::ESEKF> esekf;
    motion_model::RMeasure::Ctx r_ctx;
    motion_model::FanBladeMeasure::Ctx fan_ctx;
    motion_model::YPDMeasure::Ctx ypd_ctx;
    RuneTrackerCfg cfg;
    TimePoint last_update;
    TrackState track_state;
    int this_id = -1;
    bool is_inited = false;
    mutable double last_rot_yaw = 0;
    mutable double last_rot_roll = 0;

private:
    motion_model::State target_state;
};
} // namespace awakening::auto_buff