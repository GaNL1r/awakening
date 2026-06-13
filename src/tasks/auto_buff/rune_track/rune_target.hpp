#pragma once

#include "motion_model.hpp"
#include "tasks/auto_buff/type.hpp"
#include "utils/common/type_common.hpp"
#include <optional>
#include <vector>
namespace awakening::auto_buff {
struct RuneTrackerCfg {
    int esekf_iter_num;
    double lost_time_thres;
    int tracking_thres;
    Vec3 q_xyz;
    double q_yaw;
    double q_roll;
    double r_uv_cv;
    double r_uv_net;
    double match_gate;
    void load(const YAML::Node& config) {
        esekf_iter_num = config["esekf_iter_num"].as<int>();
        lost_time_thres = config["lost_time_thres"].as<double>();
        tracking_thres = config["tracking_thres"].as<int>();
        auto q_xyz_vec = config["q_xyz"].as<std::vector<double>>();
        q_xyz = Vec3(q_xyz_vec[0], q_xyz_vec[1], q_xyz_vec[2]);
        q_yaw = config["q_yaw"].as<double>();
        q_roll = config["q_roll"].as<double>();
        r_uv_cv = config["r_uv_cv"].as<double>();
        r_uv_net = config["r_uv_net"].as<double>();
        match_gate = config["match_gate"].as<double>();
    }
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
        const TimePoint& timestamp,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom
    ) const noexcept;
    std::optional<cv::Point2f> match_r(
        std::vector<std::pair<int, RuneFanBladeWithR>>& matched_fans,
        std::vector<RuneR>& rs,
        const TimePoint& timestamp,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom
    );
    int update(
        std::vector<std::pair<int, RuneFanBladeWithR>>& f,
        std::optional<cv::Point2f>& r,
        const TimePoint& timestamp,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom
    );
    [[nodiscard]] cv::Rect get_net_focus_roi(
        const TimePoint& timestamp,
        const ISO3& camera_cv_in_odom,
        const CameraInfo& camera_info,
        const cv::Size& image_size,
        double target_wh_ratio = 1.0
    ) const noexcept;

    const motion_model::State& get_target_state() const {
        return target_state;
    }
    template<typename F>
    void set_target_state(F&& f) {
        this_id = GOBAL_ID++; //全局状态标记，下游控制对同一id的不重复构建轨迹
        f(target_state);
    }
    [[nodiscard]] inline RuneTarget fast_copy_without_ekf() const noexcept {
        RuneTarget target;
        target.target_state = this->target_state;
        target.last_update = this->last_update;
        target.cfg = this->cfg;
        target.track_state = this->track_state;
        target.is_inited = this->is_inited;
        target.this_id = this->this_id;
        target.visable_fan_ids = this->visable_fan_ids;
        return target;
    }
    [[nodiscard]] inline bool check() const noexcept {
        auto v = track_state.is_tracking()
            && std::chrono::duration<double>(Clock::now() - last_update).count()
                < cfg.lost_time_thres;
        return v;
    }
    [[nodiscard]] inline bool need_focus() const noexcept {
        return is_inited
            && std::chrono::duration<double>(Clock::now() - last_update).count()
            < cfg.lost_time_thres;
    }
    std::vector<int> get_visable_fan_ids() const noexcept {
        return visable_fan_ids;
    }
    void write_log();
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
    std::vector<int> visable_fan_ids;

private:
    motion_model::State target_state;
};
} // namespace awakening::auto_buff