#include "rune_aimer.hpp"
#include "tasks/auto_buff/rune_track/rune_target.hpp"
#include "tasks/base/ballistic_trajectory.hpp"
#include "utils/common/type_common.hpp"
#include "utils/logger.hpp"
namespace awakening::auto_buff {
struct RuneAimer::Impl {
    struct Params {
        double prediction_delay;
        void load(const YAML::Node& config) {
            prediction_delay = config["prediction_delay"].as<double>();
        }
    } params_;
    Impl(const YAML::Node& config) {
        params_.load(config);
        ballistic_trajectory_ = BallisticTrajectory::create(config["ballistic_trajectory"]);
        base_yaw_offset_rad_ = angles::from_degrees(config["base_yaw_offset"].as<double>());
        base_pitch_offset_rad_ = angles::from_degrees(config["base_pitch_offset"].as<double>());
    }
    struct HitCtx {
        RuneTarget hit_time_target;
        double fly_time;
    };
    struct ControlPoint {
        double yaw;
        double pitch;
        int aim_id;
        AimPoint aim_point;
        bool valid;
    };
    int get_select_id(const RuneTarget& target) const noexcept {
        return target.fan_wc.get_min_visable_fan_id();
    }
    [[nodiscard]] ControlPoint select_and_get_control_point(
        const RuneTarget& target,
        const ISO3& shoot_in_gimbal_odom,
        const ISO3& gimbal_in_gimbal_odom,
        double bullet_speed
    ) const noexcept {
        int id = get_select_id(target);
        auto fan_poses = target.get_target_state().get_fan_target_pose();
        return get_control_point(
            fan_poses[id],
            shoot_in_gimbal_odom,
            gimbal_in_gimbal_odom,
            bullet_speed,
            id
        );
    }
    [[nodiscard]] ControlPoint get_control_point(
        const ISO3& fan_pose,
        const ISO3& shoot_in_gimbal_odom,
        const ISO3& gimbal_in_gimbal_odom,
        double bullet_speed,
        int aim_id
    ) const noexcept {
        ControlPoint cp;

        auto p = fan_pose.translation() - shoot_in_gimbal_odom.translation();
        auto desired_pitch_opt = ballistic_trajectory_->solve_pitch(p, bullet_speed);
        if (!desired_pitch_opt) {
            cp.valid = false;
            AWAKENING_ERROR(
                "very_aimer: get_control_point: Failed to solve pitch armor_pos: [{}, {}, {}], bullet_speed: {}",
                p.x(),
                p.y(),
                p.z(),
                bullet_speed
            );
            return cp;
        }
        const auto [yaw_offset, pitch_offset] = get_yaw_pitch_offset();
        const double desired_control_yaw = std::atan2(p.y(), p.x());
        auto desired_shoot = utils::rpy2matrix(Vec3(
            0.0,
            desired_pitch_opt.value() + pitch_offset,
            angles::normalize_angle(desired_control_yaw + yaw_offset)
        ));
        auto R_gimbal_shoot =
            gimbal_in_gimbal_odom.linear().inverse() * shoot_in_gimbal_odom.linear();
        auto desired_gimbal = desired_shoot * R_gimbal_shoot.inverse();
        auto rpy = utils::matrix2rpy(desired_gimbal);
        cp.valid = true;
        cp.yaw = rpy[2];
        cp.pitch = rpy[1];
        cp.aim_point.pose = fan_pose;
        cp.aim_id = aim_id;
        return cp;
    };

    std::optional<HitCtx> get_hit(
        const RuneTarget& target_ready_to_aim,
        double bullet_speed,
        const ISO3& shoot_in_gimbal_odom
    ) const noexcept {
        auto hit_time_target = target_ready_to_aim;
        const int roughly_select = get_select_id(target_ready_to_aim);
        const auto __fan_target_pose = hit_time_target.get_target_state().get_fan_target_pose();
        auto prev_pitch_and_fly_time_opt = ballistic_trajectory_->solve_pitch_and_flytime(
            __fan_target_pose[roughly_select].translation() - shoot_in_gimbal_odom.translation(),
            bullet_speed
        );
        if (!prev_pitch_and_fly_time_opt) {
            return std::nullopt;
        }
        auto prev_fly_time = prev_pitch_and_fly_time_opt.value().second;

        for (int iter = 0; iter < 10; ++iter) {
            auto i_target = hit_time_target;
            i_target.set_target_state([&](motion_model::State& state) {
                state.predict(prev_fly_time, i_target.voter);
            });
            const auto iter_fan_target_pose = i_target.get_target_state().get_fan_target_pose();
            auto iter_pitch_and_fly_time_opt = ballistic_trajectory_->solve_pitch_and_flytime(
                iter_fan_target_pose[roughly_select].translation()
                    - shoot_in_gimbal_odom.translation(),
                bullet_speed
            );
            if (!iter_pitch_and_fly_time_opt) {
                return std::nullopt;
            }
            if (std::abs(iter_pitch_and_fly_time_opt.value().second - prev_fly_time) < 1e-3) {
                prev_fly_time = iter_pitch_and_fly_time_opt.value().second;
                break;
            }

            prev_fly_time = iter_pitch_and_fly_time_opt.value().second;
        }
        const double predict_time = prev_fly_time + params_.prediction_delay;
        hit_time_target.set_target_state([&](auto& state) {
            state.predict(predict_time, hit_time_target.voter);
        });
        return HitCtx {
            .hit_time_target = hit_time_target,
            .fly_time = prev_fly_time,
        };
    }
    GimbalCmd
    aim(const RuneTarget& _target,
        double bullet_speed,
        const ISO3& shoot_in_gimbal_odom,
        const ISO3& gimbal_in_gimbal_odom) const noexcept {
        GimbalCmd cmd;
        cmd.appear = false;
        auto target = _target;
        target.set_target_state([&](motion_model::State& state) {
            state.predict(Clock::now(), target.voter);
        });
        auto hit_ctx = get_hit(target, bullet_speed, shoot_in_gimbal_odom);
        if (!hit_ctx) {
            return cmd;
        }
        auto hit_time_target = hit_ctx->hit_time_target;
        auto cp = select_and_get_control_point(
            hit_time_target,
            shoot_in_gimbal_odom,
            gimbal_in_gimbal_odom,
            bullet_speed
        );
        if (!cp.valid) {
            return cmd;
        }
        cmd.timestamp = Clock::now();
        cmd.yaw = angles::to_degrees(cp.yaw);
        cmd.v_yaw = angles::to_degrees(0);
        cmd.a_yaw = angles::to_degrees(0);
        cmd.pitch = angles::to_degrees(cp.pitch);
        cmd.v_pitch = angles::to_degrees(0);
        cmd.a_pitch = angles::to_degrees(0);
        cmd.target_yaw = angles::to_degrees(cp.yaw);
        cmd.target_pitch = angles::to_degrees(cp.pitch);
        cmd.fly_time = hit_ctx->fly_time;
        cmd.appear = true;
        cmd.aim_point = cp.aim_point;
        cmd.aim_point.frame_id = target.get_target_state().frame_id;
        cmd.enable_pitch_diff = 1.0;
        cmd.enable_yaw_diff = 1.0;
        cmd.select_id = 0;
        return cmd;
    }
    void set_operator_offset(std::pair<double, double> offset) {
        operator_offset_ = offset;
    }
    std::pair<double, double> get_yaw_pitch_offset() const noexcept {
        return std::make_pair(
            base_yaw_offset_rad_ + operator_offset_.first, //操作手在线调偏置
            base_pitch_offset_rad_ + operator_offset_.second
        );
    }
    BallisticTrajectory::Ptr ballistic_trajectory_;
    double base_yaw_offset_rad_;
    double base_pitch_offset_rad_;
    std::pair<double, double> operator_offset_ = std::make_pair(0, 0);
};
RuneAimer::RuneAimer(const YAML::Node& config) {
    _impl = std::make_unique<Impl>(config);
}
RuneAimer::~RuneAimer() noexcept {}
GimbalCmd RuneAimer::aim(
    const RuneTarget& target,
    double bullet_speed,
    const ISO3& shoot_in_gimbal_odom,
    const ISO3& gimbal_in_gimbal_odom
) {
    return _impl->aim(target, bullet_speed, shoot_in_gimbal_odom, gimbal_in_gimbal_odom);
}
std::pair<double, double> RuneAimer::get_yaw_pitch_offset() {
    return _impl->get_yaw_pitch_offset();
}
void RuneAimer::set_operator_offset(std::pair<double, double> offset) {
    _impl->set_operator_offset(offset);
}
} // namespace awakening::auto_buff