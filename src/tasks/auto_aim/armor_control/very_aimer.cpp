#include "very_aimer.hpp"
#include "angles.h"
#include "tasks/base/ballistic_trajectory.hpp"
#include "tasks/base/dta_utils.hpp"
#include "utils/common/type_common.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"
#include <cmath>
#include <cstdlib>
#include <deque>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>
namespace awakening::auto_aim {
struct VeryAimer::Impl {
    struct Params {
        double sample_total_time;
        int sample_horizon;
        double fire_delay_min;
        double fire_delay_max;
        double max_yaw_acc;
        double max_pitch_acc;
        double prediction_delay;
        double aim_center_more_prediction_time;
        double shooting_range_h;
        double shooting_range_w_small;
        double shooting_range_w_large;
        double min_enable_pitch_deg;
        double min_enable_yaw_deg;

        void load(const YAML::Node& config) {
            sample_total_time = config["sample_total_time"].as<double>();
            sample_horizon = config["sample_horizon"].as<int>();
            fire_delay_min = config["fire_delay_min"].as<double>();
            fire_delay_max = config["fire_delay_max"].as<double>();
            max_yaw_acc = config["max_yaw_acc"].as<double>();
            max_pitch_acc = config["max_pitch_acc"].as<double>();
            prediction_delay = config["prediction_delay"].as<double>();
            aim_center_more_prediction_time =
                config["aim_center_more_prediction_time"].as<double>();
            shooting_range_h = config["shooting_range_h"].as<double>();
            shooting_range_w_small = config["shooting_range_w_small"].as<double>();
            shooting_range_w_large = config["shooting_range_w_large"].as<double>();
            min_enable_pitch_deg = config["min_enable_pitch_deg"].as<double>();
            min_enable_yaw_deg = config["min_enable_yaw_deg"].as<double>();
        }
    } params_;
    Impl(const YAML::Node& config) {
        params_.load(config);
        ballistic_trajectory_ = BallisticTrajectory::create(config["ballistic_trajectory"]);
        base_yaw_offset_rad_ = angles::from_degrees(config["base_yaw_offset"].as<double>());
        base_pitch_offset_rad_ = angles::from_degrees(config["base_pitch_offset"].as<double>());
    }
    [[nodiscard]] int
    select_armor(const ArmorTarget& target, const AutoAimFsm& auto_aim_fsm) const noexcept {
        static int lock_id = -1;
        const auto target_state = target.get_target_state();
        const auto armors_xyza = target_state.get_armors_xyza(target.target_number);
        const int armor_num = static_cast<int>(armors_xyza.size());
        int i_chosen = 0;

        // const double center_yaw = std::atan2(target_state.pos().y(), target_state.pos().x());
        std::vector<double> delta_angles;
        delta_angles.reserve(armor_num);
        for (int i = 0; i < armor_num; ++i) {
            delta_angles.push_back(angles::normalize_angle(
                armors_xyza[i][3] - std::atan2(armors_xyza[i].y(), armors_xyza[i].x())
            ));
        }
        const auto pick_best_by_min_delta = [&](const std::vector<int>& idxs) -> int {
            int best = -1;
            double best_val = std::numeric_limits<double>::infinity();
            for (int i: idxs) {
                const double val = std::abs(delta_angles[i]);
                if (val < best_val) {
                    best_val = val;
                    best = i;
                }
            }
            return best;
        };

        if (auto_aim_fsm == AutoAimFsm::AIM_SINGLE_ARMOR
            && target.target_number != ArmorClass::OUTPOST && armor_num > 0)
        {
            constexpr double in_first = 60.0 / 57.3;
            std::vector<int> candidates;
            for (int i = 0; i < armor_num; ++i) {
                if (std::abs(delta_angles[i]) <= in_first)
                    candidates.push_back(i);
            }
            if (!candidates.empty()) {
                int pick = -1;

                if (candidates.size() == 1) {
                    pick = candidates[0];
                    lock_id = -1;
                } else {
                    if (lock_id < 0 || (lock_id != candidates[0] && lock_id != candidates[1])) {
                        lock_id = (std::abs(delta_angles[candidates[0]])
                                   < std::abs(delta_angles[candidates[1]]))
                            ? candidates[0]
                            : candidates[1];
                    }
                    pick = lock_id;
                }

                if (pick >= 0 && pick < armor_num) {
                    i_chosen = pick;
                }
            }

            return i_chosen;
        }
        if (armor_num > 0) {
            int best_idx = -1;

            if (auto_aim_fsm
                    == AutoAimFsm::AIM_WHOLE_CAR_PAIR //4选2,本质提升控制轨迹与目标轨迹重合窗口
                && target.target_number != ArmorClass::OUTPOST)
            {
                std::vector<int> all;
                if (target_state.h() < 0) { //上边的装甲板没准能碰巧到下面的？
                    all.push_back(1);
                    all.push_back(3);
                } else {
                    all.push_back(0);
                    all.push_back(2);
                }
                best_idx = pick_best_by_min_delta(all);
            }
            if (best_idx < 0) {
                std::vector<int> all(armor_num);
                std::iota(all.begin(), all.end(), 0);
                best_idx = pick_best_by_min_delta(all);
            }

            i_chosen = best_idx;
        }

        return i_chosen;
    }
    [[nodiscard]] dta_utils::ControlPoint get_control_point(
        const Vec4& armor_xyza,
        const ISO3& shoot_in_gimbal_odom,
        const ISO3& gimbal_in_gimbal_odom,
        double bullet_speed,
        int aim_id
    ) const noexcept {
        dta_utils::ControlPoint cp;

        auto p = armor_xyza.head<3>() - shoot_in_gimbal_odom.translation();
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
        cp.aim_point.pose = ISO3::Identity();
        cp.aim_point.pose.translation() = armor_xyza.head<3>();
        cp.aim_point.d_angle = armor_xyza[3];
        cp.aim_id = aim_id;
        return cp;
    };
    [[nodiscard]] dta_utils::ControlPoint select_and_get_control_point(
        const ArmorTarget& target,
        double bullet_speed,
        const AutoAimFsm& fsm,
        const ISO3& shoot_in_gimbal_odom,
        const ISO3& gimbal_in_gimbal_odom
    ) const noexcept {
        const int selected_armor = select_armor(target, fsm);
        auto armors_xyza = target.get_target_state().get_armors_xyza(target.target_number);
        if (fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER) { //瞄准中间把目标推回原来板子的位置
            auto p = target.get_target_state().pos() - shoot_in_gimbal_odom.translation();
            double center_xy_dis = std::hypot(p.x(), p.y());
            double center_yaw = std::atan2(p.y(), p.x());
            center_xy_dis -=
                target.get_target_state().get_armor_r(selected_armor, target.target_number);
            armors_xyza[selected_armor].x() = center_xy_dis * std::cos(center_yaw);
            armors_xyza[selected_armor].y() = center_xy_dis * std::sin(center_yaw);
        }
        return get_control_point(
            armors_xyza[selected_armor],
            shoot_in_gimbal_odom,
            gimbal_in_gimbal_odom,
            bullet_speed,
            selected_armor
        );
    }
    struct HitCtx {
        ArmorTarget hit_time_target;
        double fly_time;
    };
    std::optional<HitCtx> get_hit(
        const ArmorTarget& target_ready_to_aim,
        double bullet_speed,
        const AutoAimFsm& fsm,
        const ISO3& shoot_in_gimbal_odom
    ) const noexcept {
        auto hit_time_target = target_ready_to_aim;
        const int roughly_select = select_armor(hit_time_target, fsm);
        const auto __armors_xyza =
            hit_time_target.get_target_state().get_armors_xyza(hit_time_target.target_number);
        auto prev_pitch_and_fly_time_opt = ballistic_trajectory_->solve_pitch_and_flytime(
            __armors_xyza[roughly_select].head<3>() - shoot_in_gimbal_odom.translation(),
            bullet_speed
        );
        if (!prev_pitch_and_fly_time_opt) {
            return std::nullopt;
        }
        auto prev_fly_time = prev_pitch_and_fly_time_opt.value().second;

        for (int iter = 0; iter < 10; ++iter) {
            auto i_target = hit_time_target;
            i_target.set_target_state([&](armor_point_motion_model::State& state) {
                state.predict(prev_fly_time, i_target.target_number);
            });
            // auto iter_select = select_armor(i_target, fsm);//不知道哪个最好
            const auto iter_armors_xyza =
                i_target.get_target_state().get_armors_xyza(i_target.target_number);
            // auto iter_pitch_and_fly_time_opt = ballistic_trajectory_->solve_pitch_and_flytime(
            //     iter_armors_xyza[iter_select].head<3>(),
            //     bullet_speed
            // );
            auto iter_pitch_and_fly_time_opt = ballistic_trajectory_->solve_pitch_and_flytime(
                iter_armors_xyza[roughly_select].head<3>() - shoot_in_gimbal_odom.translation(),
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
        const double predict_time = prev_fly_time + params_.prediction_delay
            + (fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER ? params_.aim_center_more_prediction_time : 0
            ); //瞄准中间只能预测一段发弹延迟
        hit_time_target.set_target_state([&](auto& state) {
            state.predict(predict_time, hit_time_target.target_number);
        });
        return HitCtx {
            .hit_time_target = hit_time_target,
            .fly_time = prev_fly_time,
        };
    }
    ArmorTarget last_target_;
    TimePoint last_time_;
    dta_utils::LimitTrajectory limit_traj_;
    dta_utils::ControlPoint limit_traj_cp0_;
    Trajectory<AimPoint, double> aim_traj_;
    Trajectory<dta_utils::GimbalState, double> aim_center_target_traj_;
    dta_utils::ControlPoint aim_center_target_traj_cp0_;
    double last_fly_time_;
    int last_select_;
    [[nodiscard]] GimbalCmd very_aim(
        const ArmorTarget& _target,
        double bullet_speed,
        AutoAimFsm fsm,
        const ISO3& shoot_in_gimbal_odom,
        const ISO3& gimbal_in_gimbal_odom
    ) noexcept {
        if (_target.target_number == ArmorClass::BASE) {
            fsm = AutoAimFsm::AIM_SINGLE_ARMOR;
        }
        GimbalCmd cmd;
        cmd.appear = false;
        bool is_same = _target.this_id == last_target_.this_id;
        double time_in_traj = 0.0;
        if (is_same) { //状态完全未更新，使用第一次构建的轨迹基于时间进行推进减少重复计算
            time_in_traj = std::chrono::duration<double>(Clock::now() - last_time_).count();
        } else {
            last_target_ = _target;
            last_time_ = Clock::now();
        }
        auto target = _target;
        target.set_target_state([&](armor_point_motion_model::State& state) {
            state.predict(Clock::now(), target.target_number);
        });
        auto make_even = [](int x) { return x % 2 == 0 ? x : x + 1; };

        const int horizon = make_even(params_.sample_horizon);
        const double dt = params_.sample_total_time / horizon;
        if (!is_same) {
            auto hit_ctx_opt =
                get_hit(target, bullet_speed, fsm, shoot_in_gimbal_odom); //直接算到击中目标时间
            if (!hit_ctx_opt) {
                return cmd;
            }

            auto hit_ctx = hit_ctx_opt.value();
            auto cp0 = select_and_get_control_point(
                hit_ctx.hit_time_target,
                bullet_speed,
                fsm,
                shoot_in_gimbal_odom,
                gimbal_in_gimbal_odom
            );
            if (!cp0.valid) {
                return cmd;
            }
            last_fly_time_ = hit_ctx.fly_time;
            last_select_ = cp0.aim_id;
            auto sample_once = [&](double t,
                                   const ArmorTarget& base_target,
                                   AutoAimFsm fsm_mode,
                                   const dta_utils::ControlPoint& _cp0,
                                   dta_utils::GimbalState& out_gs,
                                   AimPoint& out_ap) -> bool {
                auto tmp_target = base_target;
                tmp_target.set_target_state([&](armor_point_motion_model::State& state) {
                    state.predict(t, tmp_target.target_number);
                }); //保证轨迹完全为不同时间对同一状态的瞄准控制点

                auto hit_opt = get_hit(tmp_target, bullet_speed, fsm_mode, shoot_in_gimbal_odom);
                if (!hit_opt)
                    return false;

                auto cp = select_and_get_control_point(
                    hit_opt->hit_time_target,
                    bullet_speed,
                    fsm_mode,
                    shoot_in_gimbal_odom,
                    gimbal_in_gimbal_odom
                );
                if (!cp.valid)
                    return false;

                out_gs.aim_id = cp.aim_id;
                out_gs.yaw_state.p = angles::normalize_angle(cp.yaw - _cp0.yaw);
                out_gs.pitch_state.p = angles::normalize_angle(cp.pitch - _cp0.pitch);
                out_ap = cp.aim_point;

                return true;
            };

            auto push_sample = [&](auto& traj_gs,
                                   auto& traj_ap,
                                   double t,
                                   const dta_utils::ControlPoint& _cp0,
                                   AutoAimFsm fsm_mode) -> bool {
                dta_utils::GimbalState gs;
                AimPoint ap;
                if (!sample_once(t, target, fsm_mode, _cp0, gs, ap)) {
                    return false;
                }
                traj_gs.push_back(gs, t);
                traj_ap.push_back(ap, t);
                return true;
            };

            auto build_traj = [&](auto& traj_gs,
                                  auto& traj_ap,
                                  const dta_utils::ControlPoint& _cp0,
                                  AutoAimFsm fsm_mode,
                                  int horizon,
                                  double dt) -> bool {
                int half = horizon / 2;
                for (int i = half; i >= 1; --i) {
                    if (!push_sample(traj_gs, traj_ap, -i * dt, _cp0, fsm_mode)) {
                        return false;
                    }
                }

                dta_utils::GimbalState gs0;
                gs0.aim_id = _cp0.aim_id;
                gs0.yaw_state.p = 0.0;
                gs0.pitch_state.p = 0.0;
                traj_gs.push_back(gs0, 0.0);
                traj_ap.push_back(_cp0.aim_point, 0.0);

                for (int i = 1; i <= half; ++i) {
                    if (!push_sample(traj_gs, traj_ap, i * dt, _cp0, fsm_mode)) {
                        return false;
                    }
                }

                return true;
            };
            limit_traj_cp0_ = cp0;
            limit_traj_.clear();
            aim_traj_.clear();

            limit_traj_.reserve(horizon + 1);
            aim_traj_.reserve(horizon + 1);

            if (!build_traj(limit_traj_, aim_traj_, limit_traj_cp0_, fsm, horizon, dt)) {
                return cmd;
            }

            limit_traj_.build_limit(params_.max_yaw_acc, params_.max_pitch_acc, time_in_traj);

            if (fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER) { //瞄准中间的目标和控制不一样
                aim_center_target_traj_.clear();

                aim_center_target_traj_.reserve(horizon + 1);
                auto aim_center_target_hit_ctx_opt = get_hit(
                    target,
                    bullet_speed,
                    AutoAimFsm::AIM_WHOLE_CAR_ARMOR,
                    shoot_in_gimbal_odom
                );
                if (!aim_center_target_hit_ctx_opt) {
                    return cmd;
                }

                auto aim_center_target_hit_ctx = aim_center_target_hit_ctx_opt.value();
                aim_center_target_traj_cp0_ = select_and_get_control_point(
                    aim_center_target_hit_ctx.hit_time_target,
                    bullet_speed,
                    AutoAimFsm::AIM_WHOLE_CAR_ARMOR,
                    shoot_in_gimbal_odom,
                    gimbal_in_gimbal_odom
                );
                aim_traj_.clear();
                aim_traj_.reserve(horizon + 1);
                if (!build_traj(
                        aim_center_target_traj_,
                        aim_traj_,
                        aim_center_target_traj_cp0_,
                        AutoAimFsm::AIM_WHOLE_CAR_ARMOR,
                        horizon,
                        dt
                    ))
                {
                    return cmd;
                }
            }
        }

        const bool use_center_target = fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER;
        const dta_utils::ControlPoint& target_traj_cp0 =
            use_center_target ? aim_center_target_traj_cp0_ : limit_traj_cp0_;
        const Trajectory<dta_utils::GimbalState, double>& target_traj = use_center_target
            ? static_cast<const Trajectory<dta_utils::GimbalState, double>&>(aim_center_target_traj_
            )
            : static_cast<const Trajectory<dta_utils::GimbalState, double>&>(limit_traj_);
        auto target_gimbal_state = target_traj.Trajectory::state_at(time_in_traj);
        //目标轨迹，一定击中目标
        auto control = limit_traj_.dta_utils::LimitTrajectory::state_at(time_in_traj);
        //控制轨迹，轨迹优化后最优控制（并非最优，下位机实际vel acc 可以基于error和上位机规划叠加）
        double control_yaw = angles::normalize_angle(control.yaw_state.p + limit_traj_cp0_.yaw);
        double control_pitch =
            angles::normalize_angle(control.pitch_state.p + limit_traj_cp0_.pitch);
        double target_yaw =
            angles::normalize_angle(target_gimbal_state.yaw_state.p + target_traj_cp0.yaw);
        double target_pitch =
            angles::normalize_angle(target_gimbal_state.pitch_state.p + target_traj_cp0.pitch);
        cmd.timestamp = Clock::now();
        cmd.yaw = angles::to_degrees(control_yaw);
        cmd.v_yaw = angles::to_degrees(control.yaw_state.v);
        cmd.a_yaw = angles::to_degrees(control.yaw_state.a);
        cmd.pitch = angles::to_degrees(control_pitch);
        cmd.v_pitch = angles::to_degrees(control.pitch_state.v);
        cmd.a_pitch = angles::to_degrees(control.pitch_state.a);
        cmd.target_yaw = angles::to_degrees(target_yaw);
        cmd.target_pitch = angles::to_degrees(target_pitch);
        cmd.fly_time = last_fly_time_;
        cmd.appear = true;
        cmd.aim_point = aim_traj_.state_at(time_in_traj);
        cmd.aim_point.frame_id = target.get_target_state().frame_id;
        cmd.select_id = last_select_;
        bool is_big = target.target_number == ArmorClass::NO1;
        auto cal_enbale_diff = [&](double _t) {
            auto aim_point = aim_traj_.state_at(_t);
            const double distance = aim_point.pose.translation().norm();
            double shooting_range_yaw;
            double half_w =
                is_big ? params_.shooting_range_w_large / 2 : params_.shooting_range_w_small / 2;
            auto cos_theta = std::cos(aim_point.d_angle);
            auto sin_theta = std::sin(aim_point.d_angle);

            auto yaw1 = std::atan2(
                aim_point.pose.translation().y() + half_w * cos_theta,
                aim_point.pose.translation().x() - half_w * sin_theta
            );
            auto yaw2 = std::atan2(
                aim_point.pose.translation().y() - half_w * cos_theta,
                aim_point.pose.translation().x() + half_w * sin_theta
            );
            auto aim_yaw =
                std::atan2(aim_point.pose.translation().y(), aim_point.pose.translation().x());
            shooting_range_yaw = std::min(
                std::abs(angles::normalize_angle(yaw1 - aim_yaw)),
                std::abs(angles::normalize_angle(yaw2 - aim_yaw))
            ); //直接算两个边缘yaw
            double shooting_range_pitch =
                std::abs(std::atan2(params_.shooting_range_h / 2, distance));
            const double pitch_factor =
                std::cos(FIFTTEN_DEGREE_RAD); //跟yaw一样逻辑还得多两次解弹道，没必要
            shooting_range_pitch *= pitch_factor;
            shooting_range_yaw =
                std::max(shooting_range_yaw, angles::from_degrees(params_.min_enable_yaw_deg));
            shooting_range_pitch =
                std::max(shooting_range_pitch, angles::from_degrees(params_.min_enable_pitch_deg));
            return std::make_pair(std::abs(shooting_range_yaw), std::abs(shooting_range_pitch));
        };
        auto enable_diff = cal_enbale_diff(time_in_traj);
        cmd.enable_yaw_diff = angles::to_degrees(enable_diff.first);
        cmd.enable_pitch_diff = angles::to_degrees(enable_diff.second);
        auto abs_angle_error = [](double from, double to) {
            return std::abs(angles::shortest_angular_distance(from, to));
        };
        cmd.fire_advice =
            abs_angle_error(angles::from_degrees(cmd.target_yaw), angles::from_degrees(cmd.yaw))
                < cmd.enable_yaw_diff
            && abs_angle_error(
                   angles::from_degrees(cmd.target_pitch),
                   angles::from_degrees(cmd.pitch)
               ) < cmd.enable_pitch_diff;
        if (fsm != AutoAimFsm::AIM_WHOLE_CAR_CENTER) {
            auto delay_fire = [&](double delay) {
                auto delay_control = limit_traj_.LimitTrajectory::state_at(time_in_traj + delay);
                auto delay_target = target_traj.Trajectory::state_at(time_in_traj + delay);
                auto delay_enable = cal_enbale_diff(time_in_traj + delay);
                const double control_yaw =
                    angles::normalize_angle(delay_control.yaw_state.p + limit_traj_cp0_.yaw);
                const double target_yaw =
                    angles::normalize_angle(delay_target.yaw_state.p + target_traj_cp0.yaw);
                const double control_pitch =
                    angles::normalize_angle(delay_control.pitch_state.p + limit_traj_cp0_.pitch);
                const double target_pitch =
                    angles::normalize_angle(delay_target.pitch_state.p + target_traj_cp0.pitch);

                return abs_angle_error(control_yaw, target_yaw) < delay_enable.first
                    && abs_angle_error(control_pitch, target_pitch) < delay_enable.second;
            };
            {
                double t_check = 0 + params_.fire_delay_min; //发射延迟内不让打弹
                while (t_check < (0 + params_.fire_delay_max) && t_check <= horizon / 2.0) {
                    if (!delay_fire(+t_check)) {
                        cmd.no_shoot();
                    }
                    t_check += (dt / 2.0);
                }
                //发射延迟提前开火？
            }
        }

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
VeryAimer::VeryAimer(const YAML::Node& config) {
    _impl = std::make_unique<Impl>(config);
}
VeryAimer::~VeryAimer() noexcept {
    _impl.reset();
}
GimbalCmd VeryAimer::very_aim(
    const ArmorTarget& target,
    double bullet_speed,
    const AutoAimFsm& fsm,
    const ISO3& shoot_in_gimbal_odom,
    const ISO3& gimbal_in_gimbal_odom
) {
    return _impl->very_aim(target, bullet_speed, fsm, shoot_in_gimbal_odom, gimbal_in_gimbal_odom);
}
std::pair<double, double> VeryAimer::get_yaw_pitch_offset() {
    return _impl->get_yaw_pitch_offset();
}
void VeryAimer::set_operator_offset(std::pair<double, double> offset) {
    _impl->set_operator_offset(offset);
}
} // namespace awakening::auto_aim
