#include "rune_target.hpp"
#include "tasks/auto_buff/rune_track/motion_model.hpp"
#include "tasks/auto_buff/type.hpp"
#include "tasks/base/dta_utils.hpp"
#include "tasks/base/web.hpp"
#include "utils/common/type_common.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <optional>
#include <vector>
namespace awakening::auto_buff {
using namespace motion_model;

namespace {
    template<typename Fan, typename Pnp>
    std::vector<std::pair<int, Fan>> match_fans_by_ypd(
        const RuneTarget& target,
        std::vector<Fan>& fans,
        const TimePoint& timestamp,
        Pnp&& pnp
    ) noexcept {
        constexpr double MAX_COST = 1e9;
        std::vector<std::pair<int, Fan>> result;
        const int n_obs = static_cast<int>(fans.size());
        std::vector<std::vector<double>> cost(n_obs, std::vector<double>(FAN_NUM, MAX_COST + 1));

        std::vector<YPDVecZ> meas_list(n_obs);
        for (int obs = 0; obs < n_obs; ++obs) {
            pnp(fans[obs]);
            meas_list[obs] = target.get_ypdmeasurement(fans[obs].pose);
        }

        auto pred_state = target.get_target_state();
        pred_state.predict(timestamp, target.voter);
        for (int obs = 0; obs < n_obs; ++obs) {
            bool in_gate = false;
            double min_d2 = std::numeric_limits<double>::max();
            for (int id = 0; id < FAN_NUM; ++id) {
                YPDMeasure measure { .ctx = { .id = id } };
                YPDVecZ z_pred;
                measure.h(pred_state.x, z_pred);

                YPDVecZ nu = meas_list[obs] - z_pred;
                nu[idx::YPD_Y] = angles::normalize_angle(nu[idx::YPD_Y]);
                nu[idx::ROT_YAW] = angles::normalize_angle(nu[idx::ROT_YAW]);
                nu[idx::ROT_ROLL] = angles::normalize_angle(nu[idx::ROT_ROLL]);
                auto R = target.ypdmeasurement_covariance(z_pred);
                const double d2 = nu.transpose() * R.ldlt().solve(nu);

                if (std::isfinite(d2) && d2 < target.cfg.match_gate) {
                    cost[obs][id] = d2;
                    in_gate = true;
                }
                min_d2 = std::min(min_d2, d2);
            }
            if (!in_gate) {
                AWAKENING_WARN("match out of gate min d2: {}", min_d2);
            }
        }

        for (auto [obs, id]: dta_utils::greedy_match(cost, n_obs, FAN_NUM, MAX_COST)) {
            result.emplace_back(id, fans[obs]);
        }
        return result;
    }
} // namespace

void RuneTarget::write_log() {
    Web::write_log("rune_target", [&](auto& j) {
        j["timestamp"] =
            static_cast<int>(std::chrono::duration<double>(last_update.time_since_epoch()).count());
        j["track_state"] = TrackState::string_by_state(track_state.tracker_state);
        auto& j_target_state = j["target_state"];
        j_target_state["cx"] = Web::val(target_state.pos().x());
        j_target_state["cy"] = Web::val(target_state.pos().y());
        j_target_state["cz"] = Web::val(target_state.pos().z());
        j_target_state["yaw"] = Web::val(target_state.yaw());
        j_target_state["roll"] = Web::val(target_state.roll());
        j_target_state["v_roll"] = Web::val(target_state.v_roll(voter));
        j_target_state["a"] = Web::val(target_state.a());
        j_target_state["w"] = Web::val(target_state.w());
        j_target_state["tau"] = Web::val(target_state.tau());
        j_target_state["visible"] = fan_wc.to_str();
        j_target_state["voter"] = voter.to_str();
    });
}
bool RuneTarget::reset(
    RuneDetection& d,
    const RuneTrackerCfg& c,
    const TimePoint& timestamp,
    int frame_id,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) {
    cfg = c;
    std::optional<ISO3> pose;
    if (d.fan_blades.size() > 0) {
        fan_pnp(d.fan_blades.front(), camera_info, camera_cv_in_odom, true);
        pose = d.fan_blades.front().pose;
    } else if (!d.fan_targets.empty() && !d.rune_rs.empty()) {
        cv::Point2f r;
        double min_area = std::numeric_limits<double>::max();
        for (const auto& rune_r: d.rune_rs) {
            if (rune_r.rr.boundingRect2f().area() < min_area) {
                min_area = rune_r.rr.boundingRect2f().area();
                r = rune_r.rr.center;
            }
        }
        d.fan_targets.front().sort_corners(r);
        fan_target_pnp(d.fan_targets.front(), r, camera_info, camera_cv_in_odom, true);
    }
    if (!pose) {
        return false;
    }
    ypd_ctx = {
        .id = 0,
    };
    r_ctx = {
        .camera_cv_in_odom = camera_cv_in_odom,
        .camera_info = camera_info.clone(),
    };
    fan_ctx = {
        .id = 0,
        .camera_cv_in_odom = camera_cv_in_odom,
        .camera_info = camera_info.clone(),
    };
    fan_target_ctx = {
        .id = 0,
        .camera_cv_in_odom = camera_cv_in_odom,
        .camera_info = camera_info.clone(),
    };
    Eigen::DiagonalMatrix<double, X_N> p0;
    p0.diagonal().setZero();
    const auto u_q = [this]() {
        Eigen::Matrix<double, X_N, X_N> q;
        return q;
    };

    const auto inject =
        [this](const Eigen::Matrix<double, X_N, 1>& delta, Eigen::Matrix<double, X_N, 1>& nominal) {
            for (int i = 0; i < X_N; i++) {
                if (i == idx::YAW || i == idx::ROLL)
                    continue;
                nominal[i] += delta[i];
            }
            nominal[idx::YAW] = angles::normalize_angle(nominal[idx::YAW] + delta[idx::YAW]);
            nominal[idx::ROLL] = angles::normalize_angle(nominal[idx::ROLL] + delta[idx::ROLL]);
        };
    voter.reset(timestamp);
    esekf = ESEKF(Predict { .dt = 0.005, .voter = voter }, u_q, inject, p0);

    esekf.value().set_iteration_num(cfg.esekf_iter_num);

    auto pos = pose.value().translation();
    auto rpy = utils::matrix2rpy(pose.value().linear());
    double a_guess = (A_LOWER + A_UPPER) / 2.0;
    double w_guess = (W_LOWER + W_UPPER) / 2.0;
    double tau = 0;
    if (std::chrono::duration<double>(timestamp - last_update).count() < cfg.big_args_continue_time)
    {
        a_guess = target_state.x[idx::A];
        w_guess = target_state.x[idx::W];
        tau = target_state.x[idx::TAU]
            + std::chrono::duration<double>(timestamp - target_state.timestamp).count();
    }
    target_state.x = Eigen::VectorXd::Zero(X_N);
    target_state.x << pos.x(), pos.y(), pos.z(), rpy[2], rpy[0], 0, tau, a_guess, w_guess;
    target_state.timestamp = timestamp;
    target_state.frame_id = frame_id;
    esekf.value().set_state(target_state.x);
    last_update = timestamp;
    is_inited = true;
    track_state.reset();
    this_id = GOBAL_ID++;
    fan_wc.reset();
    fan_wc.is_visable[0] = true;
    return true;

}
void RuneTarget::fan_pnp(
    RuneFanBladeWithR& r,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom,
    bool in_r
) noexcept {
    auto key_points = r.points;
    r.pose = utils::solve_pnp(
        key_points,
        in_r ? RuneFanBladeWithR::Point3DTargetCenterZERO<cv::Point3f>::build()
             : RuneFanBladeWithR::Point3DRZERO<cv::Point3f>::build(),
        camera_info.camera_matrix,
        camera_info.distortion_coefficients
    );
    r.pose = camera_cv_in_odom * r.pose;
}
void RuneTarget::fan_target_pnp(
    RuneFanTarget& a,
    const cv::Point2f& r,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom,
    bool in_r
) noexcept {
    a.sort_corners(r);
    auto key_points = a.key_points;
    a.pose = utils::solve_pnp(
        key_points,
        in_r ? RuneFanTarget::Point3DTargetCenterZERO<cv::Point3f>::build_no_r()
             : RuneFanTarget::Point3DRZERO<cv::Point3f>::build_no_r(),
        camera_info.camera_matrix,
        camera_info.distortion_coefficients
    );
    a.pose = camera_cv_in_odom * a.pose;
}
[[nodiscard]] Eigen::Matrix<double, motion_model::X_N, motion_model::X_N>
RuneTarget::process_noise(double dt, const Voter& v) const noexcept {
    Eigen::Matrix<double, X_N, X_N> q;
    q.setZero();
    q(idx::CX, idx::CX) = cfg.q_xyz.x();
    q(idx::CY, idx::CY) = cfg.q_xyz.y();
    q(idx::CZ, idx::CZ) = cfg.q_xyz.z();
    q(idx::YAW, idx::YAW) = cfg.q_yaw;

    utils::fill_constant_accel_noise(q, idx::ROLL, idx::V_ROLL, cfg.q_roll, dt);

    q(idx::A, idx::A) = cfg.q_a;
    q(idx::W, idx::W) = cfg.q_w;
    q(idx::TAU, idx::TAU) = cfg.q_tau;

    return q;
}
[[nodiscard]] Eigen::Matrix<double, motion_model::YPDZ_N, motion_model::YPDZ_N>
RuneTarget::ypdmeasurement_covariance(const Eigen::Matrix<double, motion_model::YPDZ_N, 1>& z
) const noexcept {
    Eigen::Matrix<double, YPDZ_N, YPDZ_N> r;

    r.setZero(); //copy下sp_vision_25 这个参数不用在观测，差不多就行
    r(idx::YPD_Y, idx::YPD_Y) = 4e-3;
    r(idx::YPD_P, idx::YPD_P) = 4e-3;
    r(idx::YPD_D, idx::YPD_D) = z[idx::YPD_D] * z[idx::YPD_D] * 0.1;
    r(idx::ROT_YAW, idx::ROT_YAW) = 0.0;
    r(idx::ROT_ROLL, idx::ROT_ROLL) = 0.05;
    return r;
}
[[nodiscard]] Eigen::Matrix<double, motion_model::YPDZ_N, 1>
RuneTarget::get_ypdmeasurement(const ISO3& pose) const noexcept {
    Eigen::Matrix<double, YPDZ_N, 1> z;
    double ax = pose.translation().x(), ay = pose.translation().y(), az = pose.translation().z();
    auto ypd_y = std::atan2(ay, ax);
    static double last_ypd_y = 0;
    ypd_y = last_ypd_y + angles::shortest_angular_distance(last_ypd_y, ypd_y);
    last_ypd_y = ypd_y;
    auto ypd_p = std::atan2(az, std::sqrt(ax * ax + ay * ay));
    auto ypd_d = std::sqrt(ax * ax + ay * ay + az * az);
    z[idx::YPD_Y] = ypd_y;
    z[idx::YPD_P] = ypd_p;
    z[idx::YPD_D] = ypd_d;
    auto rpy = utils::matrix2rpy(pose.linear());
    double yaw = rpy[2];
    z[idx::ROT_YAW] = last_rot_yaw + angles::shortest_angular_distance(last_rot_yaw, yaw);
    last_rot_yaw = z[idx::ROT_YAW];
    double roll = rpy[0];
    z[idx::ROT_ROLL] = last_rot_roll + angles::shortest_angular_distance(last_rot_roll, roll);
    last_rot_roll = z[idx::ROT_ROLL];
    return z;
}
void RuneTarget::predict_ekf(const TimePoint& timestamp) {
    if (!esekf) {
        throw std::runtime_error("ESEKF is not initialized");
    }
    auto dt = std::chrono::duration<double>(timestamp - target_state.timestamp).count();
    esekf.value().set_predict_func(Predict { .dt = dt, .voter = voter });
    esekf.value().set_update_Q([&]() { return process_noise(dt, voter); });
    target_state.x = esekf.value().predict();
    target_state.timestamp = timestamp;
    this_id = GOBAL_ID++; //全局状态标记，下游控制对同一id的不重复构建轨迹
}
std::optional<std::pair<bool, cv::Point2f>> RuneTarget::match_r(
    std::vector<std::pair<int, RuneFanBladeWithR>>& matched_fans,
    std::vector<RuneR>& rs,
    const TimePoint& timestamp,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) {
    std::optional<std::pair<bool, cv::Point2f>> best = std::nullopt;
    if (rs.empty()) {
        return best;
    }
    auto pred_state = target_state;
    pred_state.predict(timestamp, voter);
    RVecZ _r_z_pred;
    auto _r_ctx = r_ctx;
    _r_ctx.camera_cv_in_odom = camera_cv_in_odom;
    RMeasure _r_measure { .ctx = _r_ctx };
    _r_measure.h(pred_state.x, _r_z_pred);
    FanBladeVecZ _fan_z_pred;
    auto _fan_ctx = fan_ctx;
    _fan_ctx.camera_cv_in_odom = camera_cv_in_odom;
    _fan_ctx.id = 0;
    FanBladeMeasure _fan_measure { .ctx = _fan_ctx };
    _fan_measure.h(pred_state.x, _fan_z_pred);
    std::vector<cv::Point2f> r_vec;
    std::vector<double> fan_hand_length_vec;
    for (const auto& [id, fan]: matched_fans) {
        r_vec.push_back(fan.points[RuneFanBladeWithR::PointsIndex::R]);
        fan_hand_length_vec.push_back(cv::norm(
            fan.points[RuneFanBladeWithR::PointsIndex::BOTTOM]
            - fan.points[RuneFanBladeWithR::PointsIndex::R]
        ));
    }
    r_vec.emplace_back(_r_z_pred[idx::R_X], _r_z_pred[idx::R_Y]);
    fan_hand_length_vec.push_back(cv::norm(
        cv::Point2f(_fan_z_pred[idx::BOTTOM_X], _fan_z_pred[idx::BOTTOM_Y])
        - cv::Point2f(_r_z_pred[idx::R_X], _r_z_pred[idx::R_Y])
    ));

    auto avg_r = std::accumulate(
                     r_vec.begin(),
                     r_vec.end(),
                     cv::Point2f(0, 0),
                     [](const cv::Point2f& a, const cv::Point2f& b) {
                         return cv::Point2f(a.x + b.x, a.y + b.y);
                     }
                 )
        * (1.0 / (r_vec.size()));
    auto avg_hand_length =
        std::accumulate(fan_hand_length_vec.begin(), fan_hand_length_vec.end(), 0.0)
        / fan_hand_length_vec.size();
    int best_id = -1;
    double min_cost = std::numeric_limits<double>::max();
    for (size_t i = 0; i < rs.size(); ++i) {
        double error = cv::norm(rs[i].rr.center - avg_r);
        if (error > avg_hand_length * 0.2) {
            continue;
        }
        if (error < min_cost) {
            min_cost = error;
            best_id = i;
        }
    }
    if (best_id != -1) {
        best = std::make_pair(true, rs[best_id].rr.center);
        rs[best_id].laji = false;
    } else if (!matched_fans.empty() && r_vec.size() > 1) {
        avg_r = std::accumulate(
                    r_vec.begin(),
                    r_vec.end() - 1,
                    cv::Point2f(0, 0),
                    [](const cv::Point2f& a, const cv::Point2f& b) {
                        return cv::Point2f(a.x + b.x, a.y + b.y);
                    }
                )
            * (1.0 / (r_vec.size() - 1));
        best = std::make_pair(false, avg_r);
    }

    return best;
}
int RuneTarget::update(
    std::vector<std::pair<int, RuneFanBladeWithR>>& matched_fans,
    std::vector<std::pair<int, RuneFanTarget>>& matched_fan_targets,
    std::optional<std::pair<bool, cv::Point2f>>& r,
    const TimePoint& timestamp,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) {
    std::vector<std::shared_ptr<ESEKF::ObsBase>> obs;
    if (r) {
        RVecZ z;
        auto ctx = r_ctx;
        ctx.camera_cv_in_odom = camera_cv_in_odom;
        RMeasure measure { .ctx = ctx };
        const auto r_u_r = [&](const Eigen::Matrix<double, RZ_N, 1>& z) {
            Eigen::Matrix<double, RZ_N, RZ_N> _r;
            _r.setZero();
            auto r_uv = (r.value().first) ? cfg.r_uv_cv : cfg.r_uv_net;
            _r.diagonal().setConstant(r_uv);
            return _r;
        };
        const auto r_cal_residual = [](const Eigen::Matrix<double, RZ_N, 1>& z_pred,
                                       const Eigen::Matrix<double, RZ_N, 1>& z) {
            return z - z_pred;
        };
        z[idx::R_X] = r.value().second.x;
        z[idx::R_Y] = r.value().second.y;
        obs.push_back(esekf.value().make_obs(z, measure, r_u_r, r_cal_residual));
    }

    const auto fan_u_r = [&](const Eigen::Matrix<double, FanBladeZ_N, 1>& z) {
        Eigen::Matrix<double, FanBladeZ_N, FanBladeZ_N> r;
        r.setZero();
        r.diagonal().setConstant(cfg.r_uv_net);
        return r;
    };

    const auto fan_cal_residual = [](const Eigen::Matrix<double, FanBladeZ_N, 1>& z_pred,
                                     const Eigen::Matrix<double, FanBladeZ_N, 1>& z) {
        return z - z_pred;
    };
    fan_wc.reset();

    for (const auto& [id, fan]: matched_fans) {
        fan_wc.update(id, timestamp);
        FanBladeVecZ z;
        z[idx::TOP_X] = fan.points[RuneFanBladeWithR::PointsIndex::TOP].x;
        z[idx::TOP_Y] = fan.points[RuneFanBladeWithR::PointsIndex::TOP].y;
        z[idx::LEFT_X] = fan.points[RuneFanBladeWithR::PointsIndex::LEFT].x;
        z[idx::LEFT_Y] = fan.points[RuneFanBladeWithR::PointsIndex::LEFT].y;
        z[idx::RIGHT_X] = fan.points[RuneFanBladeWithR::PointsIndex::RIGHT].x;
        z[idx::RIGHT_Y] = fan.points[RuneFanBladeWithR::PointsIndex::RIGHT].y;
        z[idx::BOTTOM_X] = fan.points[RuneFanBladeWithR::PointsIndex::BOTTOM].x;
        z[idx::BOTTOM_Y] = fan.points[RuneFanBladeWithR::PointsIndex::BOTTOM].y;
        auto ctx = fan_ctx;
        ctx.id = id;
        ctx.camera_cv_in_odom = camera_cv_in_odom;
        FanBladeMeasure measure { .ctx = ctx };
        obs.push_back(esekf.value().make_obs(z, measure, fan_u_r, fan_cal_residual));
    }
    const auto fan_target_u_r = [&](const Eigen::Matrix<double, FanTargetZ_N, 1>& z) {
        Eigen::Matrix<double, FanTargetZ_N, FanTargetZ_N> r;
        r.setZero();
        r.diagonal().setConstant(cfg.r_uv_cv);
        return r;
    };

    const auto fan_target_cal_residual = [](const Eigen::Matrix<double, FanTargetZ_N, 1>& z_pred,
                                            const Eigen::Matrix<double, FanTargetZ_N, 1>& z) {
        return z - z_pred;
    };
    for (const auto& [id, fan]: matched_fan_targets) {
        fan_wc.update(id, timestamp);
        FanTargetVecZ z;
        z[idx::LT_X] = fan.key_points[RuneFanTarget::PointsIndex::LT].x;
        z[idx::LT_Y] = fan.key_points[RuneFanTarget::PointsIndex::LT].y;
        z[idx::LB_X] = fan.key_points[RuneFanTarget::PointsIndex::LB].x;
        z[idx::LB_Y] = fan.key_points[RuneFanTarget::PointsIndex::LB].y;
        z[idx::RB_X] = fan.key_points[RuneFanTarget::PointsIndex::RB].x;
        z[idx::RB_Y] = fan.key_points[RuneFanTarget::PointsIndex::RB].y;
        z[idx::RT_X] = fan.key_points[RuneFanTarget::PointsIndex::RT].x;
        z[idx::RT_Y] = fan.key_points[RuneFanTarget::PointsIndex::RT].y;
        z[idx::CEN_X] = fan.key_points[RuneFanTarget::PointsIndex::CENTER].x;
        z[idx::CEN_Y] = fan.key_points[RuneFanTarget::PointsIndex::CENTER].y;
        auto ctx = fan_target_ctx;
        ctx.id = id;
        ctx.camera_cv_in_odom = camera_cv_in_odom;
        FanTargetMeasure measure { .ctx = ctx };
        obs.push_back(esekf.value().make_obs(z, measure, fan_target_u_r, fan_target_cal_residual));
    }

    target_state.x = esekf.value().update_multi(obs);
    target_state.timestamp = timestamp;
    last_update = timestamp;
    this_id = GOBAL_ID++; //全局状态标记，下游控制对同一id的不重复构建轨迹]
    int match_fan_num = matched_fans.size() + matched_fan_targets.size();
    if (match_fan_num > 0) {
        voter.update(target_state.roll(), cfg.voter_state_need_count);
    }
    if (matched_fans.size() >= 2 || matched_fan_targets.size() >= 2) {
        voter.double_detect_count++;
        if (voter.double_detect_count > cfg.voter_mode_need_count) {
            voter.mode = Voter::Big;
        }
    }
    return match_fan_num;
}
std::vector<std::pair<int, RuneFanBladeWithR>> RuneTarget::match_fan(
    std::vector<RuneFanBladeWithR>& fans,
    const TimePoint& timestamp,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) const noexcept {
    return match_fans_by_ypd(*this, fans, timestamp, [&](RuneFanBladeWithR& fan) {
        fan_pnp(fan, camera_info, camera_cv_in_odom, false);
    });
}
std::vector<std::pair<int, RuneFanTarget>> RuneTarget::match_fan_target(
    std::vector<RuneFanTarget>& fans,
    std::optional<std::pair<bool, cv::Point2f>>& r,
    const TimePoint& timestamp,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) const noexcept {
    if (!r) {
        return {};
    }
    return match_fans_by_ypd(*this, fans, timestamp, [&](RuneFanTarget& fan) {
        fan_target_pnp(fan, r.value().second, camera_info, camera_cv_in_odom, false);
    });
}
[[nodiscard]] cv::Rect RuneTarget::get_net_focus_roi(
    const TimePoint& timestamp,
    const ISO3& camera_cv_in_odom,
    const CameraInfo& camera_info,
    const cv::Size& image_size,
    double target_wh_ratio
) const noexcept {
    if (!need_focus()) {
        return cv::Rect(0, 0, image_size.width, image_size.height);
    }

    auto tmp_target_state = target_state;
    tmp_target_state.predict(timestamp, voter);

    static std::vector<cv::Point3f> CAR_BOX;
    constexpr float car_box_half = 1.0f;
    CAR_BOX = { { 0, car_box_half, -car_box_half },
                { 0, -car_box_half, -car_box_half },
                { 0, -car_box_half, car_box_half },
                { 0, car_box_half, car_box_half } };

    auto pos = tmp_target_state.pos();
    ISO3 pose_in_odom;
    pose_in_odom.translation() = pos;
    pose_in_odom.linear() = utils::rpy2matrix(Vec3(0, 0, std::atan2(pos.x(), pos.y())));
    auto pose_in_camera_cv = camera_cv_in_odom.inverse() * pose_in_odom;

    auto pts = utils::reprojection(
        camera_info.camera_matrix,
        camera_info.distortion_coefficients,
        CAR_BOX,
        pose_in_camera_cv
    );
    cv::Rect rect = cv::boundingRect(pts);
    cv::Rect img_rect(0, 0, image_size.width, image_size.height);

    if ((rect & img_rect).area() <= 0) {
        return img_rect;
    }
    rect &= img_rect;

    double rect_w = std::max<double>(rect.width, 1.0);
    double rect_h = std::max<double>(rect.height, 1.0);
    double ratio =
        (std::isfinite(target_wh_ratio) && target_wh_ratio > 0.0) ? target_wh_ratio : 1.0;

    double target_w = rect_w;
    double target_h = rect_h;
    if (target_w / target_h < ratio) {
        target_w = target_h * ratio;
    } else {
        target_h = target_w / ratio;
    }

    double cx = rect.x + rect.width / 2.0;
    double cy = rect.y + rect.height / 2.0;
    cv::Rect ratio_rect(
        static_cast<int>(cx - target_w / 2.0),
        static_cast<int>(cy - target_h / 2.0),
        static_cast<int>(target_w),
        static_cast<int>(target_h)
    );
    ratio_rect &= img_rect;

    double dt = std::chrono::duration<double>(timestamp - last_update).count();
    double lost_dt = cfg.lost_time_thres;
    double dt_clamped = std::max(0.0, std::min(dt, lost_dt));

    int base_side = std::max(ratio_rect.width, ratio_rect.height);
    int max_side = std::max(image_size.width, image_size.height);
    int side = static_cast<int>(base_side + (max_side - base_side) * (dt_clamped / lost_dt));
    if (dt >= lost_dt)
        side = max_side;

    int square_cx = ratio_rect.x + ratio_rect.width / 2;
    int square_cy = ratio_rect.y + ratio_rect.height / 2;
    cv::Rect square(square_cx - side / 2, square_cy - side / 2, side, side);
    square &= img_rect;

    return square;
}
} // namespace awakening::auto_buff
