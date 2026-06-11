#include "rune_target.hpp"
#include "tasks/auto_buff/rune_track/motion_model.hpp"
#include "tasks/auto_buff/type.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <vector>
namespace awakening::auto_buff {
using namespace motion_model;
void RuneTarget::reset(
    RuneFanBladeWithR& f,
    const RuneTrackerCfg& c,
    const TimePoint& timestamp,
    int frame_id,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) {
    cfg = c;
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
    esekf = ESEKF(Predict { .dt = 0.005 }, u_q, inject, p0);

    esekf.value().set_iteration_num(cfg.esekf_iter_num);
    fan_pnp(f, camera_info, camera_cv_in_odom);
    auto pos = f.pose.translation();
    auto rpy = utils::matrix2rpy(f.pose.linear());
    target_state.x = Eigen::VectorXd::Zero(X_N);
    target_state.x << pos.x(), pos.y(), pos.z(), rpy[2], rpy[0], 0;
    target_state.timestamp = timestamp;
    target_state.frame_id = frame_id;
    esekf.value().set_state(target_state.x);
    last_update = timestamp;
    is_inited = true;
    track_state.reset();
    this_id = GOBAL_ID++;
}
void RuneTarget::fan_pnp(
    RuneFanBladeWithR& r,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) noexcept {
    auto key_points = r.points;
    r.pose = utils::solve_pnp(
        key_points,
        RuneKeyPoint3D<cv::Point3f>::build(),
        camera_info.camera_matrix,
        camera_info.distortion_coefficients
    );
    r.pose = camera_cv_in_odom * r.pose;
}
[[nodiscard]] Eigen::Matrix<double, motion_model::X_N, motion_model::X_N>
RuneTarget::process_noise(double dt) const noexcept {
    Eigen::Matrix<double, X_N, X_N> q;
    q.setZero();
    q(idx::CX, idx::CX) = cfg.q_xyz.x();
    q(idx::CY, idx::CY) = cfg.q_xyz.y();
    q(idx::CZ, idx::CZ) = cfg.q_xyz.z();
    q(idx::YAW, idx::YAW) = cfg.q_yaw;
    utils::fill_constant_accel_noise(q, idx::ROLL, idx::V_ROLL, cfg.q_roll, dt);
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
    r(idx::ROT_YAW, idx::ROT_YAW) = 0.05;
    r(idx::ROT_ROLL, idx::ROT_ROLL) = 0.05;
    return r;
}
[[nodiscard]] Eigen::Matrix<double, motion_model::YPDZ_N, 1>
RuneTarget::get_ypdmeasurement(RuneFanBladeWithR& fan) const noexcept {
    Eigen::Matrix<double, YPDZ_N, 1> z;
    double ax = fan.pose.translation().x(), ay = fan.pose.translation().y(),
           az = fan.pose.translation().z();
    auto ypd_y = std::atan2(ay, ax);
    static double last_ypd_y = 0;
    ypd_y = last_ypd_y + angles::shortest_angular_distance(last_ypd_y, ypd_y);
    last_ypd_y = ypd_y;
    auto ypd_p = std::atan2(az, std::sqrt(ax * ax + ay * ay));
    auto ypd_d = std::sqrt(ax * ax + ay * ay + az * az);
    z[idx::YPD_Y] = ypd_y;
    z[idx::YPD_P] = ypd_p;
    z[idx::YPD_D] = ypd_d;
    auto rpy = utils::matrix2rpy(fan.pose.linear());
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
    esekf.value().set_predict_func(Predict {
        .dt = dt,
    });
    esekf.value().set_update_Q([&]() { return process_noise(dt); });
    target_state.x = esekf.value().predict();
    target_state.timestamp = timestamp;
    this_id = GOBAL_ID++; //全局状态标记，下游控制对同一id的不重复构建轨迹
}
int RuneTarget::update(
    std::vector<std::pair<int, RuneFanBladeWithR>>& matched_fans,
    const TimePoint& timestamp,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) {
    if (matched_fans.empty()) {
        return 0;
    }
    std::vector<std::shared_ptr<ESEKF::ObsBase>> obs;
    const auto fan_u_r = [&](const Eigen::Matrix<double, FanBladeZ_N, 1>& z) {
        Eigen::Matrix<double, FanBladeZ_N, FanBladeZ_N> r;
        r.setZero();
        r.diagonal().setConstant(70.0);
        return r;
    };

    const auto fan_cal_residual = [](const Eigen::Matrix<double, FanBladeZ_N, 1>& z_pred,
                                     const Eigen::Matrix<double, FanBladeZ_N, 1>& z) {
        return z - z_pred;
    };
    std::vector<cv::Point2f> r;
    for (const auto& [id, fan]: matched_fans) {
        FanBladeVecZ z;
        z[idx::TOP_X] = fan.points[RuneKeyPointsIndex::TOP].x;
        z[idx::TOP_Y] = fan.points[RuneKeyPointsIndex::TOP].y;
        z[idx::LEFT_X] = fan.points[RuneKeyPointsIndex::LEFT].x;
        z[idx::LEFT_Y] = fan.points[RuneKeyPointsIndex::LEFT].y;
        z[idx::RIGHT_X] = fan.points[RuneKeyPointsIndex::RIGHT].x;
        z[idx::RIGHT_Y] = fan.points[RuneKeyPointsIndex::RIGHT].y;
        z[idx::BOTTOM_X] = fan.points[RuneKeyPointsIndex::BOTTOM].x;
        z[idx::BOTTOM_Y] = fan.points[RuneKeyPointsIndex::BOTTOM].y;
        z[idx::CENTER_X] = fan.points[RuneKeyPointsIndex::CENTER].x;
        z[idx::CENTER_Y] = fan.points[RuneKeyPointsIndex::CENTER].y;
        r.push_back(fan.points[RuneKeyPointsIndex::R]);
        auto ctx = fan_ctx;
        ctx.id = id;
        ctx.camera_cv_in_odom = camera_cv_in_odom;
        FanBladeMeasure measure { .ctx = ctx };
        obs.push_back(esekf.value().make_obs(z, measure, fan_u_r, fan_cal_residual));
    }
    if (!r.empty()) {
        auto avg_r = std::accumulate(r.begin(), r.end(), cv::Point2f(0.f, 0.f)) * (1.0 / r.size());
        RVecZ z;
        z[idx::R_X] = avg_r.x;
        z[idx::R_Y] = avg_r.y;
        auto ctx = r_ctx;
        RMeasure measure { .ctx = ctx };
        const auto r_u_r = [&](const Eigen::Matrix<double, RZ_N, 1>& z) {
            Eigen::Matrix<double, RZ_N, RZ_N> r;
            r.setZero();
            r.diagonal().setConstant(70.0);
            return r;
        };

        const auto r_cal_residual = [](const Eigen::Matrix<double, RZ_N, 1>& z_pred,
                                       const Eigen::Matrix<double, RZ_N, 1>& z) {
            return z - z_pred;
        };
        obs.push_back(esekf.value().make_obs(z, measure, r_u_r, r_cal_residual));
    }
    target_state.x = esekf.value().update_multi(obs);
    target_state.timestamp = timestamp;
    last_update = timestamp;
    this_id = GOBAL_ID++; //全局状态标记，下游控制对同一id的不重复构建轨迹
    return obs.size();
}
std::vector<std::pair<int, RuneFanBladeWithR>> RuneTarget::match_fan(
    std::vector<RuneFanBladeWithR>& fans,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) const noexcept {
    constexpr double MAX_COST = 1e9;
    std::vector<std::pair<int, RuneFanBladeWithR>> result;
    const int n_obs = static_cast<int>(fans.size());

    const double GATE = 10.0;

    std::vector<std::vector<double>> cost(n_obs, std::vector<double>(FAN_NUM, MAX_COST + 1));

    std::vector<YPDVecZ> meas_list(n_obs
    ); //纯图像点匹配只能纯位置误差，要不就是和match_light基于逻辑，不如随便pnp一下ypda匹配
    for (int j = 0; j < n_obs; ++j) {
        fan_pnp(fans[j], camera_info, camera_cv_in_odom);
        meas_list[j] = get_ypdmeasurement(fans[j]);
    }

    for (int j = 0; j < n_obs; ++j) {
        bool in_gate = false;
        double min_d2 = std::numeric_limits<double>::max();
        for (int id = 0; id < FAN_NUM; ++id) {
            YPDMeasure::Ctx tmp_ctx {
                .id = id,

            };
            YPDMeasure measure { .ctx = tmp_ctx };
            YPDVecZ z_pred;
            measure.h(target_state.x, z_pred);

            YPDVecZ nu = meas_list[j] - z_pred;
            nu[idx::YPD_Y] = angles::normalize_angle(nu[idx::YPD_Y]);
            nu[idx::ROT_YAW] = angles::normalize_angle(nu[idx::ROT_YAW]);
            nu[idx::ROT_ROLL] = angles::normalize_angle(nu[idx::ROT_ROLL]);
            auto R = ypdmeasurement_covariance(z_pred);
            double d2 = nu.transpose() * R.ldlt().solve(nu);

            if (std::isfinite(d2) && d2 < GATE) {
                cost[j][id] = d2;
                in_gate = true;
            }
            if (d2 < min_d2) {
                min_d2 = d2;
            }
        }
        if (!in_gate) {
            AWAKENING_WARN("match out of gate min d2: {}", min_d2);
        }
    }

    std::vector<bool> used_obs(n_obs, false);
    std::vector<bool> used_id(FAN_NUM, false);

    while (true) {
        double best = MAX_COST;
        int best_j = -1;
        int best_id = -1;

        for (int j = 0; j < n_obs; ++j) {
            if (used_obs[j])
                continue;
            for (int id = 0; id < FAN_NUM; ++id) {
                if (used_id[id])
                    continue;
                if (cost[j][id] < best) {
                    best = cost[j][id];
                    best_j = j;
                    best_id = id;
                }
            }
        }

        if (best_j < 0 || best_id < 0) {
            break;
        }

        used_obs[best_j] = true;
        used_id[best_id] = true;
        result.push_back(std::make_pair(best_id, fans[best_j]));
    }
    return result;
}
} // namespace awakening::auto_buff