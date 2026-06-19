#include "armor_target.hpp"
#include "angles.h"
#include "tasks/auto_aim/armor_track/motion_model.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/dta_utils.hpp"
#include "tasks/base/web.hpp"
#include "utils/common/type_common.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"
#include <Eigen/Geometry>
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
namespace awakening::auto_aim {
using namespace armor_point_motion_model;

void ArmorTarget::write_log() {
    Web::write_log("armor_target", [&](auto& j) {
        j["timestamp"] =
            static_cast<int>(std::chrono::duration<double>(last_update.time_since_epoch()).count());
        j["target_number"] = string_by_armor_class(target_number);
        j["track_state"] = TrackState::string_by_state(track_state.tracker_state);
        auto& j_target_state = j["target_state"];
        j_target_state["cx"] = Web::val(target_state.pos().x());
        j_target_state["cy"] = Web::val(target_state.pos().y());
        j_target_state["cz"] = Web::val(target_state.pos().z());
        j_target_state["vx"] = Web::val(target_state.vel().x());
        j_target_state["vy"] = Web::val(target_state.vel().y());
        j_target_state["vz"] = Web::val(target_state.vel().z());
        j_target_state["yaw"] = Web::val(target_state.yaw());
        j_target_state["vyaw"] = Web::val(target_state.vyaw());
        j_target_state["r"] = Web::val(target_state.r());
        j_target_state["l"] = Web::val(target_state.l());
        j_target_state["h"] = Web::val(target_state.h());
        j_target_state["wp"] = Web::val(target_state.w_p());
        j_target_state["wr"] = Web::val(target_state.w_r());
    });
}

void ArmorTarget::reset(
    Armor& a,
    const ArmorTrackerCfg& c,
    const TimePoint& timestamp,
    int frame_id,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) {
    cfg = c;
    uvmeasure_ctx = {
        .armor_num = armor_num_by_armor_class(a.number),
        .id = 0,
        .camera_cv_in_odom = camera_cv_in_odom,
        .camera_info = camera_info.clone(),
        .armor_number = a.number,
    };

    ypdmeasure_ctx = { .armor_num = armor_num_by_armor_class(a.number),
                       .id = 0,
                       .armor_number = a.number };

    target_number = a.number;
    double r_pre;
    Eigen::DiagonalMatrix<double, X_N> p0;
    p0.diagonal().setZero();
    p0.diagonal()[idx::CX] = p0.diagonal()[idx::CY] = p0.diagonal()[idx::CZ] = 1;
    p0.diagonal()[idx::VCX] = p0.diagonal()[idx::VCY] = p0.diagonal()[idx::VCZ] = 64;
    p0.diagonal()[idx::C_ROT_Z] = p0.diagonal()[idx::C_ROT_Y] = p0.diagonal()[idx::C_ROT_X] = 0.4;
    p0.diagonal()[idx::VYAW] = 100;
    if (a.number == ArmorClass::OUTPOST) {
        r_pre = 0.2765;
    } else if (a.number == ArmorClass::BASE) {
        r_pre = 0.3205;
    } else {
        r_pre = 0.26;
    }
    const auto u_q = [this]() {
        Eigen::Matrix<double, X_N, X_N> q;
        return q;
    };

    const auto inject = [this](
                            const Eigen::Matrix<double, X_N, 1>& delta,
                            Eigen::Matrix<double, X_N, 1>& nominal
                        ) {
        for (int i = 0; i < X_N; i++) {
            if (i == idx::CX || i == idx::CY || i == idx::CZ || i == idx::C_ROT_Z
                || i == idx::C_ROT_Y || i == idx::C_ROT_X)
                continue;
            nominal[i] += delta[i];
        }
        Vec3 delta_rho(delta[idx::CX], delta[idx::CY], delta[idx::CZ]);
        Vec3 delta_rot(delta[idx::C_ROT_X], delta[idx::C_ROT_Y], delta[idx::C_ROT_Z]);
        ISO3 nominal_pose = ISO3::Identity();
        nominal_pose.translation() = Vec3(nominal[idx::CX], nominal[idx::CY], nominal[idx::CZ]);
        Vec3 nominal_rot(nominal[idx::C_ROT_X], nominal[idx::C_ROT_Y], nominal[idx::C_ROT_Z]);
        nominal_pose.linear() = utils::so3_exp(nominal_rot);
        const ISO3 injected_pose = nominal_pose * utils::se3_exp(delta_rho, delta_rot);
        const Vec3 injected_rot = utils::so3_log(injected_pose.linear().eval());
        nominal[idx::CX] = injected_pose.translation().x();
        nominal[idx::CY] = injected_pose.translation().y();
        nominal[idx::CZ] = injected_pose.translation().z();
        nominal[idx::C_ROT_X] = injected_rot.x();
        nominal[idx::C_ROT_Y] = injected_rot.y();
        nominal[idx::C_ROT_Z] = injected_rot.z();
    };
    const auto box_minus = [](const Eigen::Matrix<double, X_N, 1>& nominal,
                              const Eigen::Matrix<double, X_N, 1>& value,
                              Eigen::Matrix<double, X_N, 1>& delta) {
        delta = value - nominal;

        ISO3 nominal_pose = ISO3::Identity();
        nominal_pose.translation() = Vec3(nominal[idx::CX], nominal[idx::CY], nominal[idx::CZ]);
        nominal_pose.linear() = utils::so3_exp(
            Vec3(nominal[idx::C_ROT_X], nominal[idx::C_ROT_Y], nominal[idx::C_ROT_Z])
        );

        ISO3 value_pose = ISO3::Identity();
        value_pose.translation() = Vec3(value[idx::CX], value[idx::CY], value[idx::CZ]);
        value_pose.linear() =
            utils::so3_exp(Vec3(value[idx::C_ROT_X], value[idx::C_ROT_Y], value[idx::C_ROT_Z]));

        Vec3 delta_rho;
        Vec3 delta_rot;
        utils::se3_log(nominal_pose.inverse() * value_pose, delta_rho, delta_rot);
        delta[idx::CX] = delta_rho.x();
        delta[idx::CY] = delta_rho.y();
        delta[idx::CZ] = delta_rho.z();
        delta[idx::C_ROT_X] = delta_rot.x();
        delta[idx::C_ROT_Y] = delta_rot.y();
        delta[idx::C_ROT_Z] = delta_rot.z();
    };
    esekf =
        RobotStateESEKF(Predict { .dt = 0.005, .armor_number = target_number }, u_q, inject, p0);
    esekf.value().set_box_minus_state(box_minus);

    esekf.value().set_iteration_num(cfg.esekf_iter_num);

    armor_pnp(a, camera_info, camera_cv_in_odom);
    auto armor_in_odom = a.pose;
    auto armor_in_car = ISO3::Identity();
    const double r = r_pre;
    armor_in_car.translation() << -r, 0, 0;
    const double armor_pitch = (target_number == auto_aim::ArmorClass::OUTPOST)
        ? (-auto_aim::FIFTTEN_DEGREE_RAD)
        : (auto_aim::FIFTTEN_DEGREE_RAD);
    armor_in_car.linear() = utils::rpy2matrix(Vec3(0, armor_pitch, 0));
    auto car_in_odom = armor_in_odom * armor_in_car.inverse();
    auto rpy = utils::matrix2rpy<double>(car_in_odom.linear(), utils::RPYOrder::XYZ);
    const Vec3 car_rot = utils::so3_log(car_in_odom.linear().eval());
    const double yaw = rpy[2];
    last_rot_yaw = yaw;
    target_state.x = Eigen::VectorXd::Zero(X_N);
    target_state.set_pos(car_in_odom.translation());
    target_state.x[idx::R] = r;
    target_state.x[idx::C_ROT_Z] = car_rot.z();
    target_state.x[idx::C_ROT_Y] = car_rot.y();
    target_state.x[idx::C_ROT_X] = car_rot.x();
    target_state.timestamp = timestamp;
    target_state.frame_id = frame_id;
    esekf.value().set_state(target_state.x);
    last_update = timestamp;
    is_inited = true;
    jumped = false;
    last_match_id = -1;
    if (target_number == ArmorClass::OUTPOST) {
        outpost_has_all_and_has_set_ids =
            std::make_pair(false, std::vector<bool>(armor_num(), false));
        outpost_has_all_and_has_set_ids.value().second[0] = true;
    } else {
        outpost_has_all_and_has_set_ids = std::nullopt;
    }
    track_state.reset();
    this_id = GOBAL_ID++; //全局状态标记，下游控制对同一id的不重复构建轨迹
    update_count++;
}

void ArmorTarget::armor_pnp(
    Armor& a,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) noexcept {
    auto key_points = a.key_points.landmarks();
    a.pose = utils::solve_pnp(
        key_points,
        getArmorKeyPoints3D<cv::Point3f>(a.number),
        camera_info.camera_matrix,
        camera_info.distortion_coefficients,
        cv::SOLVEPNP_IPPE
    );
    auto armor_in_odom = camera_cv_in_odom * a.pose;
    a.pose = armor_in_odom;
    if (a.number == ArmorClass::OUTPOST) {
        auto rpy = utils::matrix2rpy<double>(a.pose.linear());
        auto obj_points = getArmorKeyPoints3D<cv::Point3f>(a.number);
        const double armor_pitch = (a.number == auto_aim::ArmorClass::OUTPOST)
            ? -auto_aim::FIFTTEN_DEGREE_RAD
            : auto_aim::FIFTTEN_DEGREE_RAD;
        auto center = [](const cv::Point2f& a, const cv::Point2f& b) { return (a + b) * 0.5f; };
        auto eval_yaw = [&](double yaw) -> double {
            auto a_pose_in_odom = a.pose;
            Vec3 search_rpy(0.0, armor_pitch, yaw);
            a_pose_in_odom.linear() = utils::rpy2matrix(search_rpy);

            auto a_pose_in_camera_cv = camera_cv_in_odom.inverse() * a_pose_in_odom;
            auto img_points = utils::reprojection(
                camera_info.camera_matrix,
                camera_info.distortion_coefficients,
                obj_points,
                a_pose_in_camera_cv
            );

            double error = 0.0;
            // for (int i = 0; i < img_points.size(); i++) {
            //     error += cv::norm(img_points[i] - key_points[i]);
            //     }
            error += cv::norm(
                center(
                    img_points[ArmorKeyPointsIndex::LEFT_TOP],
                    img_points[ArmorKeyPointsIndex::RIGHT_TOP]
                )
                - center(
                    key_points[ArmorKeyPointsIndex::LEFT_TOP],
                    key_points[ArmorKeyPointsIndex::RIGHT_TOP]
                )
            );

            error += cv::norm(
                center(
                    img_points[ArmorKeyPointsIndex::LEFT_BOTTOM],
                    img_points[ArmorKeyPointsIndex::RIGHT_BOTTOM]
                )
                - center(
                    key_points[ArmorKeyPointsIndex::LEFT_BOTTOM],
                    key_points[ArmorKeyPointsIndex::RIGHT_BOTTOM]
                )
            );

            return error;
        };
        constexpr double SEARCH_RANGE_DEG = 140.0;
        constexpr double HALF_RANGE_RAD = SEARCH_RANGE_DEG * CV_PI / 180.0 * 0.5;
        double left = rpy[2] - HALF_RANGE_RAD;
        double right = rpy[2] + HALF_RANGE_RAD;
        double best_yaw = utils::golden_section_search(eval_yaw, left, right, 1e-4);
        auto best_pose = a.pose;
        best_pose.linear() = utils::rpy2matrix(Vec3(0.0, armor_pitch, best_yaw));
        a.pose = best_pose;
    }
}
Eigen::Matrix<double, UVZ_N, UVZ_N>
ArmorTarget::uvmeasurement_covariance(const Eigen::Matrix<double, UVZ_N, 1>& z) const noexcept {
    Eigen::Matrix<double, UVZ_N, UVZ_N> r;
    double u_r = std::max(
        cfg.r_uv_at_1m * log((1.0 / target_state.pos().norm()) + 1),
        cfg.r_uv_min
    ); //比较简陋的逻辑或许应该和预测灯条长度存在比例
    r.setZero();
    r.diagonal().setConstant(u_r);
    return r;
}

[[nodiscard]] Eigen::
    Matrix<double, armor_point_motion_model::YPDZ_N, armor_point_motion_model::YPDZ_N>
    ArmorTarget::ypdmeasurement_covariance(
        const Eigen::Matrix<double, armor_point_motion_model::YPDZ_N, 1>& z
    ) const noexcept {
    Eigen::Matrix<double, YPDZ_N, YPDZ_N> r;
    r.setZero(); //copy下sp_vision_25 这个参数不用在观测，差不多就行
    r(idx::YPD_Y, idx::YPD_Y) = 4e-3;
    r(idx::YPD_P, idx::YPD_P) = 4e-3;
    r(idx::YPD_D, idx::YPD_D) = std::log(z[idx::YPD_D] * z[idx::YPD_D] * 0.1 + 1) + 0.01;
    r(idx::A_ROT_X, idx::A_ROT_X) = 0.1;
    r(idx::A_ROT_Y, idx::A_ROT_Y) = 0.1;
    r(idx::A_ROT_Z, idx::A_ROT_Z) = 0.03;
    return r;
}
[[nodiscard]] Eigen::Matrix<double, armor_point_motion_model::YPDZ_N, 1>
ArmorTarget::get_ypdmeasurement(Armor& a) const noexcept {
    Eigen::Matrix<double, YPDZ_N, 1> z;
    double ax = a.pose.translation().x(), ay = a.pose.translation().y(),
           az = a.pose.translation().z();
    auto ypd_y = std::atan2(ay, ax);
    auto ypd_p = std::atan2(az, std::sqrt(ax * ax + ay * ay));
    auto ypd_d = std::sqrt(ax * ax + ay * ay + az * az);
    z[idx::YPD_Y] = ypd_y;
    z[idx::YPD_P] = ypd_p;
    z[idx::YPD_D] = ypd_d;
    auto rot_vec = utils::so3_log(a.pose.linear().eval());
    z[idx::A_ROT_X] = rot_vec.x();
    z[idx::A_ROT_Y] = rot_vec.y();
    z[idx::A_ROT_Z] = rot_vec.z();
    return z;
}
Eigen::Matrix<double, X_N, X_N> ArmorTarget::process_noise(double dt) const noexcept {
    Eigen::Matrix<double, X_N, X_N> q;
    Vec3 q_xyz;
    double q_yaw;
    double q_l, q_h;
    if (target_number == ArmorClass::OUTPOST) {
        q_xyz = cfg.qxyz_output; // 前哨站加速度方差
        q_yaw = cfg.qyaw_output; // 前哨站角加速度方差
        q_l = cfg.q_outpost_dz;
        q_h = cfg.q_outpost_dz;
    } else {
        q_xyz = cfg.qxyz_common; // 加速度方差
        q_yaw = cfg.qyaw_common; // 角加速度方差
        q_l = cfg.q_l;
        q_h = cfg.q_h;
    }

    q.setZero();
    const Mat3 car_in_odom_R = _whole_car_pose(target_state.x.data(), target_number).linear();
    const Mat3 accel_noise_car = q_xyz.asDiagonal();
    const Mat3 accel_noise_odom = car_in_odom_R * accel_noise_car * car_in_odom_R.transpose();

    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;
    constexpr std::array<int, 3> pos_idx { idx::CX, idx::CY, idx::CZ };
    constexpr std::array<int, 3> vel_idx { idx::VCX, idx::VCY, idx::VCZ };
    const Mat3 pos_vel_noise = accel_noise_car * car_in_odom_R.transpose();
    const Mat3 vel_pos_noise = car_in_odom_R * accel_noise_car;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            q(pos_idx[i], pos_idx[j]) = dt4 * 0.25 * accel_noise_car(i, j);
            q(pos_idx[i], vel_idx[j]) = dt3 * 0.5 * pos_vel_noise(i, j);
            q(vel_idx[i], pos_idx[j]) = dt3 * 0.5 * vel_pos_noise(i, j);
            q(vel_idx[i], vel_idx[j]) = dt2 * accel_noise_odom(i, j);
        }
    }

    utils::fill_constant_accel_noise(q, idx::C_ROT_Z, idx::VYAW, q_yaw, dt);

    q(idx::R, idx::R) = cfg.q_r;
    q(idx::L, idx::L) = q_l;
    q(idx::H, idx::H) = q_h;
    q(idx::C_ROT_Y, idx::C_ROT_Y) += dt * cfg.q_wpr;
    q(idx::C_ROT_X, idx::C_ROT_X) += dt * cfg.q_wpr;
    return q;
}

void ArmorTarget::predict_ekf(const TimePoint& timestamp) {
    if (!esekf) {
        throw std::runtime_error("ESEKF is not initialized");
    }
    auto dt = std::chrono::duration<double>(timestamp - target_state.timestamp).count();
    esekf.value().set_predict_func(Predict { .dt = dt, .armor_number = target_number });
    esekf.value().set_update_Q([&]() { return process_noise(dt); });
    target_state.x = esekf.value().predict();
    target_state.timestamp = timestamp;
    this_id = GOBAL_ID++; //全局状态标记，下游控制对同一id的不重复构建轨迹
}

int ArmorTarget::update(
    std::vector<std::pair<int, Armor>>& matched_armors,
    std::vector<std::tuple<int, bool, Light>>& matched_lights,
    const TimePoint& timestamp,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) {
    if (matched_armors.empty()) {
        return 0;
    }

    std::vector<std::shared_ptr<RobotStateESEKF::ObsBase>> obs;

    const auto u_r = [&](const Eigen::Matrix<double, UVZ_N, 1>& z) {
        return uvmeasurement_covariance(z);
    };

    const auto cal_residual = [](const Eigen::Matrix<double, UVZ_N, 1>& z_pred,
                                 const Eigen::Matrix<double, UVZ_N, 1>& z) { return z - z_pred; };

    auto add_armor_uv_obs = [&](Armor& armor, int id, bool is_left) {
        auto ctx = uvmeasure_ctx;
        ctx.id = id;
        ctx.is_left = is_left;
        ctx.camera_cv_in_odom = camera_cv_in_odom;
        Eigen::Matrix<double, UVZ_N, 1> z;
        auto key_points = armor.key_points.landmarks();
        if (is_left) {
            z[idx::TOP_X] = key_points[ArmorKeyPointsIndex::LEFT_TOP].x;
            z[idx::TOP_Y] = key_points[ArmorKeyPointsIndex::LEFT_TOP].y;
            z[idx::BOTTOM_X] = key_points[ArmorKeyPointsIndex::LEFT_BOTTOM].x;
            z[idx::BOTTOM_Y] = key_points[ArmorKeyPointsIndex::LEFT_BOTTOM].y;
        } else {
            z[idx::BOTTOM_X] = key_points[ArmorKeyPointsIndex::RIGHT_BOTTOM].x;
            z[idx::BOTTOM_Y] = key_points[ArmorKeyPointsIndex::RIGHT_BOTTOM].y;
            z[idx::TOP_X] = key_points[ArmorKeyPointsIndex::RIGHT_TOP].x;
            z[idx::TOP_Y] = key_points[ArmorKeyPointsIndex::RIGHT_TOP].y;
        }
        UVMeasure measure { .ctx = ctx };
        obs.push_back(esekf.value().make_obs(z, measure, u_r, cal_residual));
    };
    auto add_armor_ypd_obs = [&](Armor& armor, int id) {
        auto ctx = ypdmeasure_ctx;
        ctx.id = id;
        const auto ypd_u_r = [&](const Eigen::Matrix<double, YPDZ_N, 1>& z) {
            return ypdmeasurement_covariance(z);
        };

        const auto ypd_cal_residual = [](const Eigen::Matrix<double, YPDZ_N, 1>& z_pred,
                                         const Eigen::Matrix<double, YPDZ_N, 1>& z) {
            Eigen::Matrix<double, YPDZ_N, 1> v = z - z_pred;

            v[idx::YPD_Y] = angles::normalize_angle(v[idx::YPD_Y]);
            auto so3_residual = utils::so3_log(
                (utils::so3_exp(Vec3(z[idx::A_ROT_X], z[idx::A_ROT_Y], z[idx::A_ROT_Z]))
                 * utils::so3_exp(
                       Vec3(z_pred[idx::A_ROT_X], z_pred[idx::A_ROT_Y], z_pred[idx::A_ROT_Z])
                 )
                       .transpose())
                    .eval()
            );
            v[idx::A_ROT_X] = so3_residual.x();
            v[idx::A_ROT_Y] = so3_residual.y();
            v[idx::A_ROT_Z] = so3_residual.z();
            return v;
        };
        YPDMeasure measure { .ctx = ctx };
        obs.push_back(
            esekf.value().make_obs(get_ypdmeasurement(armor), measure, ypd_u_r, ypd_cal_residual)
        );
    };

    auto update_outpost_state = [&](int id) {
        if (!outpost_has_all_and_has_set_ids || outpost_has_all_and_has_set_ids->first) {
            return;
        }
        auto& has_ids = outpost_has_all_and_has_set_ids->second;
        has_ids[id] = true;
        if (std::all_of(has_ids.begin(), has_ids.end(), [](bool v) { return v; })) {
            outpost_has_all_and_has_set_ids->first = true;
        }
    };

    int updated = 0;
    std::vector<bool> used_id(armor_num(), false);
    for (auto& [id, armor]: matched_armors) {
        jumped |= (id != 0);
        update_outpost_state(id);
        last_match_id = id;
        used_id[id] = true;
        add_armor_uv_obs(armor, id, true);
        add_armor_uv_obs(armor, id, false);
        ++updated;
        ++update_count;
    }
    for (const auto& [id, is_left, light]: matched_lights) {
        auto ctx = uvmeasure_ctx;
        ctx.id = id;
        ctx.is_left = is_left;
        ctx.camera_cv_in_odom = camera_cv_in_odom;
        UVMeasure measure { .ctx = ctx };
        Eigen::Matrix<double, UVZ_N, 1> z;
        z[std::to_underlying(idx::TOP_X)] = light.top.x;
        z[std::to_underlying(idx::TOP_Y)] = light.top.y;
        z[std::to_underlying(idx::BOTTOM_X)] = light.bottom.x;
        z[std::to_underlying(idx::BOTTOM_Y)] = light.bottom.y;
        obs.push_back(esekf.value().make_obs(z, measure, u_r, cal_residual));
    }
    if (obs.size() > 0) {
        target_state.x = esekf.value().update_multi(obs);
        target_state.timestamp = timestamp;
        last_update = timestamp;
        this_id = GOBAL_ID++; //全局状态标记，下游控制对同一id的不重复构建轨迹
    }

    return updated;
}
std::vector<std::pair<int, Armor>> ArmorTarget::match_armor(
    std::vector<Armor>& armors,
    const TimePoint& timestamp,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) const noexcept {
    constexpr double MAX_COST = 1e9;
    std::vector<std::pair<int, Armor>> result;
    const int n_obs = static_cast<int>(armors.size());
    const int armors_num = armor_num();
    bool all_init =
        (outpost_has_all_and_has_set_ids.has_value() ? outpost_has_all_and_has_set_ids.value().first
                                                     : jumped);
    const double GATE = (all_init ? cfg.match_gate_armor : cfg.match_gate_not_all_init_armor);

    std::vector<std::vector<double>> cost(n_obs, std::vector<double>(armors_num, MAX_COST + 1));

    std::vector<YPDVecZ> meas_list(n_obs
    ); //纯图像点匹配只能纯位置误差，要不就是和match_light基于逻辑，不如随便pnp一下ypda匹配
    for (int j = 0; j < n_obs; ++j) {
        armor_pnp(armors[j], camera_info, camera_cv_in_odom);
        meas_list[j] = get_ypdmeasurement(armors[j]);
    }
    auto pred_state = target_state;
    pred_state.predict(timestamp, target_number);
    for (int j = 0; j < n_obs; ++j) {
        if (armors[j].number == ArmorClass::OUTPOST) {
            auto rpy = utils::matrix2rpy<double>(armors[j].pose.linear());
            if (rpy[1] > 0) {
                continue;
            }
        }
        bool in_gate = false;
        double min_d2 = std::numeric_limits<double>::max();
        for (int id = 0; id < armors_num; ++id) {
            YPDMeasure::Ctx tmp_ctx {
                .armor_num = armors_num,
                .id = id,
                .armor_number = target_number,

            };
            YPDMeasure measure { .ctx = tmp_ctx };
            YPDVecZ z_pred;
            measure.h(pred_state.x, z_pred);

            YPDVecZ nu = meas_list[j] - z_pred;
            nu[idx::YPD_Y] = angles::normalize_angle(nu[idx::YPD_Y]);
            auto so3_residual = utils::so3_log(
                (utils::so3_exp(Vec3(
                     meas_list[j][idx::A_ROT_X],
                     meas_list[j][idx::A_ROT_Y],
                     meas_list[j][idx::A_ROT_Z]
                 ))
                 * utils::so3_exp(
                       Vec3(z_pred[idx::A_ROT_X], z_pred[idx::A_ROT_Y], z_pred[idx::A_ROT_Z])
                 )
                       .transpose())
                    .eval()
            );
            nu[idx::A_ROT_X] = so3_residual.x();
            nu[idx::A_ROT_Y] = so3_residual.y();
            nu[idx::A_ROT_Z] = so3_residual.z();

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

    for (auto [obs, id]: dta_utils::greedy_match(cost, n_obs, armors_num, MAX_COST)) {
        result.emplace_back(id, armors[obs]);
    }
    return result;
}
std::vector<std::tuple<int, bool, Light>> ArmorTarget::match_light(
    std::vector<Light>& lights,
    std::vector<std::pair<int, Armor>>& matched_armors,
    const TimePoint& timestamp,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) const noexcept {
    constexpr double MAX_COST = 1e9;
    //可见灯条逻辑判断不优雅，不过这比较个稳定可观
    std::vector<std::tuple<int, bool, Light>> result;

    if (target_number == ArmorClass::BASE || matched_armors.size() != 1) {
        return result;
    }
    auto pred_state = target_state;
    pred_state.predict(timestamp, target_number);
    const int armors_num = armor_num();
    const int armor_id = matched_armors.front().first;
    auto predict_light = [&](int id, bool is_left) -> std::pair<cv::Point2f, cv::Point2f> {
        UVMeasure::Ctx ctx { .armor_num = armors_num,
                             .id = id,
                             .camera_cv_in_odom = camera_cv_in_odom,
                             .camera_info = camera_info,
                             .armor_number = target_number,
                             .is_left = is_left };
        UVMeasure measure { .ctx = ctx };
        UVVecZ z;
        measure.h(pred_state.x, z);

        return {
            cv::Point2f(z[std::to_underlying(idx::TOP_X)], z[std::to_underlying(idx::TOP_Y)]),
            cv::Point2f(z[std::to_underlying(idx::BOTTOM_X)], z[std::to_underlying(idx::BOTTOM_Y)])
        };
    };
    const std::array visible_mapping {
        std::pair { (armor_id + 3) % 4, false }, // 左可见
        std::pair { (armor_id + 1) % 4, true }, // 右可见
    };
    std::array<std::pair<cv::Point2f, cv::Point2f>, visible_mapping.size()> visible_lights {
        predict_light(visible_mapping[0].first, visible_mapping[0].second), // 左可见
        predict_light(visible_mapping[1].first, visible_mapping[1].second), // 右可见
    };

    const int n_obs = static_cast<int>(lights.size());
    std::vector<std::array<double, visible_lights.size()>> cost(
        n_obs,
        { MAX_COST + 1, MAX_COST + 1 }
    );

    auto calc_cost = [&](const Light& light,
                         const std::pair<cv::Point2f, cv::Point2f>& pred) -> double {
        if (matched_armors[0].second.key_points.bounding_box().contains(light.top)
            || matched_armors[0].second.key_points.bounding_box().contains(light.bottom)
            || matched_armors[0].second.key_points.bounding_box().contains(light.center))
        {
            return MAX_COST + 1;
        }
        double pred_len = cv::norm(pred.first - pred.second);
        double len_err = std::abs(light.length - pred_len);
        if (len_err > pred_len * cfg.light_match_length_ratio_gate) {
            // AWAKENING_WARN("match out of gate: light length err: {}", len_err);
            return MAX_COST + 1;
        }

        double pred_angle = std::atan2(pred.first.x - pred.second.x, pred.first.y - pred.second.y);
        double light_angle = std::atan2(light.top.x - light.bottom.x, light.top.y - light.bottom.y);
        double angle_err = std::abs(angles::normalize_angle(light_angle - pred_angle));
        if (angle_err > cfg.light_match_angle_gate) {
            // AWAKENING_WARN("match out of gate: light angle err: {}", angle_err);
            return MAX_COST + 1;
        }

        double pos_err = cv::norm(light.top - pred.first) + cv::norm(light.bottom - pred.second);
        if (pos_err > pred_len * cfg.light_match_pos_gate_by_length_ratio) {
            // AWAKENING_WARN("match out of gate: light position err: {}", pos_err);
            return MAX_COST + 1;
        }

        return pos_err;
    };

    for (int j = 0; j < n_obs; ++j) {
        for (std::size_t i = 0; i < visible_lights.size(); ++i) {
            cost[j][i] = calc_cost(lights[j], visible_lights[i]);
        }
    }

    for (auto [obs, id]: dta_utils::greedy_match(cost, n_obs, 2, MAX_COST)) {
        auto [matched_id, is_left] = visible_mapping[id];
        result.emplace_back(matched_id, is_left, lights[obs]);
    }

    return result;
}
[[nodiscard]] cv::Rect ArmorTarget::get_net_focus_roi(
    const TimePoint& timestamp,
    const ISO3& camera_cv_in_odom,
    const CameraInfo& camera_info,
    const cv::Size& image_size,
    double target_wh_ratio
) const noexcept {
    if (!need_focus()) {
        return cv::Rect(0, 0, image_size.width, image_size.height);
    }

    cv::Rect rect = expanded(timestamp, camera_cv_in_odom, camera_info, image_size);
    constexpr double expansion_ratio = 0.2;
    rect.x -= rect.width * expansion_ratio;
    rect.y -= rect.height * expansion_ratio;
    rect.width *= (1.0 + 2.0 * expansion_ratio);
    rect.height *= (1.0 + 2.0 * expansion_ratio);
    const cv::Rect img_rect = cv::Rect(0, 0, image_size.width, image_size.height);

    if ((rect & img_rect).area() <= 0) {
        return img_rect;
    }

    cv::Rect expanded_rect = rect & img_rect;

    const double rect_w = std::max<double>(expanded_rect.width, 1.0);
    const double rect_h = std::max<double>(expanded_rect.height, 1.0);
    const double ratio =
        (std::isfinite(target_wh_ratio) && target_wh_ratio > 0.0) ? target_wh_ratio : 1.0;

    double target_w = rect_w;
    double target_h = rect_h;
    if (target_w / target_h < ratio) {
        target_w = target_h * ratio;
    } else {
        target_h = target_w / ratio;
    }
    const double cx = expanded_rect.x + expanded_rect.width / 2.0;
    const double cy = expanded_rect.y + expanded_rect.height / 2.0;
    cv::Rect ratio_rect(
        static_cast<int>(cx - target_w / 2.0),
        static_cast<int>(cy - target_h / 2.0),
        static_cast<int>(target_w),
        static_cast<int>(target_h)
    );
    ratio_rect &= img_rect;
    if ((rect & img_rect).area() <= 0) {
        return img_rect;
    }
    return ratio_rect;
}
[[nodiscard]] cv::Rect ArmorTarget::expanded(
    const TimePoint& timestamp,
    const ISO3& camera_cv_in_odom,
    const CameraInfo& camera_info,
    const cv::Size& image_size
) const noexcept {
    std::vector<cv::Point2f> pts;
    auto tmp_target_state = target_state;
    tmp_target_state.predict(timestamp, target_number);
    const int armors_num = armor_num();
    for (int id = 0; id < armors_num; ++id) {
        UVMeasure::Ctx tmp_ctx {
            .armor_num = armors_num,
            .id = id,
            .camera_cv_in_odom = camera_cv_in_odom,
            .camera_info = camera_info,
            .armor_number = target_number,
        };
        UVMeasure measure { .ctx = tmp_ctx };
        UVVecZ z_pred;
        {
            measure.ctx.is_left = true;
            measure.h(tmp_target_state.x, z_pred);
            pts.push_back(cv::Point2f(z_pred[idx::TOP_X], z_pred[idx::TOP_Y]));
            pts.push_back(cv::Point2f(z_pred[idx::BOTTOM_X], z_pred[idx::BOTTOM_Y]));
        }
        {
            measure.ctx.is_left = false;
            measure.h(tmp_target_state.x, z_pred);
            pts.push_back(cv::Point2f(z_pred[idx::TOP_X], z_pred[idx::TOP_Y]));
            pts.push_back(cv::Point2f(z_pred[idx::BOTTOM_X], z_pred[idx::BOTTOM_Y]));
        }
    }
    cv::Rect rect = cv::boundingRect(pts);
    const cv::Rect img_rect = cv::Rect(0, 0, image_size.width, image_size.height);
    if ((rect & img_rect).area() <= 0) {
        return img_rect;
    }
    return rect;
}
} // namespace awakening::auto_aim
