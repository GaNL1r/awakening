#include "armor_target.hpp"
#include "angles.h"
#include "tasks/auto_aim/armor_track/motion_model.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "utils/common/type_common.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
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
    if (a.number == ArmorClass::OUTPOST) {
        p0.diagonal() << 1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0.1, 0.1;
        r_pre = 0.2765;
    } else if (a.number == ArmorClass::BASE) {
        p0.diagonal() << 1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0;
        r_pre = 0.3205;
    } else {
        p0.diagonal() << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
        r_pre = 0.26;
    }
    const auto u_q = [this]() {
        Eigen::Matrix<double, X_N, X_N> q;
        return q;
    };

    const auto inject =
        [this](const Eigen::Matrix<double, X_N, 1>& delta, Eigen::Matrix<double, X_N, 1>& nominal) {
            for (int i = 0; i < X_N; i++) {
                if (i == idx::YAW)
                    continue;
                nominal[i] += delta[i];
            }
            nominal[idx::YAW] = angles::normalize_angle(nominal[idx::YAW] + delta[idx::YAW]);
        };
    esekf =
        RobotStateESEKF(Predict { .dt = 0.005, .armor_number = target_number }, u_q, inject, p0);

    esekf.value().set_iteration_num(cfg.esekf_iter_num);

    armor_pnp(a, camera_info, camera_cv_in_odom);
    auto pos = a.pose.translation();
    const double xa = pos.x();
    const double ya = pos.y();
    const double za = pos.z();
    auto rpy = utils::matrix2euler(a.pose.linear(), utils::EulerOrder::XYZ);
    const double yaw = rpy[2];
    last_rot_yaw = yaw;
    target_state.x = Eigen::VectorXd::Zero(X_N);
    const double r = r_pre;
    const double xc = xa + r * cos(yaw);
    const double yc = ya + r * sin(yaw);
    const double zc = za;
    double l = 0.0;
    double h = 0.0;
    double w_r = 0.0;
    double w_p = 0.0;
    target_state.x << xc, 0, yc, 0, zc, 0, yaw, 0, r, l, h, w_r, w_p;
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
    this_id = GOBAL_ID++;
    update_count++;
}

void ArmorTarget::armor_pnp(
    Armor& a,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) noexcept {
    cv::Mat rvec, tvec;
    auto key_points = a.key_points.landmarks();
    if (!cv::solvePnP(
            getArmorKeyPoints3D<cv::Point3f>(a.number),
            key_points,
            camera_info.camera_matrix,
            camera_info.distortion_coefficients,
            rvec,
            tvec,
            false,
            cv::SOLVEPNP_IPPE
        ))
    {}
    cv::Mat R_cv_armor_in_camera_cv;
    cv::Rodrigues(rvec, R_cv_armor_in_camera_cv);
    Mat3 R_eigen_armor_in_camera_cv;
    cv::cv2eigen(R_cv_armor_in_camera_cv, R_eigen_armor_in_camera_cv);
    Vec3 t_eigen_armor_in_camera_cv;
    cv::cv2eigen(tvec, t_eigen_armor_in_camera_cv);
    a.pose.translation() = t_eigen_armor_in_camera_cv;
    a.pose.linear() = R_eigen_armor_in_camera_cv;
    auto armor_in_odom = camera_cv_in_odom * a.pose;
    a.pose = armor_in_odom;
    auto rpy = utils::matrix2euler(a.pose.linear(), utils::EulerOrder::XYZ);
    double yaw_raw = rpy[2];
    constexpr double SEARCH_RANGE = 140; // degree
    auto yaw0 = angles::normalize_angle(yaw_raw - SEARCH_RANGE / 2 * CV_PI / 180.0);
    auto min_error = 1e10;
    auto best_rot = a.pose.linear();
    auto obj_points = getArmorKeyPoints3D<cv::Point3f>(a.number);
    double best_yaw = yaw_raw;
    const double armor_pitch = (a.number == auto_aim::ArmorClass::OUTPOST)
        ? -auto_aim::FIFTTEN_DEGREE_RAD
        : auto_aim::FIFTTEN_DEGREE_RAD;
    for (int i = 0; i < SEARCH_RANGE; i++) {
        double yaw = angles::normalize_angle(yaw0 + i * CV_PI / 180.0);
        auto a_pose_in_odom = a.pose;
        auto search_ypr = Vec3(yaw, armor_pitch, rpy[0]);
        a_pose_in_odom.linear() = utils::euler2matrix(search_ypr, utils::EulerOrder::ZYX);
        auto a_pose_in_camera_cv = camera_cv_in_odom.inverse() * a_pose_in_odom;
        auto img_points = utils::reprojection(
            camera_info.camera_matrix,
            camera_info.distortion_coefficients,
            obj_points,
            a_pose_in_camera_cv
        );
        double error = 0;
        // for (int i = 0; i < img_points.size(); i++) {
        //     error += cv::norm(img_points[i] - key_points[i]);
        // }
        auto center = [](const cv::Point2f& a, const cv::Point2f& b) { return (a + b) * 0.5f; };
        auto quad_center = [](const cv::Point2f& lt,
                              const cv::Point2f& lb,
                              const cv::Point2f& rt,
                              const cv::Point2f& rb) { return (lt + lb + rt + rb) * 0.25f; };
        error += cv::norm(
            center(
                img_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_TOP)],
                img_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_TOP)]
            )
            - center(
                key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_TOP)],
                key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_TOP)]
            )
        );
        error += cv::norm(
            center(
                img_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_BOTTOM)],
                img_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_BOTTOM)]
            )
            - center(
                key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_BOTTOM)],
                key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_BOTTOM)]
            )
        );

        if (error < min_error) {
            min_error = error;
            best_rot = a_pose_in_odom.linear();
            best_yaw = yaw;
        }
    }
    a.pose.linear() = best_rot;
}

Eigen::Matrix<double, UVZ_N, UVZ_N>
ArmorTarget::uvmeasurement_covariance(const Eigen::Matrix<double, UVZ_N, 1>& z) const noexcept {
    Eigen::Matrix<double, UVZ_N, UVZ_N> r;

    double u_r =
        std::max(cfg.r_uv_at_1m * log((1.0 / target_state.pos().norm()) + 1), cfg.r_uv_min);

    r.setZero();
    r.diagonal().setConstant(u_r);
    return r;
}
[[nodiscard]] Eigen::Matrix<double, UVZ_N, 1>
ArmorTarget::get_uvmeasurement(Armor& a, bool left) const noexcept {
    Eigen::Matrix<double, UVZ_N, 1> z;
    auto key_points = a.key_points.landmarks();
    if (left) {
        z[idx::TOP_X] = key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_TOP)].x;
        z[idx::TOP_Y] = key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_TOP)].y;
        z[idx::BOTTOM_X] = key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_BOTTOM)].x;
        z[idx::BOTTOM_Y] = key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_BOTTOM)].y;
    } else {
        z[idx::BOTTOM_X] = key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_BOTTOM)].x;
        z[idx::BOTTOM_Y] = key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_BOTTOM)].y;
        z[idx::TOP_X] = key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_TOP)].x;
        z[idx::TOP_Y] = key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_TOP)].y;
    }

    return z;
}

[[nodiscard]] Eigen::
    Matrix<double, armor_point_motion_model::YPDZ_N, armor_point_motion_model::YPDZ_N>
    ArmorTarget::ypdmeasurement_covariance(
        const Eigen::Matrix<double, armor_point_motion_model::YPDZ_N, 1>& z
    ) const noexcept {
    Eigen::Matrix<double, YPDZ_N, YPDZ_N> r;
    const double delta_angle = angles::normalize_angle(z[idx::ROT_YAW] - z[idx::YPD_Y]);
    r.setZero();
    r(idx::YPD_Y, idx::YPD_Y) = 4e-3;
    r(idx::YPD_P, idx::YPD_P) = 4e-3;
    r(idx::YPD_D, idx::YPD_D) =
        log(std::abs(delta_angle) + 1) + z[idx::YPD_D] * z[idx::YPD_D] * 0.1;
    r(idx::ROT_YAW, idx::ROT_YAW) = log(std::abs(z[idx::YPD_D]) + 1) / 200 + 9e-2;

    return r;
}
[[nodiscard]] Eigen::Matrix<double, armor_point_motion_model::YPDZ_N, 1>
ArmorTarget::get_ypdmeasurement(Armor& a) const noexcept {
    Eigen::Matrix<double, YPDZ_N, 1> z;
    double ax = a.pose.translation().x(), ay = a.pose.translation().y(),
           az = a.pose.translation().z();
    auto ypd_y = std::atan2(ay, ax);
    static double last_ypd_y = 0;
    ypd_y = last_ypd_y + angles::shortest_angular_distance(last_ypd_y, ypd_y);
    last_ypd_y = ypd_y;
    auto ypd_p = std::atan2(az, std::sqrt(ax * ax + ay * ay));
    auto ypd_d = std::sqrt(ax * ax + ay * ay + az * az);
    z[idx::YPD_Y] = ypd_y;
    z[idx::YPD_P] = ypd_p;
    z[idx::YPD_D] = ypd_d;
    auto rpy = utils::matrix2euler(a.pose.linear(), utils::EulerOrder::XYZ);
    double yaw = rpy[2];
    z[idx::ROT_YAW] = last_rot_yaw + angles::shortest_angular_distance(last_rot_yaw, yaw);
    last_rot_yaw = z[idx::ROT_YAW];
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
    const double t = dt;
    const double q_x_x = pow(t, 4) / 4 * q_xyz.x(), q_x_vx = pow(t, 3) / 2 * q_xyz.x(),
                 q_vx_vx = pow(t, 2) * q_xyz.x();
    const double q_y_y = pow(t, 4) / 4 * q_xyz.y(), q_y_vy = pow(t, 3) / 2 * q_xyz.y(),
                 q_vy_vy = pow(t, 2) * q_xyz.y();
    const double q_z_z = pow(t, 4) / 4 * q_xyz.z(), q_z_vz = pow(t, 3) / 2 * q_xyz.z(),
                 q_vz_vz = pow(t, 2) * q_xyz.z();
    const double q_yaw_yaw = pow(t, 4) / 4 * q_yaw, q_yaw_vyaw = pow(t, 3) / 2 * q_yaw,
                 q_vyaw_vyaw = pow(t, 2) * q_yaw;
    const double q_r = cfg.q_r;
    const double q_whole_car_roll_pitch = cfg.q_whole_car_roll_pitch;
    // clang-format off
            //      xc      v_xc    yc      v_yc    zc      v_zc    yaw         v_yaw       r       l   h   w_r w_p
            q <<    q_x_x,  q_x_vx, 0,      0,      0,      0,      0,          0,          0,      0,  0,  0,  0,
                    q_x_vx, q_vx_vx,0,      0,      0,      0,      0,          0,          0,      0,  0,  0,  0,
                    0,      0,      q_y_y,  q_y_vy, 0,      0,      0,          0,          0,      0,  0,  0,  0,
                    0,      0,      q_y_vy, q_vy_vy,0,      0,      0,          0,          0,      0,  0,  0,  0,
                    0,      0,      0,      0,      q_z_z,  q_z_vz, 0,          0,          0,      0,  0,  0,  0,
                    0,      0,      0,      0,      q_z_vz, q_vz_vz,0,          0,          0,      0,  0,  0,  0,
                    0,      0,      0,      0,      0,      0,      q_yaw_yaw,  q_yaw_vyaw, 0,      0,  0,  0,  0,
                    0,      0,      0,      0,      0,      0,      q_yaw_vyaw, q_vyaw_vyaw,0,      0,  0,  0,  0,
                    0,      0,      0,      0,      0,      0,      0,          0,          q_r,    0,  0,  0,  0,
                    0,      0,      0,      0,      0,      0,      0,          0,          0,      q_l,0,  0,  0,
                    0,      0,      0,      0,      0,      0,      0,          0,          0,      0,  q_h,0,  0,
                    0,      0,      0,      0,      0,      0,      0,          0,          0,      0, 0,   q_whole_car_roll_pitch,0,
                    0,      0,      0,      0,      0,      0,      0,          0,          0,      0, 0,   0,  q_whole_car_roll_pitch;

    // clang-format on
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
    this_id = GOBAL_ID++;
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

    auto add_obs = [&](Armor& armor, int id, bool is_left) {
        auto ctx = uvmeasure_ctx;
        ctx.id = id;
        ctx.is_left = is_left;
        ctx.camera_cv_in_odom = camera_cv_in_odom;

        UVMeasure measure { .ctx = ctx };

        obs.push_back(
            esekf.value().make_obs(get_uvmeasurement(armor, is_left), measure, u_r, cal_residual)
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
        if (armor.color_classifier) {
            const auto& colors = armor.color_classifier->light_colors;

            // if (colors[Armor::ColorClassifierCtx::LEFT] != ArmorColor::NONE) {
            //     add_obs(armor, id, true);
            // }

            // if (colors[Armor::ColorClassifierCtx::RIGHT] != ArmorColor::NONE) {
            //     add_obs(armor, id, false);
            // }
            add_obs(armor, id, true);
            add_obs(armor, id, false);

        } else {
            add_obs(armor, id, true);
            add_obs(armor, id, false);
        }
        ++updated;
        ++update_count;
    }
    for (const auto& [id, is_left, light]: matched_lights) {
        if (used_id[id]) {
            continue;
        }
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

    target_state.x = esekf.value().update_multi(obs);
    target_state.timestamp = timestamp;
    last_update = timestamp;
    this_id = GOBAL_ID++;
    return updated;
}
std::vector<std::pair<int, Armor>> ArmorTarget::match_armor(
    std::vector<Armor>& armors,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) const noexcept {
    std::vector<std::pair<int, Armor>> result;
    const int n_obs = static_cast<int>(armors.size());
    const int armors_num = armor_num();
    bool all_init =
        (outpost_has_all_and_has_set_ids.has_value() ? outpost_has_all_and_has_set_ids.value().first
                                                     : jumped);
    const double GATE = (all_init ? cfg.match_gate_at_1m : cfg.match_gate_not_all_init_at_1m);

    const double max_cost = 1e9;
    std::vector<std::vector<double>> cost(n_obs, std::vector<double>(armors_num, max_cost + 1));

    std::vector<YPDVecZ> meas_list(n_obs);
    for (int j = 0; j < n_obs; ++j) {
        armor_pnp(armors[j], camera_info, camera_cv_in_odom);
        meas_list[j] = get_ypdmeasurement(armors[j]);
    }

    for (int j = 0; j < n_obs; ++j) {
        if (armors[j].number == ArmorClass::OUTPOST) {
            auto rpy = utils::matrix2euler(armors[j].pose.linear(), utils::EulerOrder::XYZ);
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
            measure.h(target_state.x, z_pred);

            YPDVecZ nu = meas_list[j] - z_pred;
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
    std::vector<bool> used_id(armors_num, false);

    while (true) {
        double best = max_cost;
        int best_j = -1;
        int best_id = -1;

        for (int j = 0; j < n_obs; ++j) {
            if (used_obs[j])
                continue;
            for (int id = 0; id < armors_num; ++id) {
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
        result.push_back(std::make_pair(best_id, armors[best_j]));
    }
    return result;
}
std::vector<std::tuple<int, bool, Light>> ArmorTarget::match_light(
    std::vector<Light>& lights,
    std::vector<std::pair<int, Armor>>& matched_armors,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) const noexcept {
    std::vector<std::tuple<int, bool, Light>> result;

    if (target_number == ArmorClass::OUTPOST || target_number == ArmorClass::BASE
        || matched_armors.size() != 1)
    {
        return result;
    }

    const int armors_num = armor_num();
    const int armor_id = matched_armors.front().first;
    constexpr double max_cost = 1e9;

    auto predict_light = [&](int id, bool is_left) -> std::pair<cv::Point2f, cv::Point2f> {
        UVMeasure::Ctx ctx { .armor_num = armors_num,
                             .id = id,
                             .camera_cv_in_odom = camera_cv_in_odom,
                             .camera_info = camera_info,
                             .armor_number = target_number,
                             .is_left = is_left };
        UVMeasure measure { .ctx = ctx };
        UVVecZ z;
        measure.h(target_state.x, z);

        return {
            cv::Point2f(z[std::to_underlying(idx::TOP_X)], z[std::to_underlying(idx::TOP_Y)]),
            cv::Point2f(z[std::to_underlying(idx::BOTTOM_X)], z[std::to_underlying(idx::BOTTOM_Y)])
        };
    };

    std::array<std::pair<cv::Point2f, cv::Point2f>, 2> visible_lights {
        predict_light((armor_id + 3) % 4, false), // 左可见
        predict_light((armor_id + 1) % 4, true), // 右可见
    };

    const int n_obs = static_cast<int>(lights.size());
    std::vector<std::array<double, visible_lights.size()>> cost(
        n_obs,
        { max_cost + 1, max_cost + 1 }
    );

    auto calc_cost = [&](const Light& light,
                         const std::pair<cv::Point2f, cv::Point2f>& pred) -> double {
        if (matched_armors[0].second.key_points.bounding_box().contains(light.top)
            || matched_armors[0].second.key_points.bounding_box().contains(light.bottom)
            || matched_armors[0].second.key_points.bounding_box().contains(light.center))
        {
            return max_cost + 1;
        }
        double pred_len = cv::norm(pred.first - pred.second);
        double len_err = std::abs(light.length - pred_len);
        if (len_err > pred_len * cfg.light_match_length_ratio_gate) {
            // AWAKENING_WARN("match out of gate: light length err: {}", len_err);
            return max_cost + 1;
        }

        double pred_angle = std::atan2(pred.first.x - pred.second.x, pred.first.y - pred.second.y);
        double light_angle = std::atan2(light.top.x - light.bottom.x, light.top.y - light.bottom.y);
        double angle_err = std::abs(angles::normalize_angle(light_angle - pred_angle));
        if (angle_err > cfg.light_match_angle_gate) {
            // AWAKENING_WARN("match out of gate: light angle err: {}", angle_err);
            return max_cost + 1;
        }

        double pos_err = cv::norm(light.top - pred.first) + cv::norm(light.bottom - pred.second);
        if (pos_err > pred_len * cfg.light_match_pos_gate_by_length_ratio) {
            // AWAKENING_WARN("match out of gate: light position err: {}", pos_err);
            return max_cost + 1;
        }

        return pos_err;
    };

    for (int j = 0; j < n_obs; ++j) {
        for (int i = 0; i < visible_lights.size(); ++i) {
            cost[j][i] = calc_cost(lights[j], visible_lights[i]);
        }
    }

    std::vector<bool> used_obs(n_obs, false);
    const std::array<std::pair<int, bool>, visible_lights.size()> mapping {
        std::pair { (armor_id + 3) % 4, false }, // 左可见
        std::pair { (armor_id + 1) % 4, true }, // 右可见
    };

    std::array<bool, mapping.size()> used_id { false, false }; // 只匹配左可见/右可见

    while (true) {
        double best = max_cost;
        int best_j = -1, best_id = -1;

        for (int j = 0; j < n_obs; ++j) {
            if (used_obs[j])
                continue;
            for (int id = 0; id < 2; ++id) {
                if (used_id[id])
                    continue;
                if (cost[j][id] < best) {
                    best = cost[j][id];
                    best_j = j;
                    best_id = id;
                }
            }
        }

        if (best_j < 0 || best_id < 0)
            break;

        used_obs[best_j] = true;
        used_id[best_id] = true;

        auto [matched_id, is_left] = mapping[best_id];
        result.emplace_back(matched_id, is_left, lights[best_j]);
    }

    return result;
}
[[nodiscard]] cv::Rect ArmorTarget::expanded_one_one(
    const TimePoint& timestamp,
    const ISO3& camera_cv_in_odom,
    const CameraInfo& camera_info,
    const cv::Size& image_size
) const noexcept {
    const double dt = std::chrono::duration<double>(timestamp - last_update).count();

    if (!is_inited || dt > cfg.lost_time_thres) {
        return cv::Rect(0, 0, image_size.width, image_size.height);
    }

    float car_box_half = std::max(target_state.r(), target_state.r() + target_state.l()) + 0.15f;
    if (target_number == ArmorClass::OUTPOST) {
        car_box_half = target_state.r() + 0.15f;
    }
    static std::vector<cv::Point3f> CAR_BOX;
    CAR_BOX = { { 0, car_box_half, -car_box_half },
                { 0, -car_box_half, -car_box_half },
                { 0, -car_box_half, car_box_half },
                { 0, car_box_half, car_box_half } };

    auto target_pos_in_odom = target_state.pos();
    if (target_number == ArmorClass::OUTPOST) {
        target_pos_in_odom.z() += (target_state.outpost01DZ() + target_state.outpost02DZ()) / 2.0;
    }
    auto target_pos_in_camera_cv = camera_cv_in_odom.inverse() * target_pos_in_odom;

    if (target_pos_in_camera_cv.z() <= 0.2) {
        return cv::Rect(0, 0, image_size.width, image_size.height);
    }

    const cv::Mat tvec =
        (cv::Mat_<double>(3, 1) << target_pos_in_camera_cv.x(),
         target_pos_in_camera_cv.y(),
         target_pos_in_camera_cv.z());

    auto target_R_in_odom = utils::euler2matrix(
        Vec3(std::atan2(target_pos_in_odom.y(), target_pos_in_odom.x()), 0, 0),
        utils::EulerOrder::ZYX
    );

    auto target_R_in_camera_cv = camera_cv_in_odom.inverse() * target_R_in_odom;

    const cv::Mat rot_mat =
        (cv::Mat_<double>(3, 3) << target_R_in_camera_cv(0, 0),
         target_R_in_camera_cv(0, 1),
         target_R_in_camera_cv(0, 2),
         target_R_in_camera_cv(1, 0),
         target_R_in_camera_cv(1, 1),
         target_R_in_camera_cv(1, 2),
         target_R_in_camera_cv(2, 0),
         target_R_in_camera_cv(2, 1),
         target_R_in_camera_cv(2, 2));

    cv::Mat rvec;
    cv::Rodrigues(rot_mat, rvec);

    std::vector<cv::Point2f> pts_2d;
    cv::projectPoints(
        CAR_BOX,
        rvec,
        tvec,
        camera_info.camera_matrix,
        camera_info.distortion_coefficients,
        pts_2d
    );

    const cv::Rect rect = cv::boundingRect(pts_2d);

    const cv::Rect img_rect(0, 0, image_size.width, image_size.height);

    if ((rect & img_rect).area() <= 0) {
        return cv::Rect(0, 0, image_size.width, image_size.height);
    }

    const double lost_dt = cfg.lost_time_thres;

    double alpha = std::clamp(dt / lost_dt, 0.0, 1.0);

    double x1 = rect.x;
    double y1 = rect.y;
    double x2 = rect.x + rect.width;
    double y2 = rect.y + rect.height;

    const double img_x1 = 0.0;
    const double img_y1 = 0.0;
    const double img_x2 = image_size.width;
    const double img_y2 = image_size.height;

    x1 = std::clamp(x1, img_x1, img_x2);
    x2 = std::clamp(x2, img_x1, img_x2);
    y1 = std::clamp(y1, img_y1, img_y2);
    y2 = std::clamp(y2, img_y1, img_y2);

    x1 = x1 + (img_x1 - x1) * alpha;
    y1 = y1 + (img_y1 - y1) * alpha;
    x2 = x2 + (img_x2 - x2) * alpha;
    y2 = y2 + (img_y2 - y2) * alpha;

    cv::Rect expanded_rect(
        static_cast<int>(x1),
        static_cast<int>(y1),
        static_cast<int>(x2 - x1),
        static_cast<int>(y2 - y1)
    );

    int cx = expanded_rect.x + expanded_rect.width / 2;
    int cy = expanded_rect.y + expanded_rect.height / 2;

    int side = std::max(expanded_rect.width, expanded_rect.height);

    cv::Rect square(cx - side / 2, cy - side / 2, side, side);

    square &= img_rect;

    return square;
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
            measure.h(target_state.x, z_pred);
            pts.push_back(cv::Point2f(z_pred[idx::TOP_X], z_pred[idx::TOP_Y]));
            pts.push_back(cv::Point2f(z_pred[idx::BOTTOM_X], z_pred[idx::BOTTOM_Y]));
        }
        {
            measure.ctx.is_left = false;
            measure.h(target_state.x, z_pred);
            pts.push_back(cv::Point2f(z_pred[idx::TOP_X], z_pred[idx::TOP_Y]));
            pts.push_back(cv::Point2f(z_pred[idx::BOTTOM_X], z_pred[idx::BOTTOM_Y]));
        }
    }
    cv::Rect rect = cv::boundingRect(pts);
    const cv::Rect img_rect(0, 0, image_size.width, image_size.height);
    if ((rect & img_rect).area() <= 0) {
        return cv::Rect(0, 0, image_size.width, image_size.height);
    }
    return rect;
}
} // namespace awakening::auto_aim