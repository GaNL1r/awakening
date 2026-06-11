#include "armor_target.hpp"
#include "angles.h"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/web.hpp"
#include "utils/utils.hpp"
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
    auto rpy = utils::matrix2rpy(a.pose.linear());
    const double yaw = rpy[2];
    last_rot_yaw = yaw;
    target_state.x = Eigen::VectorXd::Zero(X_N);
    const double r = r_pre;
    const double xc = xa + r * cos(yaw);
    const double yc = ya + r * sin(yaw);
    const double zc = za;
    double l = 0.0;
    double h = 0.0;
    target_state.x << xc, 0, yc, 0, zc, 0, yaw, 0, r, l, h;
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
    auto rpy = utils::matrix2rpy(a.pose.linear());
    double yaw_raw = rpy[2];
    constexpr double SEARCH_RANGE = 140; // 随便穷举（
    auto yaw0 = angles::normalize_angle(yaw_raw - SEARCH_RANGE / 2 * CV_PI / 180.0);
    auto min_error = 1e10;
    auto best_rot = a.pose.linear();
    auto obj_points = getArmorKeyPoints3D<cv::Point3f>(a.number);
    const double armor_pitch = (a.number == auto_aim::ArmorClass::OUTPOST)
        ? -auto_aim::FIFTTEN_DEGREE_RAD
        : auto_aim::FIFTTEN_DEGREE_RAD;
    for (int i = 0; i < SEARCH_RANGE; i++) {
        double yaw = angles::normalize_angle(yaw0 + i * CV_PI / 180.0);
        auto a_pose_in_odom = a.pose;
        auto search_rpy = Vec3(0, armor_pitch, yaw);
        a_pose_in_odom.linear() = utils::rpy2matrix(search_rpy);
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
        ); //中点为约束实验上比原来4点约束更准（

        if (error < min_error) {
            min_error = error;
            best_rot = a_pose_in_odom.linear();
        }
    }
    a.pose.linear() = best_rot;
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
    r.setZero(); //copy下sp_vision_25 这个参数不用在观测，差不多就行
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
    auto rpy = utils::matrix2rpy(a.pose.linear());
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

    q.setZero();
    utils::fill_constant_accel_noise(q, idx::CX, idx::VCX, q_xyz.x(), dt);
    utils::fill_constant_accel_noise(q, idx::CY, idx::VCY, q_xyz.y(), dt);
    utils::fill_constant_accel_noise(q, idx::CZ, idx::VCZ, q_xyz.z(), dt);
    utils::fill_constant_accel_noise(q, idx::YAW, idx::VYAW, q_yaw, dt);
    q(idx::R, idx::R) = cfg.q_r;
    q(idx::L, idx::L) = q_l;
    q(idx::H, idx::H) = q_h;
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
        add_obs(armor, id, true);
        add_obs(armor, id, false);
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
    this_id = GOBAL_ID++; //全局状态标记，下游控制对同一id的不重复构建轨迹
    return updated;
}
std::vector<std::pair<int, Armor>> ArmorTarget::match_armor(
    std::vector<Armor>& armors,
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

    for (int j = 0; j < n_obs; ++j) {
        if (armors[j].number == ArmorClass::OUTPOST) {
            auto rpy = utils::matrix2rpy(armors[j].pose.linear());
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
            nu[idx::YPD_Y] = angles::normalize_angle(nu[idx::YPD_Y]);
            nu[idx::ROT_YAW] = angles::normalize_angle(nu[idx::ROT_YAW]);
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
        double best = MAX_COST;
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
    constexpr double MAX_COST = 1e9;
    //可见灯条逻辑判断不优雅，不过这比较个稳定可观
    std::vector<std::tuple<int, bool, Light>> result;

    if (target_number == ArmorClass::BASE || matched_armors.size() != 1) {
        return result;
    }

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
        measure.h(target_state.x, z);

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

    std::vector<bool> used_obs(n_obs, false);
    std::array<bool, visible_mapping.size()> used_id { false, false }; // 只匹配左可见/右可见
    while (true) {
        double best = MAX_COST;
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

        auto [matched_id, is_left] = visible_mapping[best_id];
        result.emplace_back(matched_id, is_left, lights[best_j]);
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
