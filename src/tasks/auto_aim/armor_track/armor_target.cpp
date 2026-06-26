#include "armor_target.hpp"
#include "angles.h"
#include "tasks/auto_aim/armor_track/motion_model.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
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
#include <opencv2/imgproc.hpp>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
namespace awakening::auto_aim {
using namespace armor_point_motion_model;

namespace {
    UVVecZ get_uv_measurement(
        const cv::Point2f& top,
        const cv::Point2f& bottom,
        const CameraInfo& camera_info
    ) noexcept {
        cv::Point2f measurement_top = top;
        cv::Point2f measurement_bottom = bottom;
        if (MEASURE_NORMALIZED) {
            measurement_top = utils::undistort_point(
                camera_info.camera_matrix,
                camera_info.distortion_coefficients,
                top
            );
            measurement_bottom = utils::undistort_point(
                camera_info.camera_matrix,
                camera_info.distortion_coefficients,
                bottom
            );
        }

        const Eigen::Vector2d top_eigen(measurement_top.x, measurement_top.y);
        const Eigen::Vector2d bottom_eigen(measurement_bottom.x, measurement_bottom.y);
        UVVecZ observation;
        UVMeasure::points_to_observation(top_eigen, bottom_eigen, observation.data());
        return observation;
    }
} // namespace

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
        j_target_state["r1"] = Web::val(target_state.r1());
        j_target_state["r2"] = Web::val(target_state.r2());
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
    target_number = a.number;
    uvmeasure_ctx = {
        .armor_num = armor_num_by_armor_class(target_number),
        .id = 0,
        .camera_cv_in_odom = camera_cv_in_odom,
        .camera_info = camera_info.clone(),
        .armor_number = target_number,
    };

    double r_pre;
    Eigen::DiagonalMatrix<double, X_N> p0;
    p0.diagonal().setZero();
    p0.diagonal()[idx::CX] = p0.diagonal()[idx::CY] = p0.diagonal()[idx::CZ] = 1;
    p0.diagonal()[idx::VCX] = p0.diagonal()[idx::VCY] = p0.diagonal()[idx::VCZ] = 10;
    p0.diagonal()[idx::C_ROT_Z] = p0.diagonal()[idx::C_ROT_Y] = p0.diagonal()[idx::C_ROT_X] = 1;
    p0.diagonal()[idx::LOG_R1] = p0.diagonal()[idx::LOG_R2] = p0.diagonal()[idx::H] = 1;
    if (target_number == ArmorClass::OUTPOST) {
        p0.diagonal()[idx::OUTPOST01DZ] = p0.diagonal()[idx::OUTPOST02DZ] = 1;
    }
    p0.diagonal()[idx::VYAW] = 100;
    if (a.number == ArmorClass::OUTPOST) {
        r_pre = 0.2765;
    } else if (a.number == ArmorClass::BASE) {
        r_pre = 0.3205;
    } else {
        r_pre = 0.26;
    }
    const auto u_q = [] { return Eigen::Matrix<double, X_N, X_N>::Zero(); };
    const auto inject = [](const auto& delta, auto& nominal) { inject_state(delta, nominal); };
    const auto box_minus = [](const auto& nominal, const auto& value, auto& delta) {
        box_minus_state(nominal, value, delta);
    };
    esekf = RobotStateESEKF(
        Predict { .dt = 0.005, .armor_number = target_number },
        u_q,
        inject,
        box_minus,
        p0
    );

    esekf->set_iteration_num(cfg.esekf_iter_num);
    auto armor_in_odom = a.pose;
    auto armor_in_car = ISO3::Identity();
    const double r = r_pre;
    armor_in_car.translation() << -r, 0, 0;
    const double armor_pitch = (target_number == auto_aim::ArmorClass::OUTPOST)
        ? (-auto_aim::FIFTTEN_DEGREE_RAD)
        : (auto_aim::FIFTTEN_DEGREE_RAD);
    armor_in_car.linear() = utils::rpy2matrix(Vec3(0, armor_pitch, 0));
    auto car_in_odom = armor_in_odom * armor_in_car.inverse();
    const Vec3 car_rot = utils::so3_log(car_in_odom.linear().eval());
    target_state.x = Eigen::VectorXd::Zero(X_N);
    target_state.set_pos(car_in_odom.translation());
    target_state.x[idx::LOG_R1] = target_state.x[idx::LOG_R2] = std::log(r);
    if (target_number == ArmorClass::OUTPOST) {
        target_state.x[idx::OUTPOST01DZ] = target_state.x[idx::OUTPOST02DZ] = 0;
    }
    target_state.x[idx::C_ROT_Z] = car_rot.z();
    target_state.x[idx::C_ROT_Y] = car_rot.y();
    target_state.x[idx::C_ROT_X] = car_rot.x();
    target_state.timestamp = timestamp;
    target_state.frame_id = frame_id;
    esekf->set_state(target_state.x);
    last_update = timestamp;
    is_inited = true;
    jumped = false;
    last_match_id = -1;
    if (target_number == ArmorClass::OUTPOST) {
        outpost_has_all_and_has_set_ids =
            std::make_pair(false, std::vector<bool>(armor_num(), false));
        outpost_has_all_and_has_set_ids->second[0] = true;
    } else {
        outpost_has_all_and_has_set_ids = std::nullopt;
    }
    track_state.reset();
    this_id = GLOBAL_ID++;
    voter.reset(timestamp);
}

bool ArmorTarget::armor_pnp(
    Armor& a,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) noexcept {
    auto key_points = a.key_points.landmarks();
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    if (!cv::solvePnPGeneric(
            getArmorKeyPoints3D<cv::Point3f>(a.number),
            key_points,
            camera_info.camera_matrix,
            camera_info.distortion_coefficients,
            rvecs,
            tvecs,
            false,
            cv::SOLVEPNP_IPPE,
            cv::noArray(),
            cv::noArray()
        ))
    {
        return false;
    }

    bool has_valid = false;
    for (size_t i = 0; i < rvecs.size(); ++i) {
        cv::Mat R_cv;
        cv::Rodrigues(rvecs[i], R_cv);
        Mat3 R_eigen;
        cv::cv2eigen(R_cv, R_eigen);
        Vec3 axis_x = R_eigen.col(0);
        Vec3 t_eigen;
        cv::cv2eigen(tvecs[i], t_eigen);
        Vec3 front_normal = -axis_x;
        if (front_normal.dot(-t_eigen) > 0) {
            a.pose.translation() = t_eigen;
            a.pose.linear() = R_eigen;
            has_valid = true;
            break;
        }
    }

    auto armor_in_odom = camera_cv_in_odom * a.pose;
    a.pose = armor_in_odom;
    if (a.number == auto_aim::ArmorClass::OUTPOST || !USE_WROT) {
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

    return has_valid;
}
Eigen::Matrix<double, UVZ_N, UVZ_N>
ArmorTarget::uvmeasurement_covariance(const Eigen::Matrix<double, UVZ_N, 1>& z) const noexcept {
    Eigen::Matrix<double, UVZ_N, UVZ_N> r;
    r.setZero();
    auto length = z[idx::UV_LENGTH];
    if(MEASURE_NORMALIZED){
        length *= (uvmeasure_ctx.camera_info.camera_fx() + uvmeasure_ctx.camera_info.camera_fy()) / 2.0;
    }
    const double sigma_px = cfg.r_sigma_px_by_length_ratio*length;
    double sigma_x = sigma_px;
    double sigma_y = sigma_px;
    if (MEASURE_NORMALIZED) {
        sigma_x /= uvmeasure_ctx.camera_info.camera_fx();
        sigma_y /= uvmeasure_ctx.camera_info.camera_fy();
    }
    double sigma_length = cfg.r_sigma_length_by_length_ratio*length;
    if (MEASURE_NORMALIZED) {
        sigma_length /=
            ((uvmeasure_ctx.camera_info.camera_fx() + uvmeasure_ctx.camera_info.camera_fy()) / 2.0);
    }
    double sigma_angle = cfg.r_sigma_angle;
    sigma_angle *= std::cos(z(idx::UV_ANGLE));
    r(idx::UV_ANGLE, idx::UV_ANGLE) = sigma_angle * sigma_angle / 2.0;
    r(idx::UV_CENTER_X, idx::UV_CENTER_X) = sigma_x * sigma_x / 2.0;
    r(idx::UV_CENTER_Y, idx::UV_CENTER_Y) = sigma_y * sigma_y / 2.0;
    r(idx::UV_LENGTH, idx::UV_LENGTH) = sigma_length * sigma_length / 2.0;
    return r;
}

Eigen::Matrix<double, X_N, X_N> ArmorTarget::process_noise(double dt) const noexcept {
    Eigen::Matrix<double, X_N, X_N> q;
    Vec3 q_xyz_body;
    double q_yaw;
    if (target_number == ArmorClass::OUTPOST) {
        q_xyz_body = cfg.qxyz_output; // 前哨站车体系加速度方差
        q_yaw = cfg.qyaw_output; // 前哨站角加速度方差
    } else {
        q_xyz_body = cfg.qxyz_common; // 车体系加速度方差
        q_yaw = cfg.qyaw_common; // 角加速度方差
    }
    q.setZero();
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;
    const Mat3 car_in_odom_R = whole_car_pose(target_state.x.data(), target_number).linear();
    const Mat3 Q_acc_body = q_xyz_body.asDiagonal();
    const Mat3 Q_acc_odom = car_in_odom_R * Q_acc_body * car_in_odom_R.transpose();
    constexpr std::array<int, 3> pos_idx { idx::CX, idx::CY, idx::CZ };
    constexpr std::array<int, 3> vel_idx { idx::VCX, idx::VCY, idx::VCZ };
    constexpr std::array<int, 3> rot_idx { idx::C_ROT_X, idx::C_ROT_Y, idx::C_ROT_Z };
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            q(pos_idx[i], pos_idx[j]) = 0.25 * dt4 * Q_acc_odom(i, j);
            q(pos_idx[i], vel_idx[j]) = 0.5 * dt3 * Q_acc_odom(i, j);
            q(vel_idx[i], pos_idx[j]) = 0.5 * dt3 * Q_acc_odom(i, j);
            q(vel_idx[i], vel_idx[j]) = dt2 * Q_acc_odom(i, j);
        }
    }
    q(idx::VYAW, idx::VYAW) += dt2 * q_yaw;
    q(idx::C_ROT_Z, idx::VYAW) += 0.5 * dt3 * q_yaw;
    q(idx::VYAW, idx::C_ROT_Z) += 0.5 * dt3 * q_yaw;
    q(idx::C_ROT_Z, idx::C_ROT_Z) += 0.25 * dt4 * q_yaw;
    const Mat3 Q_wpr_body = (Vec3(cfg.q_wpr, cfg.q_wpr, 0)).asDiagonal();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            q(rot_idx[i], rot_idx[j]) += dt * Q_wpr_body(i, j);
        }
    }
    q(idx::LOG_R1, idx::LOG_R1) = cfg.q_r / (target_state.r1() * target_state.r1());
    if (target_number == ArmorClass::OUTPOST) {
        q(idx::OUTPOST01DZ, idx::OUTPOST01DZ) = cfg.q_outpost_dz;
        q(idx::OUTPOST02DZ, idx::OUTPOST02DZ) = cfg.q_outpost_dz;
    } else {
        q(idx::LOG_R2, idx::LOG_R2) = cfg.q_r / (target_state.r2() * target_state.r2());
        q(idx::H, idx::H) = cfg.q_h;
    }
    return q;
}

void ArmorTarget::predict_ekf(const TimePoint& timestamp) {
    if (!esekf) {
        throw std::runtime_error("ESEKF is not initialized");
    }
    auto dt = std::chrono::duration<double>(timestamp - target_state.timestamp).count();
    esekf->set_predict_func(Predict { .dt = dt, .armor_number = target_number, .voter = voter });
    esekf->set_update_Q([&]() { return process_noise(dt); });
    target_state.x = esekf->predict();
    target_state.timestamp = timestamp;
    this_id = GLOBAL_ID++;
}

int ArmorTarget::update(
    std::vector<std::pair<int, Armor>>& matched_armors,
    std::vector<std::tuple<int, bool, Light>>& matched_lights,
    const TimePoint& timestamp,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) {
    if (matched_armors.empty() && matched_lights.empty()) {
        return 0;
    }
    uvmeasure_ctx.camera_cv_in_odom = camera_cv_in_odom;

    std::vector<std::shared_ptr<RobotStateESEKF::ObsBase>> obs;

    const auto cal_residual = [](const Eigen::Matrix<double, UVZ_N, 1>& z_pred,
                                 const Eigen::Matrix<double, UVZ_N, 1>& z) {
        return UVMeasure::residual(z_pred, z);
    };
    const int armors_num = armor_num();
    auto add_uv_obs = [&](const cv::Point2f& top, const cv::Point2f& bottom, int id, bool is_left) {
        auto ctx = uvmeasure_ctx;
        ctx.id = id;
        ctx.is_left = is_left;
        ctx.normalized = MEASURE_NORMALIZED;
        const auto observation = get_uv_measurement(top, bottom, camera_info);
        const auto u_r = [this](const Eigen::Matrix<double, UVZ_N, 1>& z) {
            return uvmeasurement_covariance(z);
        };
        UVMeasure measure { .ctx = ctx };
        obs.push_back(esekf->make_obs(observation, measure, u_r, cal_residual));
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

    std::vector<bool> used_id(armor_num(), false);
    {
        for (auto& [id, armor]: matched_armors) {
            jumped |= (id != 0);
            update_outpost_state(id);
            last_match_id = id;
            used_id[id] = true;
            const auto key_points = armor.key_points.landmarks();
            add_uv_obs(
                key_points[ArmorKeyPointsIndex::LEFT_TOP],
                key_points[ArmorKeyPointsIndex::LEFT_BOTTOM],
                id,
                true
            );
            add_uv_obs(
                key_points[ArmorKeyPointsIndex::RIGHT_TOP],
                key_points[ArmorKeyPointsIndex::RIGHT_BOTTOM],
                id,
                false
            );
            // add_ypd_obs(armor, id);
        }
    }

    for (const auto& [id, is_left, light]: matched_lights) {
        if (used_id[id]) {
            continue;
        }
        used_id[id] = true;
        add_uv_obs(light.top, light.bottom, id, is_left);
    }
    if (!obs.empty()) {
        target_state.x = esekf->update_multi(obs);
        target_state.timestamp = timestamp;
        last_update = timestamp;
        this_id = GLOBAL_ID++;
        voter.update(
            utils::matrix2rpy(
                utils::so3_exp(Vec3(
                    target_state.x[idx::C_ROT_X],
                    target_state.x[idx::C_ROT_Y],
                    target_state.x[idx::C_ROT_Z]
                )),
                utils::RPYOrder::ZYX
            )[2],
            timestamp
        );
    }
    update_count += obs.size();
    return obs.size();
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
    auto pred_state = target_state;
    pred_state.predict(timestamp, target_number);
    const int armors_num = armor_num();
    std::vector<int> maybe_visible;
    std::vector<std::pair<double, int>> angle_dis_in_camera_cv;
    for (int i = 0; i < armors_num; i++) {
        auto pose_in_odom = armor_pose(pred_state.x.data(), i, armors_num, target_number);
        auto pose_in_camera_cv = camera_cv_in_odom.inverse() * pose_in_odom;
        Vec3 axis_x = pose_in_camera_cv.linear().col(0);
        Vec3 front_normal = -axis_x;
        auto normal = pose_in_camera_cv.linear().col(0);
        angle_dis_in_camera_cv.emplace_back(front_normal.dot(-pose_in_camera_cv.translation()), i);
    }
    std::sort(
        angle_dis_in_camera_cv.begin(),
        angle_dis_in_camera_cv.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    const auto visible_count = std::min<std::size_t>(3, angle_dis_in_camera_cv.size());
    for (std::size_t i = 0; i < visible_count; ++i) {
        maybe_visible.push_back(angle_dis_in_camera_cv[i].second);
    }
    const bool all_init =
        outpost_has_all_and_has_set_ids ? outpost_has_all_and_has_set_ids->first : jumped;
    std::vector<std::vector<double>> cost(
        n_obs,
        std::vector<double>(maybe_visible.size(), MAX_COST + 1)
    );
    std::vector<std::pair<UVVecZ, UVVecZ>> meas_list(n_obs);
    for (int i = 0; i < n_obs; ++i) {
        auto key_points = armors[i].key_points.landmarks();
        meas_list[i].first = get_uv_measurement(
            key_points[ArmorKeyPointsIndex::LEFT_TOP],
            key_points[ArmorKeyPointsIndex::LEFT_BOTTOM],
            camera_info
        );
        meas_list[i].second = get_uv_measurement(
            key_points[ArmorKeyPointsIndex::RIGHT_TOP],
            key_points[ArmorKeyPointsIndex::RIGHT_BOTTOM],
            camera_info
        );
    }
    for (int j = 0; j < n_obs; ++j) {
        bool in_gate = false;
        double min_d2 = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < maybe_visible.size(); ++i) {
            const int id = maybe_visible[i];
            auto ctx = uvmeasure_ctx;
            ctx.id = id;
            ctx.camera_cv_in_odom = camera_cv_in_odom;
            ctx.is_left = true;
            UVMeasure measure { .ctx = ctx };
            UVVecZ z_pred_l;
            measure.h(pred_state.x, z_pred_l);
            auto nu_l = UVMeasure::residual(z_pred_l, meas_list[j].first);
            ctx.is_left = false;
            UVMeasure measure_right { .ctx = ctx };
            UVVecZ z_pred_r;
            measure_right.h(pred_state.x, z_pred_r);
            auto nu_r = UVMeasure::residual(z_pred_r, meas_list[j].second);
            auto R_l = uvmeasurement_covariance(z_pred_l);
            auto R_r = uvmeasurement_covariance(z_pred_r);
            Eigen::VectorXd nu(nu_l.size() + nu_r.size());
            nu << nu_l, nu_r;
            Eigen::MatrixXd R(R_l.rows() + R_r.rows(), R_l.cols() + R_r.cols());
            R.setZero();
            R.topLeftCorner(R_l.rows(), R_l.cols()) = R_l;
            R.bottomRightCorner(R_r.rows(), R_r.cols()) = R_r;
            double d2 = nu.transpose() * R.ldlt().solve(nu);

            if (std::isfinite(d2)
                && d2 < (!all_init ? cfg.armor_match_gate_not_all_init : cfg.armor_match_gate))
            {
                cost[j][i] = d2;
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
    for (auto [obs, i]: dta_utils::greedy_match(cost, n_obs, maybe_visible.size(), MAX_COST)) {
        result.emplace_back(maybe_visible[i], armors[obs]);
    }
    return result;
}
std::pair<cv::Point2f, cv::Point2f> ArmorTarget::predict_light(
    int armor_id,
    bool is_left,
    const armor_point_motion_model::State& state,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) const noexcept {
    UVMeasure::Ctx ctx { .armor_num = armor_num(),
                         .id = armor_id,
                         .camera_cv_in_odom = camera_cv_in_odom,
                         .camera_info = camera_info,
                         .armor_number = target_number,
                         .is_left = is_left };
    UVMeasure measure { .ctx = ctx };
    return measure.projected_points(state.x);
}
std::vector<std::tuple<int, bool, Light>> ArmorTarget::match_light(
    std::vector<Light>& lights,
    const std::vector<std::pair<int, Armor>>& matched_armors,
    const TimePoint& timestamp,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) const noexcept {
    if (target_number == ArmorClass::BASE || matched_armors.empty()) {
        return {};
    }
    // return {};
    auto pred_state = target_state;
    pred_state.predict(timestamp, target_number);
    const int armors_num = armor_num();
    std::vector<bool> matched(armors_num, false);
    for (auto& [id, _]: matched_armors) {
        matched[id] = true;
    }
    std::vector<std::pair<double, int>> angle_dis_in_camera_cv;
    for (int i = 0; i < armors_num; i++) {
        auto pose_in_odom = armor_pose(pred_state.x.data(), i, armors_num, target_number);
        auto pose_in_camera_cv = camera_cv_in_odom.inverse() * pose_in_odom;
        Vec3 axis_x = pose_in_camera_cv.linear().col(0);
        Vec3 front_normal = -axis_x;
        auto normal = pose_in_camera_cv.linear().col(0);
        angle_dis_in_camera_cv.emplace_back(front_normal.dot(-pose_in_camera_cv.translation()), i);
    }
    std::sort(
        angle_dis_in_camera_cv.begin(),
        angle_dis_in_camera_cv.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    if (angle_dis_in_camera_cv.empty()) {
        return {};
    }
    std::vector<std::tuple<int, bool, std::pair<cv::Point2f, cv::Point2f>>> visible_lights;
    auto maybe_visible = [&](int armor_id, bool is_left) {
        visible_lights.emplace_back(
            armor_id,
            is_left,
            predict_light(armor_id, is_left, pred_state, camera_info, camera_cv_in_odom)
        );
    };
    const auto closest = std::min_element(
        angle_dis_in_camera_cv.begin(),
        angle_dis_in_camera_cv.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    maybe_visible((closest->second + armors_num - 1) % armors_num, false);
    maybe_visible((closest->second + 1) % armors_num, true);
    maybe_visible(closest->second, false);
    maybe_visible(closest->second, true);
    return match_light(lights, matched_armors, visible_lights);
}
std::vector<std::tuple<int, bool, Light>> ArmorTarget::match_light(
    std::vector<Light>& lights,
    const std::vector<std::pair<int, Armor>>& matched_armors,
    const std::vector<std::tuple<int, bool, std::pair<cv::Point2f, cv::Point2f>>>& visible_lights
) const noexcept {
    std::vector<std::tuple<int, bool, Light>> result;
    if (visible_lights.empty()) {
        return result;
    }
    const int n_obs = static_cast<int>(lights.size());
    constexpr double MAX_COST = 1e9;
    std::vector<std::vector<double>> cost(
        n_obs,
        std::vector<double>(visible_lights.size(), MAX_COST + 1)
    );
    auto calc_cost =
        [&](const Light& light,
            const std::tuple<int, bool, std::pair<cv::Point2f, cv::Point2f>>& visible_light
        ) -> double {
        const auto& pred = std::get<2>(visible_light);
        // for (const auto& [_, armor]: matched_armors) {
        //     if (!is_light_separate_from_armor(armor.key_points, light)) {
        //         return MAX_COST + 1;
        //     }
        // }

        const double pred_len = cv::norm(pred.first - pred.second);
        const double len_err = std::abs(light.length - pred_len);
        if (len_err > pred_len * cfg.light_match_length_ratio_gate) {
            return MAX_COST + 1;
        }

        const double pred_angle =
            std::atan2(pred.first.x - pred.second.x, pred.first.y - pred.second.y);
        const double light_angle =
            std::atan2(light.top.x - light.bottom.x, light.top.y - light.bottom.y);
        const double angle_err = std::abs(angles::normalize_angle(light_angle - pred_angle));
        if (angle_err > cfg.light_match_angle_gate) {
            return MAX_COST + 1;
        }

        const double pos_err =
            cv::norm(light.top - pred.first) + cv::norm(light.bottom - pred.second);
        if (pos_err > pred_len * cfg.light_match_pos_gate_by_length_ratio) {
            return MAX_COST + 1;
        }

        return pos_err;
    };

    for (int j = 0; j < n_obs; ++j) {
        for (std::size_t i = 0; i < visible_lights.size(); ++i) {
            cost[j][i] = calc_cost(lights[j], visible_lights[i]);
        }
    }

    for (auto [obs, id]: dta_utils::greedy_match(cost, n_obs, visible_lights.size(), MAX_COST)) {
        const auto& [armor_id, is_left, _] = visible_lights[id];
        lights[obs].laji = false;
        result.emplace_back(armor_id, is_left, lights[obs]);
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
    constexpr double expand_ratio = 1.4;
    rect = utils::expand_and_clip_rect(rect, expand_ratio, image_size);
    const cv::Rect img_rect(0, 0, image_size.width, image_size.height);

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
    const double dt_clamped = std::clamp(dt, 0.0, lost_dt);

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
[[nodiscard]] cv::Rect ArmorTarget::expanded(
    const TimePoint& timestamp,
    const ISO3& camera_cv_in_odom,
    const CameraInfo& camera_info,
    const cv::Size& image_size
) const noexcept {
    auto pts = expanded_pts(timestamp, camera_cv_in_odom, camera_info);
    cv::Rect rect = cv::boundingRect(pts);
    const cv::Rect img_rect = cv::Rect(0, 0, image_size.width, image_size.height);
    if ((rect & img_rect).area() <= 0) {
        return img_rect;
    }
    return rect;
}
[[nodiscard]] std::vector<cv::Point2f> ArmorTarget::expanded_pts(
    const TimePoint& timestamp,
    const ISO3& camera_cv_in_odom,
    const CameraInfo& camera_info
) const noexcept {
    std::vector<cv::Point2f> pts;
    auto tmp_target_state = target_state;
    tmp_target_state.predict(timestamp, target_number);
    const int armors_num = armor_num();
    pts.reserve(armors_num * 4);
    for (int id = 0; id < armors_num; ++id) {
        for (const bool is_left: { true, false }) {
            const auto light =
                predict_light(id, is_left, tmp_target_state, camera_info, camera_cv_in_odom);
            pts.push_back(light.first);
            pts.push_back(light.second);
        }
    }
    return pts;
}
} // namespace awakening::auto_aim
