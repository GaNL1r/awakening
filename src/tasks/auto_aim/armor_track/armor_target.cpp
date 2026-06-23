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
#include <opencv2/imgproc.hpp>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
namespace awakening::auto_aim {
using namespace armor_point_motion_model;

namespace {

    struct UVObservation {
        Eigen::Matrix<double, UVZ_N, 1> z;
        double length_px;
        double measurement_length;
    };

    bool is_light_in_armor(const ArmorKeyPoints2D& armor_key_points, const Light& light) {
        auto on_segment = [](const auto& a, const auto& b, const auto& p) {
            return std::min(a.x, b.x) <= p.x && p.x <= std::max(a.x, b.x)
                && std::min(a.y, b.y) <= p.y && p.y <= std::max(a.y, b.y);
        };
        auto cross = [](const auto& a, const auto& b, const auto& c) {
            const auto ab = b - a;
            const auto ac = c - a;
            return static_cast<double>(ab.x * ac.y - ab.y * ac.x);
        };
        auto segments_intersect = [&](const cv::Point2f& a1,
                                      const cv::Point2f& a2,
                                      const cv::Point2f& b1,
                                      const cv::Point2f& b2) {
            const double d1 = cross(a1, a2, b1);
            const double d2 = cross(a1, a2, b2);
            const double d3 = cross(b1, b2, a1);
            const double d4 = cross(b1, b2, a2);

            constexpr double EPS = 1e-6;
            if (((d1 > EPS && d2 < -EPS) || (d1 < -EPS && d2 > EPS))
                && ((d3 > EPS && d4 < -EPS) || (d3 < -EPS && d4 > EPS)))
            {
                return true;
            }

            if (std::abs(d1) <= EPS && on_segment(a1, a2, b1)) {
                return true;
            }
            if (std::abs(d2) <= EPS && on_segment(a1, a2, b2)) {
                return true;
            }
            if (std::abs(d3) <= EPS && on_segment(b1, b2, a1)) {
                return true;
            }
            if (std::abs(d4) <= EPS && on_segment(b1, b2, a2)) {
                return true;
            }
            return false;
        };
        const std::array<cv::Point2f, 4> quad = {
            armor_key_points.points[std::to_underlying(ArmorKeyPointsIndex::LEFT_TOP)].value(),
            armor_key_points.points[std::to_underlying(ArmorKeyPointsIndex::LEFT_BOTTOM)].value(),
            armor_key_points.points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_BOTTOM)].value(),
            armor_key_points.points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_TOP)].value()
        };
        const std::vector<cv::Point2f> polygon(quad.begin(), quad.end());

        if (cv::pointPolygonTest(polygon, light.top, false) >= 0
            || cv::pointPolygonTest(polygon, light.bottom, false) >= 0
            || cv::pointPolygonTest(polygon, light.center, false) >= 0)
        {
            return false;
        }

        for (std::size_t i = 0; i < quad.size(); ++i) {
            const auto& a = quad[i];
            const auto& b = quad[(i + 1) % quad.size()];
            if (segments_intersect(light.top, light.bottom, a, b)) {
                return false;
            }
        }

        return true;
    }

    UVObservation get_uv_measurement(
        const cv::Point2f& top,
        const cv::Point2f& bottom,
        const CameraInfo& camera_info
    ) noexcept {
        UVObservation observation;
        observation.z.setZero();
        observation.length_px = cv::norm(top - bottom);

        cv::Point2f measurement_top = top;
        cv::Point2f measurement_bottom = bottom;
        if (measure_normalized) {
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
        UVMeasure::points_to_observation(top_eigen, bottom_eigen, observation.z.data());
        observation.measurement_length = cv::norm(measurement_top - measurement_bottom);
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
    p0.diagonal()[idx::L] = p0.diagonal()[idx::R] = p0.diagonal()[idx::H] = 1;
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
    const auto u_q = [this]() {
        Eigen::Matrix<double, X_N, X_N> q;
        return q;
    };

    const auto inject =
        [this](const Eigen::Matrix<double, X_N, 1>& delta, Eigen::Matrix<double, X_N, 1>& nominal) {
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
        nominal_pose.linear() =
            utils::so3_exp(Vec3(nominal[idx::C_ROT_X], nominal[idx::C_ROT_Y], nominal[idx::C_ROT_Z])
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
    esekf = RobotStateESEKF(
        Predict { .dt = 0.005, .armor_number = target_number },
        u_q,
        inject,
        box_minus,
        p0
    );

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
}

void ArmorTarget::armor_pnp(
    Armor& a,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom,
    bool opt
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
    if (opt) {
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
Eigen::Matrix<double, UVZ_N, UVZ_N> ArmorTarget::uvmeasurement_covariance(
    const Eigen::Matrix<double, UVZ_N, 1>& z,
    double length_px,
    double measurement_length
) const noexcept {
    Eigen::Matrix<double, UVZ_N, UVZ_N> r;
    r.setZero();

    const double sigma_px = cfg.r_sigma_px;
    double sigma_x = sigma_px;
    double sigma_y = sigma_px;
    if (measure_normalized) {
        sigma_x /= uvmeasure_ctx.camera_info.camera_fx();
        sigma_y /= uvmeasure_ctx.camera_info.camera_fy();
    }
    double sigma_half_length = cfg.r_sigma_half_length;

    if (measure_normalized) {
        sigma_half_length /= uvmeasure_ctx.camera_info.camera_fx();
    }
    double sigma_angle = cfg.r_sigma_angle;
    sigma_angle *= std::cos(z(idx::UV_ANGLE));
    r(idx::UV_ANGLE, idx::UV_ANGLE) = sigma_angle * sigma_angle / 2.0;
    r(idx::UV_CENTER_X, idx::UV_CENTER_X) = sigma_x * sigma_x / 2.0;
    r(idx::UV_CENTER_Y, idx::UV_CENTER_Y) = sigma_y * sigma_y / 2.0;
    r(idx::UV_LENGTH, idx::UV_LENGTH) = sigma_half_length * sigma_half_length / 2.0;
    return r;
}

Eigen::Matrix<double, X_N, X_N> ArmorTarget::process_noise(double dt) const noexcept {
    Eigen::Matrix<double, X_N, X_N> q;
    Vec3 q_xyz;
    double q_yaw;
    if (target_number == ArmorClass::OUTPOST) {
        q_xyz = cfg.qxyz_output; // 前哨站加速度方差
        q_yaw = cfg.qyaw_output; // 前哨站角加速度方差
    } else {
        q_xyz = cfg.qxyz_common; // 加速度方差
        q_yaw = cfg.qyaw_common; // 角加速度方差
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
    q(idx::R, idx::R) = cfg.q_r;
    if (target_number == ArmorClass::OUTPOST) {
        q(idx::OUTPOST01DZ, idx::OUTPOST01DZ) = cfg.q_outpost_dz;
        q(idx::OUTPOST02DZ, idx::OUTPOST02DZ) = cfg.q_outpost_dz;
    } else {
        q(idx::L, idx::L) = cfg.q_l;
        q(idx::H, idx::H) = cfg.q_h;
    }
    constexpr std::array<int, 3> rot_idx { idx::C_ROT_X, idx::C_ROT_Y, idx::C_ROT_Z };
    const Vec3 yaw_axis_odom = car_in_odom_R * Vec3::UnitZ();
    for (int i = 0; i < 3; ++i) {
        const int ri = rot_idx[i];
        q(ri, idx::VYAW) += 0.5 * dt3 * q_yaw * yaw_axis_odom[i];
        q(idx::VYAW, ri) += 0.5 * dt3 * q_yaw * yaw_axis_odom[i];
        for (int j = 0; j < 3; ++j) {
            q(ri, rot_idx[j]) += 0.25 * dt4 * q_yaw * yaw_axis_odom[i] * yaw_axis_odom[j];
        }
    }
    q(idx::VYAW, idx::VYAW) += dt2 * q_yaw;
    const Mat3 Q_wpr_body = (Vec3(cfg.q_wpr, cfg.q_wpr, 0.0)).asDiagonal();
    const Mat3 Q_wpr_odom = car_in_odom_R * Q_wpr_body * car_in_odom_R.transpose();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            q(rot_idx[i], rot_idx[j]) += dt * Q_wpr_odom(i, j);
        }
    }
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
    bool all_init =
        (outpost_has_all_and_has_set_ids.has_value() ? outpost_has_all_and_has_set_ids.value().first
                                                     : jumped);
    uvmeasure_ctx.camera_cv_in_odom = camera_cv_in_odom;

    std::vector<std::shared_ptr<RobotStateESEKF::ObsBase>> obs;

    const auto cal_residual = [](const Eigen::Matrix<double, UVZ_N, 1>& z_pred,
                                 const Eigen::Matrix<double, UVZ_N, 1>& z) {
        return UVMeasure::residual(z_pred, z);
    };
    const int armors_num = armor_num();
    auto add_armor_uv_obs = [&](Armor& armor, int id, bool is_left) {
        auto ctx = uvmeasure_ctx;
        ctx.id = id;
        ctx.is_left = is_left;
        ctx.normalized = measure_normalized;
        auto key_points = armor.key_points.landmarks();
        UVObservation observation;
        if (is_left) {
            observation = get_uv_measurement(
                key_points[ArmorKeyPointsIndex::LEFT_TOP],
                key_points[ArmorKeyPointsIndex::LEFT_BOTTOM],
                camera_info
            );
        } else {
            observation = get_uv_measurement(
                key_points[ArmorKeyPointsIndex::RIGHT_TOP],
                key_points[ArmorKeyPointsIndex::RIGHT_BOTTOM],
                camera_info
            );
        }
        const auto u_r = [this,
                          length_px = observation.length_px,
                          measurement_length = observation.measurement_length](
                             const Eigen::Matrix<double, UVZ_N, 1>& z
                         ) { return uvmeasurement_covariance(z, length_px, measurement_length); };
        UVMeasure measure { .ctx = ctx };
        obs.push_back(esekf.value().make_obs(observation.z, measure, u_r, cal_residual));
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
    for (auto& [id, armor]: matched_armors) {
        jumped |= (id != 0);
        update_outpost_state(id);
        last_match_id = id;
        used_id[id] = true;
        add_armor_uv_obs(armor, id, true);
        add_armor_uv_obs(armor, id, false);
    }
    // if (matched_lights.size() >= 2) {
    for (const auto& [id, is_left, light]: matched_lights) {
        if (used_id[id]) {
            continue;
        }
        auto ctx = uvmeasure_ctx;
        ctx.id = id;
        used_id[id] = true;
        ctx.is_left = is_left;
        ctx.normalized = measure_normalized;
        UVMeasure measure { .ctx = ctx };
        const auto observation = get_uv_measurement(light.top, light.bottom, camera_info);
        const auto u_r = [this,
                          length_px = observation.length_px,
                          measurement_length = observation.measurement_length](
                             const Eigen::Matrix<double, UVZ_N, 1>& z
                         ) { return uvmeasurement_covariance(z, length_px, measurement_length); };
        obs.push_back(esekf.value().make_obs(observation.z, measure, u_r, cal_residual));
    }
    // }

    if (obs.size() > 0) {
        target_state.x = esekf.value().update_multi(obs);
        target_state.timestamp = timestamp;
        last_update = timestamp;
        this_id = GOBAL_ID++; //全局状态标记，下游控制对同一id的不重复构建轨迹
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
    const int armors_num = armor_num();
    bool all_init =
        (outpost_has_all_and_has_set_ids.has_value() ? outpost_has_all_and_has_set_ids.value().first
                                                     : jumped);
    std::vector<std::vector<double>> cost(n_obs, std::vector<double>(armors_num, MAX_COST + 1));
    std::vector<std::pair<std::pair<cv::Point2f, cv::Point2f>, std::pair<cv::Point2f, cv::Point2f>>>
        meas_list(n_obs);
    for (int j = 0; j < n_obs; ++j) {
        auto key_points = armors[j].key_points.landmarks();
        meas_list[j].first = std::make_pair(
            key_points[ArmorKeyPointsIndex::LEFT_TOP],
            key_points[ArmorKeyPointsIndex::LEFT_BOTTOM]
        );
        meas_list[j].second = std::make_pair(
            key_points[ArmorKeyPointsIndex::RIGHT_TOP],
            key_points[ArmorKeyPointsIndex::RIGHT_BOTTOM]
        );
    }
    auto pred_state = target_state;
    pred_state.predict(timestamp, target_number);
    for (int j = 0; j < n_obs; ++j) {
        bool in_gate = false;
        double min_pos_err = std::numeric_limits<double>::max();
        for (int id = 0; id < armors_num; ++id) {
            UVMeasure::Ctx tmp_ctx {
                .armor_num = armors_num,
                .id = id,
                .camera_cv_in_odom = camera_cv_in_odom,
                .camera_info = camera_info,
                .armor_number = target_number,
                .normalized = false,
            };
            UVMeasure measure { .ctx = tmp_ctx };
            measure.ctx.is_left = true;
            auto light1 = measure.projected_points(pred_state.x);
            measure.ctx.is_left = false;
            auto light2 = measure.projected_points(pred_state.x);
            auto avg_length_pred =
                (cv::norm(light1.first - light1.second) + cv::norm(light2.first - light2.second))
                / 2.0;
            // auto avg_length_meas =
            //     (cv::norm(meas_list[j].first.first - meas_list[j].first.second)
            //      + cv::norm(meas_list[j].second.first - meas_list[j].second.second))
            //     / 2.0;
            // if (std::abs(avg_length_pred - avg_length_meas) > avg_length_pred * 0.2) {
            //     continue;
            // }
            // auto avg_angle_pred = (std::atan2(light1.second.y - light1.first.y, light1.second.x - light1.first.x) +
            //                        std::atan2(light2.second.y - light2.first.y, light2.second.x - light2.first.x)) / 2.0;
            // auto avg_angle_meas = (std::atan2(meas_list[j].first.second.y - meas_list[j].first.first.y, meas_list[j].first.second.x - meas_list[j].first.first.x) +
            //                        std::atan2(meas_list[j].second.second.y - meas_list[j].second.first.y, meas_list[j].second.second.x - meas_list[j].second.first.x)) / 2.0;
            // if (std::abs(avg_angle_pred - avg_angle_meas) > 0.2) {
            //     continue;
            // }
            double pos_err = cv::norm(light1.first - meas_list[j].first.first)
                + cv::norm(light2.first - meas_list[j].second.first)
                + cv::norm(light1.second - meas_list[j].first.second)
                + cv::norm(light2.second - meas_list[j].second.second);

            if (std::isfinite(pos_err)
                && pos_err < avg_length_pred
                        * (all_init ? cfg.armor_match_pos_gate_all_init_by_length_ratio
                                    : cfg.armor_match_pos_gate_by_length_ratio))
            {
                cost[j][id] = pos_err;
                in_gate = true;
            }
            if (pos_err < min_pos_err) {
                min_pos_err = pos_err;
            }
        }
        if (!in_gate) {
            AWAKENING_WARN("match out of gate min pos_err: {}", min_pos_err);
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
    std::vector<std::tuple<int, bool, Light>> result;

    if (target_number == ArmorClass::BASE || matched_armors.size() != 1) {
        return result;
    }
    auto pred_state = target_state;
    pred_state.predict(timestamp, target_number);
    const int armors_num = armor_num();
    const int armor_id = matched_armors.front().first;
    auto& matched_armor = matched_armors.front().second;
    auto predict_light = [&](int id, bool is_left) -> std::pair<cv::Point2f, cv::Point2f> {
        UVMeasure::Ctx ctx { .armor_num = armors_num,
                             .id = id,
                             .camera_cv_in_odom = camera_cv_in_odom,
                             .camera_info = camera_info,
                             .armor_number = target_number,
                             .is_left = is_left };
        UVMeasure measure { .ctx = ctx };
        return measure.projected_points(pred_state.x);
    };
    struct VisibleLight {
        int armor_id;
        bool is_left;
        std::pair<cv::Point2f, cv::Point2f> projected_points;
    };
    const std::array visible_mapping {
        std::pair { (armor_id + armors_num - 1) % armors_num, false }, // 左可见
        std::pair { (armor_id + 1) % armors_num, true }, // 右可见
    };
    std::array<VisibleLight, visible_mapping.size()> visible_lights {
        VisibleLight { .armor_id = visible_mapping[0].first,
                       .is_left = visible_mapping[0].second,
                       .projected_points =
                           predict_light(visible_mapping[0].first, visible_mapping[0].second) },
        VisibleLight { .armor_id = visible_mapping[1].first,
                       .is_left = visible_mapping[1].second,
                       .projected_points =
                           predict_light(visible_mapping[1].first, visible_mapping[1].second) }
    };

    const int n_obs = static_cast<int>(lights.size());
    std::vector<std::array<double, visible_lights.size()>> cost(
        n_obs,
        { MAX_COST + 1, MAX_COST + 1 }
    );

    auto calc_cost = [&](const Light& light, const VisibleLight& visible_light) -> double {
        const auto& pred = visible_light.projected_points;
        for (auto& [_, armor]: matched_armors) {
            if (!is_light_in_armor(armor.key_points, light)) {
                return MAX_COST + 1;
            }
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
        const auto& matched_light = visible_lights[id];
        result.emplace_back(matched_light.armor_id, matched_light.is_left, lights[obs]);
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
    const cv::Rect img_rect = cv::Rect(0, 0, image_size.width, image_size.height);

    if ((rect & img_rect).area() <= 100) {
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
        {
            measure.ctx.is_left = true;
            const auto light = measure.projected_points(tmp_target_state.x);
            pts.push_back(light.first);
            pts.push_back(light.second);
        }
        {
            measure.ctx.is_left = false;
            const auto light = measure.projected_points(tmp_target_state.x);
            pts.push_back(light.first);
            pts.push_back(light.second);
        }
    }
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
    for (int id = 0; id < armors_num; ++id) {
        UVMeasure::Ctx tmp_ctx {
            .armor_num = armors_num,
            .id = id,
            .camera_cv_in_odom = camera_cv_in_odom,
            .camera_info = camera_info,
            .armor_number = target_number,
        };
        UVMeasure measure { .ctx = tmp_ctx };
        {
            measure.ctx.is_left = true;
            const auto light = measure.projected_points(tmp_target_state.x);
            pts.push_back(light.first);
            pts.push_back(light.second);
        }
        {
            measure.ctx.is_left = false;
            const auto light = measure.projected_points(tmp_target_state.x);
            pts.push_back(light.first);
            pts.push_back(light.second);
        }
    }
    return pts;
}
} // namespace awakening::auto_aim
