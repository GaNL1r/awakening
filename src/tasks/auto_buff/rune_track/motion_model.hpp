#pragma once
#include "KalmanHyLib/error_state_extended_kalman_filter.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/auto_buff/type.hpp"
#include "tasks/base/common.hpp"
#include "utils/common/type_common.hpp"
#include "utils/utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <chrono>
#include <cstdlib>
#include <opencv2/core/types.hpp>
#include <optional>
#include <utility>
#include <vector>
namespace awakening::auto_buff::motion_model {

namespace idx {
    enum { CX, CY, CZ, YAW, ROLL, V_ROLL, X_N };
    enum { R_X, R_Y, _R_Z_N };
    enum { TOP_X, TOP_Y, LEFT_X, LEFT_Y, BOTTOM_X, BOTTOM_Y, RIGHT_X, RIGHT_Y, _FanBlade_Z_N };

    enum { YPD_Y, YPD_P, YPD_D, ROT_YAW, ROT_ROLL, _YPD_Z_N };
} // namespace idx
constexpr int X_N = idx::X_N;
constexpr int RZ_N = idx::_R_Z_N;
constexpr int FanBladeZ_N = idx::_FanBlade_Z_N;
constexpr int YPDZ_N = idx::_YPD_Z_N;

using VecX = Eigen::Matrix<double, X_N, 1>;
using RVecZ = Eigen::Matrix<double, RZ_N, 1>;
using FanBladeVecZ = Eigen::Matrix<double, FanBladeZ_N, 1>;
using YPDVecZ = Eigen::Matrix<double, YPDZ_N, 1>;
template<typename T>
inline T normalize_angle(T a) {
    const T two_pi = T(2.0 * M_PI);
    return a - two_pi * floor((a + T(M_PI)) / two_pi);
}

struct Predict {
    double dt { 0.0 };

    template<typename T>
    inline void operator()(const T x0[X_N], T x1[X_N]) const {
        std::copy(x0, x0 + X_N, x1);
        x1[idx::ROLL] += x0[idx::V_ROLL] * T(dt);

        clamp(x1);
    }

    template<typename T>
    inline void clamp(T x[X_N]) const {}
    inline void f(const VecX& x0, VecX& x1) const {
        assert(x0.size() == X_N);
        assert(x1.size() == X_N);
        operator()(x0.data(), x1.data());
    }
};

struct RMeasure {
    struct Ctx {
        ISO3 camera_cv_in_odom = ISO3::Identity();
        CameraInfo camera_info;
    } ctx;

    template<typename T>
    inline void operator()(const T x[X_N], T z[RZ_N]) const {
        Eigen::Transform<T, 3, Eigen::Isometry> pose_in_odom =
            Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
        pose_in_odom.translation() << x[idx::CX], x[idx::CY], x[idx::CZ];

        Eigen::Transform<T, 3, Eigen::Isometry> camera_cv_in_odom_jet;
        camera_cv_in_odom_jet.matrix() = ctx.camera_cv_in_odom.matrix().template cast<T>();

        auto pose_in_camera_cv = camera_cv_in_odom_jet.inverse() * pose_in_odom;
        std::vector<Eigen::Matrix<T, 2, 1>> img_pts_jet;
        utils::project_points_jets(
            { cv::Point3f(0.1, 0, 0) },
            pose_in_camera_cv,
            ctx.camera_info.camera_matrix,
            ctx.camera_info.distortion_coefficients,
            img_pts_jet
        );
        z[idx::R_X] = img_pts_jet[0].x();
        z[idx::R_Y] = img_pts_jet[0].y();
    }

    inline void h(const VecX& x, RVecZ& z) const {
        operator()(x.data(), z.data());
    }
};
struct FanBladeMeasure {
    struct Ctx {
        int id = 0;
        ISO3 camera_cv_in_odom = ISO3::Identity();
        CameraInfo camera_info;
    } ctx;

    template<typename T>
    inline void operator()(const T x[X_N], T z[RZ_N]) const {
        auto pose_in_odom = fan_pose(x);
        Eigen::Transform<T, 3, Eigen::Isometry> camera_cv_in_odom_jet;
        camera_cv_in_odom_jet.matrix() = ctx.camera_cv_in_odom.matrix().template cast<T>();

        auto pose_in_camera_cv = camera_cv_in_odom_jet.inverse() * pose_in_odom;
        std::vector<Eigen::Matrix<T, 2, 1>> img_pts_jet;
        utils::project_points_jets(
            RuneKeyPoint3D<cv::Point3f>::build(),
            pose_in_camera_cv,
            ctx.camera_info.camera_matrix,
            ctx.camera_info.distortion_coefficients,
            img_pts_jet
        );
        z[idx::TOP_X] = img_pts_jet[RuneKeyPointsIndex::TOP].x();
        z[idx::TOP_Y] = img_pts_jet[RuneKeyPointsIndex::TOP].y();
        z[idx::LEFT_X] = img_pts_jet[RuneKeyPointsIndex::LEFT].x();
        z[idx::LEFT_Y] = img_pts_jet[RuneKeyPointsIndex::LEFT].y();
        z[idx::BOTTOM_X] = img_pts_jet[RuneKeyPointsIndex::BOTTOM].x();
        z[idx::BOTTOM_Y] = img_pts_jet[RuneKeyPointsIndex::BOTTOM].y();
        z[idx::RIGHT_X] = img_pts_jet[RuneKeyPointsIndex::RIGHT].x();
        z[idx::RIGHT_Y] = img_pts_jet[RuneKeyPointsIndex::RIGHT].y();
    }
    template<typename T>
    inline Eigen::Transform<T, 3, Eigen::Isometry> fan_pose(const T x[X_N]) const {
        auto roll = normalize_angle(x[idx::ROLL] + T(ctx.id) * T(2.0 * M_PI / FAN_NUM));

        Eigen::Transform<T, 3, Eigen::Isometry> rune_in_odom =
            Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
        rune_in_odom.translation() << x[idx::CX], x[idx::CY], x[idx::CZ];
        auto yaw = ceres::atan2(x[idx::CY], x[idx::CX]);
        // auto yaw = x[idx::YAW];
        Eigen::Quaternion<T> q_yaw_rune_in_odom(Eigen::AngleAxis<T>(yaw, Eigen::Vector3<T>::UnitZ())
        );
        Eigen::Quaternion<T> q_pitch_rune_in_odom(
            Eigen::AngleAxis<T>(T(0.0), Eigen::Vector3<T>::UnitY())
        );
        Eigen::Quaternion<T> q_roll_rune_in_odom(
            Eigen::AngleAxis<T>(roll, Eigen::Vector3<T>::UnitX())
        );
        rune_in_odom.linear() =
            (q_yaw_rune_in_odom * q_pitch_rune_in_odom * q_roll_rune_in_odom).toRotationMatrix();
        Eigen::Transform<T, 3, Eigen::Isometry> pose_in_odom = rune_in_odom;
        return pose_in_odom;
    }
    template<typename T>
    inline Eigen::Transform<T, 3, Eigen::Isometry> fan_target_pose(const T x[X_N]) const {
        auto roll = normalize_angle(x[idx::ROLL] + T(ctx.id) * T(2.0 * M_PI / FAN_NUM));

        Eigen::Transform<T, 3, Eigen::Isometry> fan_target_in_rune =
            Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
        fan_target_in_rune.translation() << T(0), T(0), T(RUNE_R2_FAN_TARGET_CENTER);
        Eigen::Transform<T, 3, Eigen::Isometry> rune_in_odom =
            Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
        rune_in_odom.translation() << x[idx::CX], x[idx::CY], x[idx::CZ];
        auto yaw = ceres::atan2(x[idx::CY], x[idx::CX]);
        // auto yaw = x[idx::YAW];
        Eigen::Quaternion<T> q_yaw_rune_in_odom(Eigen::AngleAxis<T>(yaw, Eigen::Vector3<T>::UnitZ())
        );
        Eigen::Quaternion<T> q_pitch_rune_in_odom(
            Eigen::AngleAxis<T>(T(0.0), Eigen::Vector3<T>::UnitY())
        );
        Eigen::Quaternion<T> q_roll_rune_in_odom(
            Eigen::AngleAxis<T>(roll, Eigen::Vector3<T>::UnitX())
        );
        rune_in_odom.linear() =
            (q_yaw_rune_in_odom * q_pitch_rune_in_odom * q_roll_rune_in_odom).toRotationMatrix();
        Eigen::Transform<T, 3, Eigen::Isometry> pose_in_odom = rune_in_odom * fan_target_in_rune;
        return pose_in_odom;
    }

    inline void h(const VecX& x, FanBladeVecZ& z) const {
        operator()(x.data(), z.data());
    }
};
struct YPDMeasure {
    struct Ctx {
        int id = 0;
    } ctx;

    template<typename T>
    inline void operator()(const T x[X_N], T z[RZ_N]) const {
        auto roll = normalize_angle(x[idx::ROLL] + T(ctx.id) * T(2.0 * M_PI / FAN_NUM));
        T cx = x[idx::CX];
        T cy = x[idx::CY];
        T cz = x[idx::CZ];

        T xy_dist = ceres::sqrt(cx * cx + cy * cy);
        T dist = ceres::sqrt(xy_dist * xy_dist + cz * cz);
        // Observation model
        z[idx::YPD_Y] = ceres::atan2(cy, cx); // yaw
        z[idx::YPD_P] = ceres::atan2(cz, xy_dist); // pitch
        z[idx::YPD_D] = dist; // distance
        z[idx::ROT_YAW] = x[idx::YAW]; // orientation_yaw
        z[idx::ROT_ROLL] = roll; // orientation_roll
    }

    inline void h(const VecX& x, YPDVecZ& z) const {
        operator()(x.data(), z.data());
    }
};

struct State {
    VecX x;
    TimePoint timestamp;
    int frame_id = 0;
    inline std::vector<ISO3> get_fan_target_pose() const {
        std::vector<ISO3> r;
        for (int i = 0; i < FAN_NUM; ++i) {
            FanBladeMeasure::Ctx ctx;
            ctx.id = i;
            FanBladeMeasure m {
                .ctx = ctx,
            };
            ISO3 pose = m.fan_target_pose(x.data());

            r.push_back(pose);
        }

        return r;
    }
    inline std::vector<ISO3> get_fan_pose() const {
        std::vector<ISO3> r;
        for (int i = 0; i < FAN_NUM; ++i) {
            FanBladeMeasure::Ctx ctx;
            ctx.id = i;
            FanBladeMeasure m {
                .ctx = ctx,
            };
            ISO3 pose = m.fan_pose(x.data());

            r.push_back(pose);
        }

        return r;
    }
    inline void predict(const TimePoint& t) {
        auto dt = std::chrono::duration<double>(t - timestamp).count();
        predict(dt);
    }
    inline void predict(double dt) {
        Predict p {
            .dt = dt,
        };
        p.f(x, x);
        timestamp +=
            std::chrono::duration_cast<TimePoint::duration>(std::chrono::duration<double>(dt));
    }
    Vec3 pos() const {
        return Vec3(x[idx::CX], x[idx::CY], x[idx::CZ]);
    }
    double yaw() const {
        return x[idx::YAW];
    }
    double roll() const {
        return x[idx::ROLL];
    }
    double v_roll() const {
        return x[idx::V_ROLL];
    }
};

// using RobotStateEKF = kalman_hybird_lib::ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;

using ESEKF = kalman_hybird_lib::ErrorStateEKF<X_N, Predict>;

} // namespace awakening::auto_buff::motion_model