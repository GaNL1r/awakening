#pragma once
#include "KalmanHyLib/error_state_extended_kalman_filter.hpp"
#include "tasks/auto_aim/type.hpp"
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
    enum { R_X, R_Y, R_Z_N };
    enum { YPD_Y, YPD_P, YPD_D, ROT_YAW, ROT_ROLL, _YPD_Z_N };
} // namespace idx
constexpr int X_N = idx::X_N;
constexpr int RZ_N = idx::R_Z_N;
constexpr int YPDZ_N = idx::_YPD_Z_N;
using VecX = Eigen::Matrix<double, X_N, 1>;
using RVecZ = Eigen::Matrix<double, RZ_N, 1>;
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
            { cv::Point3f(0, 0, 0) },
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

struct State {
    VecX x;
    TimePoint timestamp;
    int frame_id = 0;
};

// using RobotStateEKF = kalman_hybird_lib::ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;

using ESEKF = kalman_hybird_lib::ErrorStateEKF<X_N, Predict>;

} // namespace awakening::auto_buff::motion_model