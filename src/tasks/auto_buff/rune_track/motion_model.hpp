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
#include <optional>
#include <utility>
#include <vector>
namespace awakening::auto_buff::motion_model {

namespace idx {
    enum { CX, CY, CZ, YAW, ROLL, V_ROLL, X_N };
    enum {  R_X, R_Y, R_Z_N };
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
        int id { 0 };
        ISO3 camera_cv_in_odom = ISO3::Identity();
        CameraInfo camera_info;
    } ctx;

    template<typename T>
    inline void operator()(const T x[X_N], T z[RZ_N]) const {}

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