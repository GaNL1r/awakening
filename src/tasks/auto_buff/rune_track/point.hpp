#pragma once
#include "KalmanHyLib/error_state_extended_kalman_filter.hpp"
#include "tasks/base/common.hpp"
#include "utils/common/type_common.hpp"
#include "utils/utils.hpp"
#include <algorithm>
#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>
namespace awakening::rune_point_motion_model {

constexpr int X_N = 11;
constexpr int Z_N = 8;

using VecX = Eigen::Matrix<double, X_N, 1>;
using VecZ = Eigen::Matrix<double, Z_N, 1>;

namespace idx {
    enum { CX, VCX, CY, VCY, CZ, VCZ, YAW, VYAW, R, P1, P2 };
    constexpr int L = P1;
    constexpr int H = P2;
    constexpr int OUTPOST01DZ = P1;
    constexpr int OUTPOST02DZ = P2;
    enum {
        LEFT_TOP_X,
        LEFT_TOP_Y,
        LEFT_BOTTOM_X,
        LEFT_BOTTOM_Y,
        RIGHT_BOTTOM_X,
        RIGHT_BOTTOM_Y,
        RIGHT_TOP_X,
        RIGHT_TOP_Y
    };
} // namespace idx

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

struct Measure {
    struct Ctx {
        int id { 0 };
        ISO3 camera_cv_in_odom = ISO3::Identity();
        CameraInfo camera_info;

    } ctx;

    template<typename T>
    inline void operator()(const T x[X_N], T z[Z_N]) const {}

    inline void h(const VecX& x, VecZ& z) const {
        operator()(x.data(), z.data());
    }
};
struct State {
    VecX x;
    TimePoint timestamp;
    int frame_id = 0;
};

// using RobotStateEKF = kalman_hybird_lib::ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;

using RobotStateESEKF = kalman_hybird_lib::ErrorStateEKF<X_N, Predict>;

} // namespace awakening::rune_point_motion_model