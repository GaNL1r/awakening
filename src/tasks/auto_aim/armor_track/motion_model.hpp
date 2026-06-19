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
namespace awakening::auto_aim::armor_point_motion_model {

namespace idx {
    enum { CX, VCX, CY, VCY, CZ, VCZ, C_ROT_Z, VYAW, R, P1, P2, C_ROT_Y, C_ROT_X, X_N };
    constexpr int L = P1;
    constexpr int H = P2;
    constexpr int OUTPOST01DZ = P1;
    constexpr int OUTPOST02DZ = P2;
    enum { TOP_X, TOP_Y, BOTTOM_X, BOTTOM_Y, _UVZ_N };
    enum { YPD_Y, YPD_P, YPD_D, A_ROT_X, A_ROT_Y, A_ROT_Z, _YPD_Z_N };
} // namespace idx
constexpr int X_N = idx::X_N;
constexpr int UVZ_N = idx::_UVZ_N;
constexpr int YPDZ_N = idx::_YPD_Z_N;
constexpr double OUTPOST_R = 0.27;
constexpr double OUTPOST_LEVEL_DZ = 0.1;
using VecX = Eigen::Matrix<double, X_N, 1>;
using UVVecZ = Eigen::Matrix<double, UVZ_N, 1>;
using YPDVecZ = Eigen::Matrix<double, YPDZ_N, 1>;
template<typename T>
inline T normalize_angle(T a) {
    const T two_pi = T(2.0 * M_PI);
    return a - two_pi * floor((a + T(M_PI)) / two_pi);
}

template<typename T>
inline Eigen::Matrix<T, 3, 3> _car_rotation(const T x[X_N], ArmorClass armor_number) {
    const bool building =
        (armor_number == auto_aim::ArmorClass::OUTPOST || armor_number == auto_aim::ArmorClass::BASE
        );
    if (building) {
        Eigen::Matrix<T, 3, 1> yaw_rotvec;
        yaw_rotvec << T(0), T(0), x[idx::C_ROT_Z];
        return utils::so3_exp(yaw_rotvec);
    }
    return utils::so3_exp(Eigen::Matrix<T, 3, 1>(x[idx::C_ROT_X], x[idx::C_ROT_Y], x[idx::C_ROT_Z]));
}

struct Predict {
    double dt { 0.0 };

    auto_aim::ArmorClass armor_number = auto_aim::ArmorClass::UNKNOWN;

    template<typename T>
    inline void operator()(const T x0[X_N], T x1[X_N]) const {
        std::copy(x0, x0 + X_N, x1);

        x1[idx::CX] += x0[idx::VCX] * T(dt);
        x1[idx::CY] += x0[idx::VCY] * T(dt);
        x1[idx::CZ] += x0[idx::VCZ] * T(dt);

        if (armor_number != auto_aim::ArmorClass::BASE) {
            Eigen::Matrix<T, 3, 1> delta_rot;
            delta_rot << T(0), T(0), x0[idx::VYAW] * T(dt);
            const Eigen::Matrix<T, 3, 3> R1 =
                (utils::so3_exp(Eigen::Matrix<T, 3, 1>(x0[idx::C_ROT_X], x0[idx::C_ROT_Y], x0[idx::C_ROT_Z])) * utils::so3_exp(delta_rot)).eval();
            auto rot_vec = utils::so3_log(R1);
            x1[idx::C_ROT_X] = rot_vec.x();
            x1[idx::C_ROT_Y] = rot_vec.y();
            x1[idx::C_ROT_Z] = rot_vec.z();
        }

        clamp(x1);
    }

    template<typename T>
    inline void clamp(T x[X_N]) const {
        auto& r = x[idx::R];
        auto& l = x[idx::L];
        auto& h = x[idx::H];
        auto& vyaw = x[idx::VYAW];
        if (armor_number != auto_aim::ArmorClass::OUTPOST) {
            if (r + l < T(0.1) || r + l > T(0.5)) {
                r = T(0.25);
                l = T(0);
            }

            if (ceres::abs(h) > T(0.5)) {
                h = T(0.0);
            }
        } else {
            r = T(OUTPOST_R);
            constrain_outpost_dz(x);
        }
        if (armor_number == auto_aim::ArmorClass::BASE
            || armor_number == auto_aim::ArmorClass::OUTPOST) {
            x[idx::C_ROT_Y] = T(0.0);
            x[idx::C_ROT_X] = T(0.0);
        }

        if (ceres::abs(vyaw) > T(20.0)) {
            vyaw = T(0.0);
        }
        if (armor_number == auto_aim::ArmorClass::BASE) {
            x[idx::VYAW] = T(0.0);
        }
    }
    inline void f(const VecX& x0, VecX& x1) const {
        assert(x0.size() == X_N);
        assert(x1.size() == X_N);
        operator()(x0.data(), x1.data());
    }

private:
    template<typename T>
    static inline void constrain_outpost_dz(T x[X_N]) {
        const T dz = T(OUTPOST_LEVEL_DZ);
        const T candidates[6][2] = {
            { -dz, -T(2.0) * dz }, { -T(2.0) * dz, -dz }, { -dz, dz },
            { dz, -dz },           { dz, T(2.0) * dz },   { T(2.0) * dz, dz },
        };

        T best_dz1 = candidates[0][0];
        T best_dz2 = candidates[0][1];
        T best_cost =
            squared(x[idx::OUTPOST01DZ] - best_dz1) + squared(x[idx::OUTPOST02DZ] - best_dz2);

        for (int i = 1; i < 6; ++i) {
            const T cost = squared(x[idx::OUTPOST01DZ] - candidates[i][0])
                + squared(x[idx::OUTPOST02DZ] - candidates[i][1]);
            if (cost < best_cost) {
                best_cost = cost;
                best_dz1 = candidates[i][0];
                best_dz2 = candidates[i][1];
            }
        }
        if (ceres::abs(x[idx::OUTPOST01DZ]) < T(1e-6)) {
            x[idx::OUTPOST01DZ] = T(0.0);
        } else {
            x[idx::OUTPOST01DZ] = best_dz1;
        }
        if (ceres::abs(x[idx::OUTPOST02DZ]) < T(1e-6)) {
            x[idx::OUTPOST02DZ] = T(0.0);
        } else {
            x[idx::OUTPOST02DZ] = best_dz2;
        }
    }

    template<typename T>
    static inline T squared(const T& value) {
        return value * value;
    }
};
template<typename T>
inline T _get_armor_r(const T x[X_N], int id, int armor_num) {
    const bool use_lh = (armor_num == 4) && (id & 1);
    return use_lh ? x[idx::R] + x[idx::L] : x[idx::R];
}
template<typename T>
inline Eigen::Transform<T, 3, Eigen::Isometry>
_whole_car_pose(const T x[X_N], ArmorClass armor_number) {
    Eigen::Transform<T, 3, Eigen::Isometry> car_in_odom =
        Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
    car_in_odom.translation() << x[idx::CX], x[idx::CY], x[idx::CZ];
    car_in_odom.linear() = _car_rotation(x, armor_number);
    return car_in_odom;
}
template<typename T>
inline Eigen::Transform<T, 3, Eigen::Isometry>
_armor_pose(const T x[X_N], int id, int armor_num, ArmorClass armor_number) {
    auto yaw = normalize_angle(T(id) * T(2.0 * M_PI / armor_num));
    const bool outpost = (armor_number == auto_aim::ArmorClass::OUTPOST);
    const bool use_lh = (armor_num == 4) && (id & 1);
    const T r = _get_armor_r(x, id, armor_num);
    auto ax = -ceres::cos(yaw) * r;
    auto ay = -ceres::sin(yaw) * r;
    T az;
    if (outpost) {
        az = (id == 0)  ? T(0)
            : (id == 1) ? T(0) + x[idx::OUTPOST01DZ]
            : (id == 2) ? T(0) + x[idx::OUTPOST02DZ]
                        : T(0);
    } else {
        az = use_lh ? T(0) + x[idx::H] : T(0);
    }
    Eigen::Transform<T, 3, Eigen::Isometry> pose_in_car;
    pose_in_car.translation() << ax, ay, az;

    const T armor_pitch = (armor_number == auto_aim::ArmorClass::OUTPOST)
        ? T(-auto_aim::FIFTTEN_DEGREE_RAD)
        : T(auto_aim::FIFTTEN_DEGREE_RAD);

    Eigen::Quaternion<T> q_yaw_in_car(Eigen::AngleAxis<T>(yaw, Eigen::Vector3<T>::UnitZ()));
    Eigen::Quaternion<T> q_pitch_in_car(Eigen::AngleAxis<T>(armor_pitch, Eigen::Vector3<T>::UnitY())
    );
    Eigen::Quaternion<T> q_roll_in_car(Eigen::AngleAxis<T>(T(0.0), Eigen::Vector3<T>::UnitX()));
    pose_in_car.linear() = (q_yaw_in_car * q_pitch_in_car * q_roll_in_car).toRotationMatrix();

    Eigen::Transform<T, 3, Eigen::Isometry> pose_in_odom =
        _whole_car_pose(x, armor_number) * pose_in_car;
    return pose_in_odom;
}
struct UVMeasure {
    struct Ctx {
        int armor_num { 4 };
        int id { 0 };
        ISO3 camera_cv_in_odom = ISO3::Identity();
        CameraInfo camera_info;
        auto_aim::ArmorClass armor_number = auto_aim::ArmorClass::UNKNOWN;
        bool is_left;
    } ctx;

    template<typename T>
    inline void operator()(const T x[X_N], T z[UVZ_N]) const {
        auto pose_in_odom = _armor_pose(x, ctx.id, ctx.armor_num, ctx.armor_number);

        Eigen::Transform<T, 3, Eigen::Isometry> camera_cv_in_odom_jet;
        camera_cv_in_odom_jet.matrix() = ctx.camera_cv_in_odom.matrix().template cast<T>();

        auto pose_in_camera_cv = camera_cv_in_odom_jet.inverse() * pose_in_odom;

        std::vector<cv::Point3f> object_points =
            getArmorLightKeyPoints3D<cv::Point3f>(ctx.armor_number, ctx.is_left);

        std::vector<Eigen::Matrix<T, 2, 1>> img_pts_jet;
        utils::project_points_jets(
            object_points,
            pose_in_camera_cv,
            ctx.camera_info.camera_matrix,
            ctx.camera_info.distortion_coefficients,
            img_pts_jet
        );

        z[idx::TOP_X] = img_pts_jet[0].x();
        z[idx::TOP_Y] = img_pts_jet[0].y();
        z[idx::BOTTOM_X] = img_pts_jet[1].x();
        z[idx::BOTTOM_Y] = img_pts_jet[1].y();
    }

    inline void h(const VecX& x, UVVecZ& z) const {
        operator()(x.data(), z.data());
    }
};
struct YPDMeasure {
    struct Ctx {
        int armor_num { 4 };
        int id { 0 };
        auto_aim::ArmorClass armor_number = auto_aim::ArmorClass::UNKNOWN;
    } ctx;

    template<typename T>
    inline void operator()(const T x[X_N], T z[YPDZ_N]) const {
        auto pose_in_odom = _armor_pose(x, ctx.id, ctx.armor_num, ctx.armor_number);
        T ax = pose_in_odom.translation().x();
        T ay = pose_in_odom.translation().y();
        T az = pose_in_odom.translation().z();
        auto rot_vec = utils::so3_log<T>(pose_in_odom.linear());
        T xy_dist = ceres::sqrt(ax * ax + ay * ay);
        T dist = ceres::sqrt(xy_dist * xy_dist + az * az);
        // Observation model
        z[idx::YPD_Y] = ceres::atan2(ay, ax); // yaw
        z[idx::YPD_P] = ceres::atan2(az, xy_dist); // pitch
        z[idx::YPD_D] = dist; // distance
        z[idx::A_ROT_X] = rot_vec.x();
        z[idx::A_ROT_Y] = rot_vec.y();
        z[idx::A_ROT_Z] = rot_vec.z();
    }

    inline void h(const VecX& x, YPDVecZ& z) const {
        operator()(x.data(), z.data());
    }
};

struct State {
    VecX x;
    TimePoint timestamp;
    int frame_id = 0;

    inline std::vector<ISO3> get_armors_pose(auto_aim::ArmorClass armor_number) const {
        std::vector<ISO3> r;
        int armor_num = armor_num_by_armor_class(armor_number);
        r.reserve(armor_num);
        for (int i = 0; i < armor_num; ++i) {
            auto pose = _armor_pose(x.data(), i, armor_num, armor_number);
            r.push_back(pose);
        }

        return r;
    }

    inline void predict(const TimePoint& t, auto_aim::ArmorClass armor_number) {
        auto dt = std::chrono::duration<double>(t - timestamp).count();
        predict(dt, armor_number);
    }
    inline void predict(double dt, auto_aim::ArmorClass armor_number) {
        Predict p { .dt = dt, .armor_number = armor_number };
        auto tmp_x = x;
        p.f(tmp_x, x);
        timestamp +=
            std::chrono::duration_cast<TimePoint::duration>(std::chrono::duration<double>(dt));
    }
    inline double get_armor_r(int id, auto_aim::ArmorClass armor_number) const {
        return _get_armor_r(x.data(), id, armor_num_by_armor_class(armor_number));
    }
    inline void set_pos(const Vec3& p) noexcept {
        x[idx::CX] = p.x();
        x[idx::CY] = p.y();
        x[idx::CZ] = p.z();
    }
    inline Vec3 pos() const noexcept {
        return Vec3(x[idx::CX], x[idx::CY], x[idx::CZ]);
    }
    inline void set_vel(const Vec3& v) noexcept {
        x[idx::VCX] = v.x();
        x[idx::VCY] = v.y();
        x[idx::VCZ] = v.z();
    }
    inline Vec3 vel() const noexcept {
        return Vec3(x[idx::VCX], x[idx::VCY], x[idx::VCZ]);
    }

    inline double yaw() const noexcept {
        return utils::matrix2rpy<double>(utils::so3_exp(Eigen::Matrix<double, 3, 1>(x[idx::C_ROT_X], x[idx::C_ROT_Y], x[idx::C_ROT_Z]))).z();
    }
    inline double vyaw() const noexcept {
        return x[idx::VYAW];
    }

    inline double r() const noexcept {
        return x[idx::R];
    }
    inline double l() const noexcept {
        return x[idx::L];
    }
    inline double h() const noexcept {
        return x[idx::H];
    }
    inline double outpost01DZ() const noexcept {
        return x[idx::OUTPOST01DZ];
    }
    inline double outpost02DZ() const noexcept {
        return x[idx::OUTPOST02DZ];
    }
    inline double w_p() const noexcept {
        return utils::matrix2rpy<double>(utils::so3_exp(Eigen::Matrix<double, 3, 1>(x[idx::C_ROT_X], x[idx::C_ROT_Y], x[idx::C_ROT_Z]))).y();
    }
    inline double w_r() const noexcept {
        return utils::matrix2rpy<double>(utils::so3_exp(Eigen::Matrix<double, 3, 1>(x[idx::C_ROT_X], x[idx::C_ROT_Y], x[idx::C_ROT_Z]))).x();
    }
};

// using RobotStateEKF = kalman_hybird_lib::ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;

using RobotStateESEKF = kalman_hybird_lib::ErrorStateEKF<X_N, Predict>;

} // namespace awakening::auto_aim::armor_point_motion_model
