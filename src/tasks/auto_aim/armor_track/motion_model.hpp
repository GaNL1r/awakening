#pragma once
#include "KalmanHyLib/error_state_extended_kalman_filter.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "utils/common/type_common.hpp"
#include "utils/utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <ceres/ceres.h>
#include <chrono>
#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>
namespace awakening::auto_aim::armor_point_motion_model {

namespace idx {
    enum { CX, VCX, CY, VCY, CZ, VCZ, C_ROT_Z, VYAW, R1, P1, P2, C_ROT_Y, C_ROT_X, X_N };
    constexpr int R2 = P1;
    constexpr int H = P2;
    constexpr int OUTPOST01DZ = P1;
    constexpr int OUTPOST02DZ = P2;
    enum { UV_ANGLE, UV_CENTER_X, UV_CENTER_Y, UV_LENGTH, _UVZ_N };
} // namespace idx
constexpr int X_N = idx::X_N;
constexpr int UVZ_N = idx::_UVZ_N;
constexpr double OUTPOST_R = 0.55/2.0;
constexpr double OUTPOST_WZ = 2.51;
using VecX = Eigen::Matrix<double, X_N, 1>;
using UVVecZ = Eigen::Matrix<double, UVZ_N, 1>;
template<typename T>
inline T normalize_angle(T a) {
    const T two_pi = T(2.0 * M_PI);
    return a - two_pi * ceres::floor((a + T(M_PI)) / two_pi);
}

template<typename T>
inline Eigen::Matrix<T, 3, 3> car_rotation(const T x[X_N], ArmorClass armor_number) {
    if (armor_number == ArmorClass::OUTPOST || armor_number == ArmorClass::BASE) {
        Eigen::Matrix<T, 3, 1> yaw_rotvec;
        yaw_rotvec << T(0), T(0), x[idx::C_ROT_Z];
        return utils::so3_exp(yaw_rotvec);
    }
    return utils::so3_exp(Eigen::Matrix<T, 3, 1>(x[idx::C_ROT_X], x[idx::C_ROT_Y], x[idx::C_ROT_Z])
    );
}
template<typename T>
inline T armor_radius(const T x[X_N], int id, int armor_num) {
    const bool is_r2 = (armor_num == 4) && (id & 1);
    return is_r2 ? x[idx::R2] : x[idx::R1];
}
template<typename T>
inline Eigen::Transform<T, 3, Eigen::Isometry>
whole_car_pose(const T x[X_N], ArmorClass armor_number) {
    Eigen::Transform<T, 3, Eigen::Isometry> car_in_odom =
        Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
    car_in_odom.translation() << x[idx::CX], x[idx::CY], x[idx::CZ];
    car_in_odom.linear() = car_rotation(x, armor_number);
    return car_in_odom;
}
template<typename T>
inline Eigen::Transform<T, 3, Eigen::Isometry>
armor_pose(const T x[X_N], int id, int armor_num, ArmorClass armor_number) {
    const T yaw = normalize_angle(T(id) * T(2.0 * M_PI / armor_num));
    const bool outpost = armor_number == ArmorClass::OUTPOST;
    const bool is_r2 = (armor_num == 4) && (id & 1);
    const T r = armor_radius(x, id, armor_num);
    const T ax = -ceres::cos(yaw) * r;
    const T ay = -ceres::sin(yaw) * r;
    T az = T(0);
    if (outpost) {
        if (id == 1)
            az = x[idx::OUTPOST01DZ];
        else if (id == 2)
            az = x[idx::OUTPOST02DZ];
    } else {
        az = is_r2 ? x[idx::H] : T(0);
    }
    auto pose_in_car = Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
    pose_in_car.translation() << ax, ay, az;

    const T armor_pitch = outpost ? T(-FIFTTEN_DEGREE_RAD) : T(FIFTTEN_DEGREE_RAD);
    pose_in_car.linear() =
        utils::rpy2matrix(Eigen::Vector3<T>(T(0), armor_pitch, yaw), utils::RPYOrder::ZYX);
    return whole_car_pose(x, armor_number) * pose_in_car;
}

template<class StateVector>
inline auto state_rotation(const StateVector& state) {
    using Scalar = typename std::decay_t<StateVector>::Scalar;
    using Vec3T = Eigen::Matrix<Scalar, 3, 1>;
    return utils::so3_exp(Vec3T(state[idx::C_ROT_X], state[idx::C_ROT_Y], state[idx::C_ROT_Z]));
}

template<class DeltaVector, class StateVector>
inline void inject_state(const DeltaVector& delta, StateVector& nominal) {
    using Scalar = typename std::decay_t<DeltaVector>::Scalar;
    using Vec3T = Eigen::Matrix<Scalar, 3, 1>;

    for (int i = 0; i < X_N; ++i) {
        const bool rotation_component = i == idx::C_ROT_X || i == idx::C_ROT_Y || i == idx::C_ROT_Z;
        if (!rotation_component)
            nominal[i] += delta[i];
    }

    const Vec3T delta_rot(delta[idx::C_ROT_X], delta[idx::C_ROT_Y], delta[idx::C_ROT_Z]);
    const Vec3T injected_rot =
        utils::so3_log((state_rotation(nominal) * utils::so3_exp(delta_rot)).eval());
    nominal[idx::C_ROT_X] = injected_rot.x();
    nominal[idx::C_ROT_Y] = injected_rot.y();
    nominal[idx::C_ROT_Z] = injected_rot.z();
}

template<class StateVector, class DeltaVector>
inline void
box_minus_state(const StateVector& nominal, const StateVector& value, DeltaVector& delta) {
    using Scalar = typename std::decay_t<StateVector>::Scalar;
    using Vec3T = Eigen::Matrix<Scalar, 3, 1>;

    delta = value - nominal;
    const Vec3T delta_rot =
        utils::so3_log((state_rotation(nominal).transpose() * state_rotation(value)).eval());
    delta[idx::C_ROT_X] = delta_rot.x();
    delta[idx::C_ROT_Y] = delta_rot.y();
    delta[idx::C_ROT_Z] = delta_rot.z();
}
struct Voter {
    enum {
        Collecting,
        Clockwise,
        Counterclockwise,
    } state = Collecting;
    void reset(const TimePoint& t) {
        *this = {};
        start_t = t;
    }
    void update(double yaw1, const TimePoint& t) {
        if (t - start_t < std::chrono::milliseconds(1000)) {
            return;
        }
        const double diff = angles::normalize_angle(yaw1 - last_state_yaw);
        if (std::abs(diff) < 0.05) {
            return;
        }
        if (diff > 0) {
            clock_wise_count++;
        } else {
            clock_wise_count--;
        }
        last_state_yaw = yaw1;
        if (std::abs(clock_wise_count) > 10) {
            state = clock_wise_count > 0 ? Clockwise : Counterclockwise;
        } else {
            state = Collecting;
        }
    }
    TimePoint start_t;
    int clock_wise_count = 0;
    double last_state_yaw = 0.0;
};
struct Predict {
    double dt { 0.0 };

    auto_aim::ArmorClass armor_number = auto_aim::ArmorClass::UNKNOWN;
    Voter voter;
    template<typename T>
    inline void operator()(const T x0[X_N], T x1[X_N]) const {
        std::copy(x0, x0 + X_N, x1);

        x1[idx::CX] += x0[idx::VCX] * T(dt);
        x1[idx::CY] += x0[idx::VCY] * T(dt);
        x1[idx::CZ] += x0[idx::VCZ] * T(dt);

        if (armor_number != auto_aim::ArmorClass::BASE) {
            Eigen::Matrix<T, 3, 1> delta_rot;
            if (armor_number == ArmorClass::OUTPOST) {
                if (voter.state == Voter::Clockwise) {
                    delta_rot << T(0), T(0), T(OUTPOST_WZ) * T(dt);
                    x1[idx::VYAW] = T(OUTPOST_WZ);
                } else if (voter.state == Voter::Counterclockwise) {
                    delta_rot << T(0), T(0), -T(OUTPOST_WZ) * T(dt);
                    x1[idx::VYAW] = -T(OUTPOST_WZ);
                } else {
                    delta_rot << T(0), T(0), x0[idx::VYAW] * T(dt);
                }
            } else {
                delta_rot << T(0), T(0), x0[idx::VYAW] * T(dt);
            }

            const Eigen::Matrix<T, 3, 3> R1 =
                (utils::so3_exp(
                     Eigen::Matrix<T, 3, 1>(x0[idx::C_ROT_X], x0[idx::C_ROT_Y], x0[idx::C_ROT_Z])
                 )
                 * utils::so3_exp(delta_rot))
                    .eval();
            auto rot_vec = utils::so3_log(R1);
            x1[idx::C_ROT_X] = rot_vec.x();
            x1[idx::C_ROT_Y] = rot_vec.y();
            x1[idx::C_ROT_Z] = rot_vec.z();
        }

        clamp(x1);
    }

    template<typename T>
    inline void clamp(T x[X_N]) const {
        // auto& r = x[idx::R];
        // auto& l = x[idx::L];
        auto& h = x[idx::H];
        auto& vyaw = x[idx::VYAW];
        if (armor_number != auto_aim::ArmorClass::OUTPOST) {
            // if (r + l < T(0.1) || r + l > T(0.5)) {
            //     r = T(0.25);
            //     l = T(0);
            // }

            if (ceres::abs(h) > T(0.5)) {
                h = T(0.0);
            }
        } else {
            if (ceres::abs(x[idx::OUTPOST01DZ]) > T(0.3)) {
                x[idx::OUTPOST01DZ] = T(0.0);
            }
            if (ceres::abs(x[idx::OUTPOST02DZ]) > T(0.3)) {
                x[idx::OUTPOST02DZ] = T(0.0);
            }
            x[idx::R1] = T(OUTPOST_R);
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
};

struct UVMeasure {
    template<typename T>
    using ImagePoint = Eigen::Matrix<T, 2, 1>;

    struct Ctx {
        int armor_num;
        int id;
        ISO3 camera_cv_in_odom = ISO3::Identity();
        CameraInfo camera_info;
        auto_aim::ArmorClass armor_number = auto_aim::ArmorClass::UNKNOWN;
        bool is_left;
        bool normalized = false;
    } ctx;

    template<typename T>
    inline std::array<ImagePoint<T>, 2> project_points(const T x[X_N]) const {
        auto pose_in_odom = armor_pose(x, ctx.id, ctx.armor_num, ctx.armor_number);

        Eigen::Transform<T, 3, Eigen::Isometry> camera_cv_in_odom_jet;
        camera_cv_in_odom_jet.matrix() = ctx.camera_cv_in_odom.matrix().template cast<T>();

        auto pose_in_camera_cv = camera_cv_in_odom_jet.inverse() * pose_in_odom;

        std::vector<cv::Point3f> object_points =
            getArmorLightKeyPoints3D<cv::Point3f>(ctx.armor_number, ctx.is_left);

        std::vector<ImagePoint<T>> pts_jet;
        if (ctx.normalized) {
            utils::project_points_jets_normalized(
                object_points,
                pose_in_camera_cv,
                ctx.camera_info.distortion_coefficients,
                pts_jet
            );
        } else {
            utils::project_points_jets(
                object_points,
                pose_in_camera_cv,
                ctx.camera_info.camera_matrix,
                ctx.camera_info.distortion_coefficients,
                pts_jet
            );
        }

        return { pts_jet[0], pts_jet[1] };
    }

    template<typename T>
    static inline void
    points_to_observation(const ImagePoint<T>& top, const ImagePoint<T>& bottom, T z[UVZ_N]) {
        const ImagePoint<T> delta = top - bottom;
        const ImagePoint<T> center = (top + bottom) / T(2);
        z[idx::UV_ANGLE] = ceres::atan2(delta.x(), delta.y());
        z[idx::UV_CENTER_X] = center.x();
        z[idx::UV_CENTER_Y] = center.y();
        z[idx::UV_LENGTH] = ceres::sqrt(delta.squaredNorm());
    }

    template<typename T>
    inline void operator()(const T x[X_N], T z[UVZ_N]) const {
        const auto points = project_points(x);
        points_to_observation(points[0], points[1], z);
    }

    inline std::pair<cv::Point2f, cv::Point2f> projected_points(const VecX& x) const {
        const auto points = project_points(x.data());
        return { cv::Point2f(points[0].x(), points[0].y()),
                 cv::Point2f(points[1].x(), points[1].y()) };
    }

    template<typename T>
    static inline Eigen::Matrix<T, UVZ_N, 1>
    residual(const Eigen::Matrix<T, UVZ_N, 1>& z_pred, const Eigen::Matrix<T, UVZ_N, 1>& z) {
        Eigen::Matrix<T, UVZ_N, 1> v = z - z_pred;
        v[idx::UV_ANGLE] = normalize_angle(v[idx::UV_ANGLE]);
        return v;
    }
    inline void h(const VecX& x, UVVecZ& z) const {
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
            auto pose = armor_pose(x.data(), i, armor_num, armor_number);
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
        return armor_radius(x.data(), id, armor_num_by_armor_class(armor_number));
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

    inline Vec3 rpy() const noexcept {
        return utils::matrix2rpy<double>(
            utils::so3_exp(Vec3(x[idx::C_ROT_X], x[idx::C_ROT_Y], x[idx::C_ROT_Z])),
            utils::RPYOrder::XYZ
        );
    }
    inline double yaw() const noexcept {
        return rpy().z();
    }
    inline double vyaw() const noexcept {
        return x[idx::VYAW];
    }

    inline double r1() const noexcept {
        return x[idx::R1];
    }
    inline double r2() const noexcept {
        return x[idx::R2];
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
        return rpy().y();
    }
    inline double w_r() const noexcept {
        return rpy().x();
    }
};

using RobotStateESEKF = kalman_hybird_lib::ErrorStateEKF<X_N, Predict>;

} // namespace awakening::auto_aim::armor_point_motion_model