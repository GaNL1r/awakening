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
    enum { CX, VCX, CY, VCY, CZ, VCZ, YAW, VYAW, R, P1, P2, X_N };
    constexpr int L = P1;
    constexpr int H = P2;
    constexpr int OUTPOST01DZ = P1;
    constexpr int OUTPOST02DZ = P2;
    enum { TOP_X, TOP_Y, BOTTOM_X, BOTTOM_Y, _UVZ_N };
    enum { YPD_Y, YPD_P, YPD_D, ROT_YAW, _YPD_Z_N };
} // namespace idx
constexpr int X_N = idx::X_N;
constexpr int UVZ_N = idx::_UVZ_N;
constexpr int YPDZ_N = idx::_YPD_Z_N;
using VecX = Eigen::Matrix<double, X_N, 1>;
using UVVecZ = Eigen::Matrix<double, UVZ_N, 1>;
using YPDVecZ = Eigen::Matrix<double, YPDZ_N, 1>;
template<typename T>
inline T normalize_angle(T a) {
    const T two_pi = T(2.0 * M_PI);
    return a - two_pi * floor((a + T(M_PI)) / two_pi);
}
// template<typename T>
// inline Eigen::Vector<T, 3> armor_vel(const T x[X_N], int id, int armor_num) {
//     auto yaw = normalize_angle(x[idx::YAW] + T(id) * T(2.0 * M_PI / armor_num));
//     const bool use_lh = (armor_num == 4) && (id & 1);
//     const T r = use_lh ? x[idx::R] + x[idx::L] : x[idx::R];
//     Eigen::Vector<T, 3> p(-ceres::cos(yaw) * r, -ceres::sin(yaw) * r, T(0));
//     Eigen::Vector<T, 3> omega(0.0, 0.0, x[idx::VYAW]);

//     Eigen::Vector<T, 3> vel_armor_in_car = omega.cross(p);
//     Eigen::Vector<T, 3> vel_car_in_odom(x[idx::VCX], x[idx::VCY], x[idx::VCZ]);
//     Eigen::Transform<T, 3, Eigen::Isometry> car_in_odom =
//         Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
//     car_in_odom.translation() << x[idx::CX], x[idx::CY], x[idx::CZ];
//     Eigen::Quaternion<T> q_yaw_car_in_odom(Eigen::AngleAxis<T>(T(0.0), Eigen::Vector3<T>::UnitZ()));
//     Eigen::Quaternion<T> q_pitch_car_in_odom(Eigen::AngleAxis<T>(T(0.0), Eigen::Vector3<T>::UnitY())
//     );
//     Eigen::Quaternion<T> q_roll_car_in_odom(Eigen::AngleAxis<T>(T(0.0), Eigen::Vector3<T>::UnitX())
//     );
//     car_in_odom.linear() =
//         (q_yaw_car_in_odom * q_pitch_car_in_odom * q_roll_car_in_odom).toRotationMatrix();
//     Eigen::Vector<T, 3> vel_armor_in_odom =
//         vel_car_in_odom + (car_in_odom.linear() * vel_armor_in_car);
//     return vel_armor_in_odom;
// }
struct Predict {
    double dt { 0.0 };

    auto_aim::ArmorClass armor_number = auto_aim::ArmorClass::UNKNOWN;

    template<typename T>
    inline void operator()(const T x0[X_N], T x1[X_N]) const {
        std::copy(x0, x0 + X_N, x1);

        if (armor_number != auto_aim::ArmorClass::OUTPOST
            && armor_number != auto_aim::ArmorClass::BASE) {
            x1[idx::CX] += x0[idx::VCX] * T(dt);
            x1[idx::CY] += x0[idx::VCY] * T(dt);
            x1[idx::CZ] += x0[idx::VCZ] * T(dt);
        } else {
            x1[idx::VCX] = x1[idx::VCY] = x1[idx::VCZ] = T(0);
        }

        if (armor_number != auto_aim::ArmorClass::BASE) {
            x1[idx::YAW] += x0[idx::VYAW] * T(dt);
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
            r = T(0.27);
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
        auto pose_in_odom = armor_pose(x);

        Eigen::Transform<T, 3, Eigen::Isometry> camera_cv_in_odom_jet;
        camera_cv_in_odom_jet.matrix() = ctx.camera_cv_in_odom.matrix().template cast<T>();

        auto pose_in_camera_cv = camera_cv_in_odom_jet.inverse() * pose_in_odom;

        std::vector<cv::Point3f> object_points = getArmorKeyPoints3D<cv::Point3f>(ctx.armor_number);

        std::vector<Eigen::Matrix<T, 2, 1>> img_pts_jet;
        utils::project_points_jets(
            object_points,
            pose_in_camera_cv,
            ctx.camera_info.camera_matrix,
            ctx.camera_info.distortion_coefficients,
            img_pts_jet
        );
        if (ctx.is_left) {
            z[idx::TOP_X] =
                img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::LEFT_TOP)].x();
            z[idx::TOP_Y] =
                img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::LEFT_TOP)].y();
            z[idx::BOTTOM_X] =
                img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::LEFT_BOTTOM)].x();
            z[idx::BOTTOM_Y] =
                img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::LEFT_BOTTOM)].y();
        } else {
            z[idx::TOP_X] =
                img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::RIGHT_TOP)].x();
            z[idx::TOP_Y] =
                img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::RIGHT_TOP)].y();
            z[idx::BOTTOM_X] =
                img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::RIGHT_BOTTOM)].x();
            z[idx::BOTTOM_Y] =
                img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::RIGHT_BOTTOM)].y();
        }
    }

    inline void h(const VecX& x, UVVecZ& z) const {
        operator()(x.data(), z.data());
    }

    template<typename T>
    inline T get_armor_r(const T x[X_N]) const {
        const bool use_lh = (ctx.armor_num == 4) && (ctx.id & 1);
        return use_lh ? x[idx::R] + x[idx::L] : x[idx::R];
    }

    template<typename T>
    inline Eigen::Transform<T, 3, Eigen::Isometry> armor_pose(const T x[X_N]) const {
        auto yaw = normalize_angle(x[idx::YAW] + T(ctx.id) * T(2.0 * M_PI / ctx.armor_num));

        const bool outpost = (ctx.armor_number == auto_aim::ArmorClass::OUTPOST);
        const bool use_lh = (ctx.armor_num == 4) && (ctx.id & 1);

        const T r = get_armor_r(x);

        auto ax = -ceres::cos(yaw) * r;
        auto ay = -ceres::sin(yaw) * r;
        T az;
        if (outpost) {
            az = (ctx.id == 0)  ? T(0)
                : (ctx.id == 1) ? T(0) + x[idx::OUTPOST01DZ]
                : (ctx.id == 2) ? T(0) + x[idx::OUTPOST02DZ]
                                : T(0);
        } else {
            az = use_lh ? T(0) + x[idx::H] : T(0);
        }
        Eigen::Transform<T, 3, Eigen::Isometry> pose_in_car;
        pose_in_car.translation() << ax, ay, az;

        const T armor_pitch = (ctx.armor_number == auto_aim::ArmorClass::OUTPOST)
            ? T(-auto_aim::FIFTTEN_DEGREE_RAD)
            : T(auto_aim::FIFTTEN_DEGREE_RAD);

        Eigen::Quaternion<T> q_yaw_in_car(Eigen::AngleAxis<T>(yaw, Eigen::Vector3<T>::UnitZ()));
        Eigen::Quaternion<T> q_pitch_in_car(
            Eigen::AngleAxis<T>(armor_pitch, Eigen::Vector3<T>::UnitY())
        );
        Eigen::Quaternion<T> q_roll_in_car(Eigen::AngleAxis<T>(T(0.0), Eigen::Vector3<T>::UnitX()));
        pose_in_car.linear() = (q_yaw_in_car * q_pitch_in_car * q_roll_in_car).toRotationMatrix();
        Eigen::Transform<T, 3, Eigen::Isometry> car_in_odom =
            Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
        car_in_odom.translation() << x[idx::CX], x[idx::CY], x[idx::CZ];
        Eigen::Quaternion<T> q_yaw_car_in_odom(
            Eigen::AngleAxis<T>(T(0.0), Eigen::Vector3<T>::UnitZ())
        );
        Eigen::Quaternion<T> q_pitch_car_in_odom(
            Eigen::AngleAxis<T>(T(0.0), Eigen::Vector3<T>::UnitY())
        );
        Eigen::Quaternion<T> q_roll_car_in_odom(
            Eigen::AngleAxis<T>(T(0.0), Eigen::Vector3<T>::UnitX())
        );
        car_in_odom.linear() =
            (q_yaw_car_in_odom * q_pitch_car_in_odom * q_roll_car_in_odom).toRotationMatrix(); // 本来想考虑整车在空间旋转，不过有点毛病
        Eigen::Transform<T, 3, Eigen::Isometry> pose_in_odom = car_in_odom * pose_in_car;
        return pose_in_odom;
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
        auto yaw = normalize_angle(x[idx::YAW] + T(ctx.id) * T(2.0 * M_PI / ctx.armor_num));

        const bool outpost = (ctx.armor_number == auto_aim::ArmorClass::OUTPOST);
        const bool use_lh = (ctx.armor_num == 4) && (ctx.id & 1);

        const T r = get_armor_r(x);

        auto ax = x[idx::CX] - ceres::cos(yaw) * r;
        auto ay = x[idx::CY] - ceres::sin(yaw) * r;
        T az;
        if (outpost) {
            az = (ctx.id == 0)  ? x[idx::CZ]
                : (ctx.id == 1) ? x[idx::CZ] + x[idx::OUTPOST01DZ]
                : (ctx.id == 2) ? x[idx::CZ] + x[idx::OUTPOST02DZ]
                                : x[idx::CZ];
        } else {
            az = use_lh ? x[idx::CZ] + x[idx::H] : x[idx::CZ];
        }
        T xy_dist = ceres::sqrt(ax * ax + ay * ay);
        T dist = ceres::sqrt(xy_dist * xy_dist + az * az);
        // Observation model
        z[idx::YPD_Y] = ceres::atan2(ay, ax); // yaw
        z[idx::YPD_P] = ceres::atan2(az, xy_dist); // pitch
        z[idx::YPD_D] = dist; // distance
        z[idx::ROT_YAW] = yaw; // orientation_yaw
    }

    inline void h(const VecX& x, YPDVecZ& z) const {
        operator()(x.data(), z.data());
    }

    template<typename T>
    inline T get_armor_r(const T x[X_N]) const {
        const bool use_lh = (ctx.armor_num == 4) && (ctx.id & 1);
        return use_lh ? x[idx::R] + x[idx::L] : x[idx::R];
    }
};

struct State {
    VecX x;
    TimePoint timestamp;
    int frame_id = 0;

    inline std::vector<Vec4> get_armors_xyza(auto_aim::ArmorClass armor_number) const {
        std::vector<Vec4> r;
        int armor_num = armor_num_by_armor_class(armor_number);
        r.reserve(armor_num);
        for (int i = 0; i < armor_num; ++i) {
            UVMeasure::Ctx ctx;
            ctx.id = i;
            ctx.armor_num = armor_num;
            ctx.armor_number = armor_number;
            UVMeasure m {
                .ctx = ctx,
            };
            auto pose_in_odom = m.armor_pose(x.data());
            double ax, ay, az, ayaw;
            ax = pose_in_odom.translation().x();
            ay = pose_in_odom.translation().y();
            az = pose_in_odom.translation().z();
            auto ypr = utils::matrix2euler(pose_in_odom.linear(), utils::EulerOrder::ZYX);
            ayaw = ypr[0];
            r.push_back({ ax, ay, az, ayaw });
        }
        return r;
    }

    inline std::vector<ISO3> get_armors_pose(auto_aim::ArmorClass armor_number) const {
        std::vector<ISO3> r;
        const double armor_pitch = (armor_number == auto_aim::ArmorClass::OUTPOST)
            ? -auto_aim::FIFTTEN_DEGREE_RAD
            : auto_aim::FIFTTEN_DEGREE_RAD;
        int armor_num = armor_num_by_armor_class(armor_number);
        r.reserve(armor_num);
        for (int i = 0; i < armor_num; ++i) {
            UVMeasure::Ctx ctx;
            ctx.id = i;
            ctx.armor_num = armor_num;
            ctx.armor_number = armor_number;
            UVMeasure m {
                .ctx = ctx,
            };
            ISO3 pose = m.armor_pose(x.data());

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
        p.f(x, x);
        timestamp +=
            std::chrono::duration_cast<TimePoint::duration>(std::chrono::duration<double>(dt));
    }
    inline double get_armor_r(int id, auto_aim::ArmorClass armor_number) const {
        UVMeasure::Ctx ctx {
            .armor_num = armor_num_by_armor_class(armor_number),
            .id = id,
        };
        UVMeasure m {
            .ctx = ctx,
        };
        return m.get_armor_r(x.data());
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
        return x[idx::YAW];
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
};

// using RobotStateEKF = kalman_hybird_lib::ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;

using RobotStateESEKF = kalman_hybird_lib::ErrorStateEKF<X_N, Predict>;

} // namespace awakening::auto_aim::armor_point_motion_model