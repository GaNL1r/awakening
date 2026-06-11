#pragma once
#include "angles.h"
#include "utils/common/type_common.hpp"
#include <cmath>
#include <numbers>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <optional>
#include <pwd.h>
#include <regex>
#include <utility>
#include <vector>
namespace awakening::utils {

inline Quaternion rpy2quat(const Vec3& rpy) {
    AngleAxis roll(rpy.x(), Vec3::UnitX());
    AngleAxis pitch(rpy.y(), Vec3::UnitY());
    AngleAxis yaw(rpy.z(), Vec3::UnitZ());
    Quaternion q { yaw * pitch * roll };
    q.normalize();
    return q;
}

inline Mat3 rpy2matrix(const Vec3& rpy) {
    return rpy2quat(rpy).toRotationMatrix();
}

inline Vec3 matrix2rpy(const Mat3& R) {
    const double roll = std::atan2(R(2, 1), R(2, 2));
    const double pitch = std::atan2(-R(2, 0), std::hypot(R(2, 1), R(2, 2)));
    const double yaw = std::atan2(R(1, 0), R(0, 0));
    return { roll, pitch, yaw };
}

inline Vec3 quat2rpy(const Quaternion& q) {
    return matrix2rpy(q.normalized().toRotationMatrix());
}

inline std::string expand_env(const std::string& s) {
    std::regex env_re(R"(\$\{([^}]+)\})");
    std::smatch match;
    std::string result = s;
    while (std::regex_search(result, match, env_re)) {
        const char* env = std::getenv(match[1].str().c_str());
        std::string val = env ? env : "";
        result.replace(match.position(0), match.length(0), val);
    }
    return result;
}
template<typename Func>
void dt_once(Func&& func, std::chrono::duration<double> dt) noexcept {
    static auto last_call = std::chrono::steady_clock::now();

    auto now = std::chrono::steady_clock::now();
    if (now - last_call >= dt) {
        last_call = now;
        func();
    }
}
template<typename T>
concept Point2DLike = requires(T p) {
    {
        p.x
        } -> std::convertible_to<float>;
    {
        p.y
        } -> std::convertible_to<float>;
    T { 0.f, 0.f };
};
template<Point2DLike T>
[[nodiscard]] inline T transform_point2D(const Eigen::Matrix3f& H, const T& p) noexcept {
    const Eigen::Vector3f hp { p.x, p.y, 1.f };
    const Eigen::Vector3f tp = H * hp;
    return { tp.x(), tp.y() };
}
inline cv::Rect2f transform_rect(const Eigen::Matrix3f& H, const cv::Rect2f& rect) {
    cv::Point2f p1(rect.x, rect.y);
    cv::Point2f p2(rect.x + rect.width, rect.y);
    cv::Point2f p3(rect.x, rect.y + rect.height);
    cv::Point2f p4(rect.x + rect.width, rect.y + rect.height);

    auto tp1 = utils::transform_point2D(H, p1);
    auto tp2 = utils::transform_point2D(H, p2);
    auto tp3 = utils::transform_point2D(H, p3);
    auto tp4 = utils::transform_point2D(H, p4);

    float min_x = std::min({ tp1.x, tp2.x, tp3.x, tp4.x });
    float min_y = std::min({ tp1.y, tp2.y, tp3.y, tp4.y });
    float max_x = std::max({ tp1.x, tp2.x, tp3.x, tp4.x });
    float max_y = std::max({ tp1.y, tp2.y, tp3.y, tp4.y });

    return cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
}
inline cv::Mat letterbox(
    const cv::Mat& img,
    Eigen::Matrix3f& transform_matrix,
    const int new_shape_w,
    const int new_shape_h
) noexcept {
    const int img_h = img.rows;
    const int img_w = img.cols;

    const float scale = std::min((float)new_shape_h / img_h, (float)new_shape_w / img_w);
    const int resize_h = int(img_h * scale + 0.5f);
    const int resize_w = int(img_w * scale + 0.5f);

    const int pad_h = new_shape_h - resize_h;
    const int pad_w = new_shape_w - resize_w;
    const int top = pad_h / 2;
    const int left = pad_w / 2;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat out;
    if (pad_h == 0 && pad_w == 0) {
        out = resized;
    } else {
        cv::copyMakeBorder(
            resized,
            out,
            top,
            pad_h - top,
            left,
            pad_w - left,
            cv::BORDER_CONSTANT,
            cv::Scalar(114, 114, 114)
        );
    }

    const float inv_scale = 1.0f / scale;

    transform_matrix << inv_scale, 0, -left * inv_scale, 0, inv_scale, -top * inv_scale, 0, 0, 1;

    return out;
}
inline std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
    return s;
}
template<std::size_t N1, std::size_t N2>
consteval auto concat(const char (&a)[N1], const char (&b)[N2]) {
    std::array<char, N1 + N2 - 1> result {}; // -1 是因为两个字面量都有 '\0'
    for (std::size_t i = 0; i < N1 - 1; ++i)
        result[i] = a[i];
    for (std::size_t i = 0; i < N2; ++i)
        result[i + N1 - 1] = b[i]; // 包含 '\0'
    return result;
}
template<typename T>
inline T from_vector(const std::vector<uint8_t>& data) {
    T packet {};
    std::memcpy(&packet, data.data(), sizeof(T));
    return packet;
}

template<typename T>
inline std::vector<uint8_t> to_vector(const T& data) {
    std::vector<uint8_t> packet(sizeof(T));
    std::memcpy(packet.data(), &data, sizeof(T));
    return packet;
}

inline std::vector<cv::Point2f> reprojection(
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const std::vector<cv::Point3f>& object_points,
    const ISO3& pose_in_camera_cv
) noexcept {
    cv::Mat rvec, R_cv;
    Mat3 R = pose_in_camera_cv.linear();
    cv::eigen2cv(R, R_cv);
    cv::Rodrigues(R_cv, rvec);
    auto t = pose_in_camera_cv.translation();
    const cv::Mat tvec = (cv::Mat_<double>(3, 1) << t.x(), t.y(), t.z());

    std::vector<cv::Point2f> pts_2d;
    pts_2d.reserve(object_points.size());
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, pts_2d);
    return pts_2d;
}
template<typename T>
inline void project_points_jets(
    const std::vector<cv::Point3f>& obj_pts,
    const Eigen::Transform<T, 3, Eigen::Isometry>& pose_cam,
    const cv::Mat& K,
    const cv::Mat& dist_coeffs,
    std::vector<Eigen::Matrix<T, 2, 1>>& img_pts_jet
) {
    if (obj_pts.empty())
        return;
    if (K.empty() || K.rows != 3 || K.cols != 3)
        throw std::runtime_error("Invalid K");
    if (dist_coeffs.empty())
        throw std::runtime_error("Invalid dist_coeffs");

    const Eigen::Matrix<T, 3, 3>& R = pose_cam.linear();
    const Eigen::Matrix<T, 3, 1>& t = pose_cam.translation();

    const T fx = T(K.at<double>(0, 0));
    const T fy = T(K.at<double>(1, 1));
    const T cx = T(K.at<double>(0, 2));
    const T cy = T(K.at<double>(1, 2));

    auto get_dist = [&](int i) -> double {
        return (dist_coeffs.rows == 1) ? dist_coeffs.at<double>(0, i)
                                       : dist_coeffs.at<double>(i, 0);
    };

    const int n_dist = dist_coeffs.rows * dist_coeffs.cols;
    const T k1 = n_dist > 0 ? T(get_dist(0)) : T(0);
    const T k2 = n_dist > 1 ? T(get_dist(1)) : T(0);
    const T p1 = n_dist > 2 ? T(get_dist(2)) : T(0);
    const T p2 = n_dist > 3 ? T(get_dist(3)) : T(0);
    const T k3 = n_dist > 4 ? T(get_dist(4)) : T(0);

    img_pts_jet.clear();
    img_pts_jet.reserve(obj_pts.size());

    for (const auto& pt3: obj_pts) {
        Eigen::Matrix<T, 3, 1> Pw(T(pt3.x), T(pt3.y), T(pt3.z));
        Eigen::Matrix<T, 3, 1> Pc = R * Pw + t;
        T Xc = Pc(0), Yc = Pc(1), Zc = Pc(2);
        T xp = Xc / Zc;
        T yp = Yc / Zc;

        T r2 = xp * xp + yp * yp;
        T r4 = r2 * r2;
        T r6 = r4 * r2;

        T radial = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
        T xd = xp * radial + T(2) * p1 * xp * yp + p2 * (r2 + T(2) * xp * xp);
        T yd = yp * radial + p1 * (r2 + T(2) * yp * yp) + T(2) * p2 * xp * yp;

        T u = fx * xd + cx;
        T v = fy * yd + cy;

        img_pts_jet.emplace_back(u, v);
    }
}
[[nodiscard]] inline double lerp_angle(double a0, double a1, double t) noexcept {
    double d = std::remainder(a1 - a0, 2.0 * M_PI);
    return a0 + t * d;
}
[[nodiscard]] inline Vec3 load_vec3(const YAML::Node& node) {
    auto vec = node.as<std::vector<double>>();
    return Vec3(vec[0], vec[1], vec[2]);
}
[[nodiscard]] inline Mat3 load_mat3(const YAML::Node& node) {
    Mat3 result;

    if (node.IsSequence() && node.size() == 9) {
        // 一维数组
        auto vec = node.as<std::vector<double>>();
        for (int i = 0; i < 9; ++i) {
            result(i / 3, i % 3) = vec[i];
        }
    } else {
        // 二维数组
        auto mat = node.as<std::vector<std::vector<double>>>();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result(i, j) = mat[i][j];
            }
        }
    }

    return result;
}
[[nodiscard]] inline ISO3 load_isometry3(const YAML::Node& node) {
    auto trans = load_vec3(node["t"]);
    auto rot = load_mat3(node["R"]);
    ISO3 result = ISO3::Identity();
    result.translation() = trans;
    result.linear() = rot;
    return result;
}
[[nodiscard]] inline cv::Rect2f load_rect2f(const YAML::Node& node) {
    auto x = node["x"].as<double>();
    auto y = node["y"].as<double>();
    auto w = node["w"].as<double>();
    auto h = node["h"].as<double>();
    return cv::Rect2f(x, y, w, h);
}
inline std::optional<std::string> get_arg(int i, int argc, char* argv[]) {
    if (i < argc) {
        std::cout << "get args " << std::string(argv[i]) << std::endl;
        return std::make_optional(std::string(argv[i]));
    }
    return std::nullopt;
};
template<class LandMark, class ObjectPoint>
inline ISO3 solve_pnp(
    const LandMark& landmarks,
    const ObjectPoint& object_points,
    const cv::Mat& camera_matrix,
    const cv::Mat& distortion_coefficients,
    int flags = cv::SOLVEPNP_ITERATIVE
) {
    ISO3 pose;
    cv::Mat rvec, tvec;
    cv::solvePnP(
        object_points,
        landmarks,
        camera_matrix,
        distortion_coefficients,
        rvec,
        tvec,
        false,
        flags
    );
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Mat3 R_eigen;
    cv::cv2eigen(R_cv, R_eigen);
    pose.linear() = R_eigen;
    Vec3 t_eigen;
    cv::cv2eigen(tvec, t_eigen);
    pose.translation() = t_eigen;
    return pose;
}
template<class Mat>
inline void
fill_constant_accel_noise(Mat& q, int pos_idx, int vel_idx, double noise, double dt) noexcept {
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;

    q(pos_idx, pos_idx) = dt4 * 0.25 * noise;
    q(pos_idx, vel_idx) = dt3 * 0.5 * noise;
    q(vel_idx, pos_idx) = q(pos_idx, vel_idx);
    q(vel_idx, vel_idx) = dt2 * noise;
}
[[nodiscard]] inline double sigmoid(double x) noexcept {
    return x >= 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
}

[[nodiscard]] inline float rect_ioU(const cv::Rect2f& a, const cv::Rect2f& b) noexcept {
    const cv::Rect2f inter = a & b;
    const float inter_area = inter.area();
    const float union_area = a.area() + b.area() - inter_area;
    if (union_area <= 0.f || std::isnan(union_area))
        return 0.f;
    return inter_area / union_area;
}
} // namespace awakening::utils
