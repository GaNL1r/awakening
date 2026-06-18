#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace awakening::calibration {

inline bool yaml_bool(const YAML::Node& node, const std::string& key, bool fallback) {
    return node[key] ? node[key].as<bool>() : fallback;
}

inline std::string yaml_string(
    const YAML::Node& node,
    const std::string& key,
    const std::string& fallback
) {
    return node[key] ? node[key].as<std::string>() : fallback;
}

inline double yaml_double(const YAML::Node& node, const std::string& key, double fallback) {
    return node[key] ? node[key].as<double>() : fallback;
}

inline cv::Size pattern_size(const YAML::Node& yaml) {
    return { yaml["pattern_cols"].as<int>(), yaml["pattern_rows"].as<int>() };
}

inline std::vector<cv::Point3f> object_points_xy(
    const cv::Size& size,
    const float spacing
) {
    std::vector<cv::Point3f> points;
    points.reserve(size.width * size.height);
    for (int r = 0; r < size.height; ++r) {
        for (int c = 0; c < size.width; ++c) {
            points.push_back({ c * spacing, r * spacing, 0.0F });
        }
    }
    return points;
}

inline std::vector<cv::Point3f> object_points_centered_yz(
    const cv::Size& size,
    const float spacing
) {
    std::vector<cv::Point3f> points;
    points.reserve(size.width * size.height);
    for (int r = 0; r < size.height; ++r) {
        for (int c = 0; c < size.width; ++c) {
            const float y = (-c + 0.5F * static_cast<float>(size.width)) * spacing;
            const float z = (-r + 0.5F * static_cast<float>(size.height)) * spacing;
            points.push_back({ 0.0F, y, z });
        }
    }
    return points;
}

inline std::vector<cv::Point3f> make_object_points(
    const YAML::Node& yaml,
    const cv::Size& size,
    const float spacing
) {
    const auto frame = yaml_string(yaml, "object_frame", "xy");
    if (frame == "xy") {
        return object_points_xy(size, spacing);
    }
    if (frame == "centered_yz") {
        return object_points_centered_yz(size, spacing);
    }
    throw std::runtime_error("unsupported object_frame: " + frame);
}

inline bool find_pattern(
    const cv::Mat& image,
    const cv::Size& size,
    const std::string& board_type,
    std::vector<cv::Point2f>& points
) {
    if (board_type == "circles" || board_type == "circle_grid") {
        return cv::findCirclesGrid(image, size, points, cv::CALIB_CB_SYMMETRIC_GRID);
    }
    if (board_type == "chessboard") {
        const int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
        const bool ok = cv::findChessboardCorners(image, size, points, flags);
        if (!ok) {
            return false;
        }
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }
        cv::cornerSubPix(
            gray,
            points,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01)
        );
        return true;
    }
    throw std::runtime_error("unsupported board_type: " + board_type);
}

inline std::string indexed_path(
    const std::string& folder,
    int index,
    const std::string& extension
) {
    std::ostringstream oss;
    oss << folder << "/" << index << extension;
    return oss.str();
}

inline Eigen::Quaterniond read_quaternion(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open quaternion file: " + path);
    }
    double w = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    file >> w >> x >> y >> z;
    if (!file) {
        throw std::runtime_error("invalid quaternion file: " + path);
    }
    Eigen::Quaterniond q(w, x, y, z);
    q.normalize();
    return q;
}

inline cv::Mat mat_from_vector(const std::vector<double>& data, int rows, int cols) {
    if (static_cast<int>(data.size()) != rows * cols) {
        throw std::runtime_error("matrix data size mismatch");
    }
    cv::Mat mat(rows, cols, CV_64F);
    std::memcpy(mat.ptr<double>(), data.data(), data.size() * sizeof(double));
    return mat;
}

inline cv::Mat read_camera_matrix(const YAML::Node& yaml) {
    if (yaml["camera_matrix"]) {
        return mat_from_vector(yaml["camera_matrix"].as<std::vector<double>>(), 3, 3);
    }
    if (yaml["camera_info"] && yaml["camera_info"]["camera_matrix"]) {
        return mat_from_vector(
            yaml["camera_info"]["camera_matrix"]["data"].as<std::vector<double>>(),
            3,
            3
        );
    }
    throw std::runtime_error("missing camera_matrix");
}

inline cv::Mat read_distortion_coeffs(const YAML::Node& yaml) {
    if (yaml["distort_coeffs"]) {
        const auto data = yaml["distort_coeffs"].as<std::vector<double>>();
        return mat_from_vector(data, 1, static_cast<int>(data.size()));
    }
    if (yaml["camera_info"] && yaml["camera_info"]["distortion_coefficients"]) {
        const auto data =
            yaml["camera_info"]["distortion_coefficients"]["data"].as<std::vector<double>>();
        return mat_from_vector(data, 1, static_cast<int>(data.size()));
    }
    throw std::runtime_error("missing distort_coeffs");
}

inline std::vector<double> mat_to_vector(const cv::Mat& mat) {
    cv::Mat continuous = mat.isContinuous() ? mat : mat.clone();
    return {
        continuous.ptr<double>(),
        continuous.ptr<double>() + continuous.total() * continuous.channels()
    };
}

inline Eigen::Matrix3d eigen_row_major_3x3(const std::vector<double>& data) {
    if (data.size() != 9) {
        throw std::runtime_error("expected 9 values for 3x3 matrix");
    }
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> mat(data.data());
    return mat;
}

inline Eigen::Vector3d euler_degrees(
    const Eigen::Matrix3d& R,
    int a0,
    int a1,
    int a2
) {
    return R.eulerAngles(a0, a1, a2) * 180.0 / CV_PI;
}

inline void show_detection(
    const cv::Mat& image,
    const cv::Size& size,
    const std::vector<cv::Point2f>& points,
    bool success,
    const std::string& title,
    bool show
) {
    if (!show) {
        return;
    }
    cv::Mat drawing = image.clone();
    cv::drawChessboardCorners(drawing, size, points, success);
    if (drawing.cols > 1280 || drawing.rows > 900) {
        const double scale = std::min(1280.0 / drawing.cols, 900.0 / drawing.rows);
        cv::resize(drawing, drawing, {}, scale, scale);
    }
    cv::imshow(title, drawing);
    cv::waitKey(0);
}

inline void print_yaml_result(const YAML::Emitter& emitter) {
    std::cout << "\n" << emitter.c_str() << "\n";
}

inline void emit_isometry(YAML::Emitter& out, const cv::Mat& R, const cv::Mat& t) {
    out << YAML::BeginMap;
    out << YAML::Key << "t";
    out << YAML::Value << YAML::Flow << mat_to_vector(t);
    out << YAML::Key << "R";
    out << YAML::Value << YAML::Flow << mat_to_vector(R);
    out << YAML::EndMap;
}

} // namespace awakening::calibration
