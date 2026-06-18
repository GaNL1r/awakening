#include "calibration_common.hpp"

#include <cfloat>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

using namespace awakening::calibration;

const std::string keys =
    "{help h usage ? |                         | print help}"
    "{config-path c  | config/calibration.yaml | calibration yaml path}"
    "{@input-folder  | assets/calibration      | folder containing 1.jpg, 2.jpg, ...}";

int main(int argc, char** argv) {
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }

    const auto input_folder = cli.get<std::string>(0);
    const auto config_path = cli.get<std::string>("config-path");
    const auto yaml = YAML::LoadFile(config_path);

    const cv::Size size = pattern_size(yaml);
    const auto board_type = yaml_string(yaml, "board_type", "circles");
    const auto image_extension = yaml_string(yaml, "image_extension", ".jpg");
    const auto show = yaml_bool(yaml, "show_detection", true);
    const auto spacing =
        static_cast<float>(yaml_double(yaml, "center_distance_mm", 40.0));
    const auto object_points_one = make_object_points(yaml, size, spacing);

    cv::Size image_size;
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;

    for (int i = 1;; ++i) {
        const auto image_path = indexed_path(input_folder, i, image_extension);
        const auto image = cv::imread(image_path);
        if (image.empty()) {
            break;
        }

        image_size = image.size();
        std::vector<cv::Point2f> points;
        const bool success = find_pattern(image, size, board_type, points);
        show_detection(image, size, points, success, "calibrate_camera", show);
        std::cout << "[" << (success ? "success" : "failure") << "] " << image_path << "\n";

        if (!success) {
            continue;
        }
        image_points.emplace_back(std::move(points));
        object_points.emplace_back(object_points_one);
    }

    if (image_points.size() < 3) {
        std::cerr << "need at least 3 valid calibration images, got " << image_points.size()
                  << "\n";
        return 1;
    }

    cv::Mat camera_matrix;
    cv::Mat distortion_coeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    const auto criteria = cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
        100,
        DBL_EPSILON
    );
    cv::calibrateCamera(
        object_points,
        image_points,
        image_size,
        camera_matrix,
        distortion_coeffs,
        rvecs,
        tvecs,
        cv::CALIB_FIX_K3,
        criteria
    );

    double error_sum = 0.0;
    size_t total_points = 0;
    for (size_t i = 0; i < object_points.size(); ++i) {
        std::vector<cv::Point2f> reprojected;
        cv::projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            distortion_coeffs,
            reprojected
        );
        total_points += reprojected.size();
        for (size_t j = 0; j < reprojected.size(); ++j) {
            error_sum += cv::norm(image_points[i][j] - reprojected[j]);
        }
    }
    const double reprojection_error = error_sum / static_cast<double>(total_points);

    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Comment("reprojection_error_px: " + std::to_string(reprojection_error));
    out << YAML::Key << "camera_matrix";
    out << YAML::Value << YAML::Flow << mat_to_vector(camera_matrix);
    out << YAML::Key << "distort_coeffs";
    out << YAML::Value << YAML::Flow << mat_to_vector(distortion_coeffs);
    out << YAML::Key << "camera_info";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "camera_matrix";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "rows" << YAML::Value << 3;
    out << YAML::Key << "cols" << YAML::Value << 3;
    out << YAML::Key << "data" << YAML::Value << YAML::Flow << mat_to_vector(camera_matrix);
    out << YAML::EndMap;
    out << YAML::Key << "distortion_model" << YAML::Value << "plumb_bob";
    out << YAML::Key << "distortion_coefficients";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "rows" << YAML::Value << 1;
    out << YAML::Key << "cols" << YAML::Value
        << static_cast<int>(distortion_coeffs.total());
    out << YAML::Key << "data" << YAML::Value << YAML::Flow << mat_to_vector(distortion_coeffs);
    out << YAML::EndMap;
    out << YAML::EndMap;
    out << YAML::EndMap;

    print_yaml_result(out);
    return 0;
}
