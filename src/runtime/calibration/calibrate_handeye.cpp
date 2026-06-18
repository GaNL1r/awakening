#include "calibration_common.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

using namespace awakening::calibration;

const std::string keys =
    "{help h usage ? |                         | print help}"
    "{config-path c  | config/calibration.yaml | calibration yaml path}"
    "{@input-folder  | assets/calibration      | folder containing 1.jpg + 1.txt, ...}";

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
    const auto quaternion_extension = yaml_string(yaml, "quaternion_extension", ".txt");
    const auto show = yaml_bool(yaml, "show_detection", true);
    const auto spacing =
        static_cast<float>(yaml_double(yaml, "center_distance_mm", 40.0));
    const auto object_points = make_object_points(yaml, size, spacing);
    const auto R_gimbal2imubody_data =
        yaml["R_gimbal2imubody"].as<std::vector<double>>();
    const auto R_gimbal2imubody = eigen_row_major_3x3(R_gimbal2imubody_data);
    const auto camera_matrix = read_camera_matrix(yaml);
    const auto distortion_coeffs = read_distortion_coeffs(yaml);

    std::vector<cv::Mat> R_gimbal2world_list;
    std::vector<cv::Mat> t_gimbal2world_list;
    std::vector<cv::Mat> R_target2camera_list;
    std::vector<cv::Mat> tvecs;

    for (int i = 1;; ++i) {
        const auto image_path = indexed_path(input_folder, i, image_extension);
        const auto q_path = indexed_path(input_folder, i, quaternion_extension);
        const auto image = cv::imread(image_path);
        if (image.empty()) {
            break;
        }

        const auto q = read_quaternion(q_path);
        const Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
        const Eigen::Matrix3d R_gimbal2world =
            R_gimbal2imubody.transpose() * R_imubody2imuabs * R_gimbal2imubody;

        std::vector<cv::Point2f> points;
        const bool success = find_pattern(image, size, board_type, points);
        show_detection(image, size, points, success, "calibrate_handeye", show);
        std::cout << "[" << (success ? "success" : "failure") << "] " << image_path << "\n";
        if (!success) {
            continue;
        }

        cv::Mat rvec;
        cv::Mat tvec;
        if (!cv::solvePnP(
                object_points,
                points,
                camera_matrix,
                distortion_coeffs,
                rvec,
                tvec,
                false,
                cv::SOLVEPNP_IPPE
            ))
        {
            std::cerr << "[failure] solvePnP " << image_path << "\n";
            continue;
        }

        cv::Mat R_gimbal2world_cv;
        cv::eigen2cv(R_gimbal2world, R_gimbal2world_cv);
        R_gimbal2world_list.emplace_back(R_gimbal2world_cv);
        t_gimbal2world_list.emplace_back((cv::Mat_<double>(3, 1) << 0, 0, 0));
        cv::Mat R_target2camera;
        cv::Rodrigues(rvec, R_target2camera);
        R_target2camera_list.emplace_back(R_target2camera);
        tvecs.emplace_back(tvec);
    }

    if (R_target2camera_list.size() < 3) {
        std::cerr << "need at least 3 valid handeye samples, got " << R_target2camera_list.size()
                  << "\n";
        return 1;
    }

    cv::Mat R_camera2gimbal;
    cv::Mat t_camera2gimbal;
    cv::calibrateHandEye(
        R_gimbal2world_list,
        t_gimbal2world_list,
        R_target2camera_list,
        tvecs,
        R_camera2gimbal,
        t_camera2gimbal
    );
    t_camera2gimbal /= 1e3;

    Eigen::Matrix3d R_camera2gimbal_eigen;
    cv::cv2eigen(R_camera2gimbal, R_camera2gimbal_eigen);
    const Eigen::Matrix3d R_gimbal2ideal {
        { 0, -1, 0 },
        { 0, 0, -1 },
        { 1, 0, 0 },
    };
    const Eigen::Vector3d camera_ypr =
        euler_degrees(R_gimbal2ideal * R_camera2gimbal_eigen, 1, 0, 2);

    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "R_gimbal2imubody";
    out << YAML::Value << YAML::Flow << R_gimbal2imubody_data;
    out << YAML::Comment(
        "camera ideal offset deg yaw=" + std::to_string(camera_ypr[0])
        + " pitch=" + std::to_string(camera_ypr[1])
        + " roll=" + std::to_string(camera_ypr[2])
    );
    out << YAML::Key << "R_camera2gimbal";
    out << YAML::Value << YAML::Flow << mat_to_vector(R_camera2gimbal);
    out << YAML::Key << "t_camera2gimbal";
    out << YAML::Value << YAML::Flow << mat_to_vector(t_camera2gimbal);
    out << YAML::Key << "camera_in_gimbal";
    out << YAML::Value;
    emit_isometry(out, R_camera2gimbal, t_camera2gimbal);
    out << YAML::EndMap;

    print_yaml_result(out);
    return 0;
}
