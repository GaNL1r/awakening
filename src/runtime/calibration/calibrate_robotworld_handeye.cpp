#include "calibration_common.hpp"

#include <cmath>
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

    std::vector<cv::Mat> R_world2gimbal_list;
    std::vector<cv::Mat> t_world2gimbal_list;
    std::vector<cv::Mat> R_board2camera_list;
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
        const Eigen::Matrix3d R_world2gimbal = R_gimbal2world.transpose();

        std::vector<cv::Point2f> points;
        const bool success = find_pattern(image, size, board_type, points);
        show_detection(image, size, points, success, "calibrate_robotworld_handeye", show);
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

        cv::Mat R_world2gimbal_cv;
        cv::eigen2cv(R_world2gimbal, R_world2gimbal_cv);
        R_world2gimbal_list.emplace_back(R_world2gimbal_cv);
        t_world2gimbal_list.emplace_back((cv::Mat_<double>(3, 1) << 0, 0, 0));
        cv::Mat R_board2camera;
        cv::Rodrigues(rvec, R_board2camera);
        R_board2camera_list.emplace_back(R_board2camera);
        tvecs.emplace_back(tvec);
    }

    if (R_board2camera_list.size() < 3) {
        std::cerr << "need at least 3 valid robot-world handeye samples, got "
                  << R_board2camera_list.size() << "\n";
        return 1;
    }

    cv::Mat R_world2board;
    cv::Mat t_world2board;
    cv::Mat R_gimbal2camera;
    cv::Mat t_gimbal2camera;
    cv::calibrateRobotWorldHandEye(
        R_board2camera_list,
        tvecs,
        R_world2gimbal_list,
        t_world2gimbal_list,
        R_world2board,
        t_world2board,
        R_gimbal2camera,
        t_gimbal2camera
    );
    t_gimbal2camera /= 1e3;
    t_world2board /= 1e3;

    cv::Mat R_camera2gimbal;
    cv::Mat R_board2world;
    cv::transpose(R_gimbal2camera, R_camera2gimbal);
    cv::transpose(R_world2board, R_board2world);
    cv::Mat t_camera2gimbal = -R_camera2gimbal * t_gimbal2camera;
    cv::Mat t_board2world = -R_board2world * t_world2board;

    Eigen::Matrix3d R_camera2gimbal_eigen;
    Eigen::Matrix3d R_board2world_eigen;
    cv::cv2eigen(R_camera2gimbal, R_camera2gimbal_eigen);
    cv::cv2eigen(R_board2world, R_board2world_eigen);
    const Eigen::Matrix3d R_gimbal2ideal {
        { 0, -1, 0 },
        { 0, 0, -1 },
        { 1, 0, 0 },
    };
    const Eigen::Vector3d camera_ypr =
        euler_degrees(R_gimbal2ideal * R_camera2gimbal_eigen, 1, 0, 2);
    const Eigen::Vector3d board_ypr = euler_degrees(R_board2world_eigen, 2, 1, 0);
    const double board_x = t_board2world.at<double>(0);
    const double board_y = t_board2world.at<double>(1);
    const double board_distance = std::hypot(board_x, board_y);

    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "R_gimbal2imubody";
    out << YAML::Value << YAML::Flow << R_gimbal2imubody_data;
    out << YAML::Comment(
        "camera ideal offset deg yaw=" + std::to_string(camera_ypr[0])
        + " pitch=" + std::to_string(camera_ypr[1])
        + " roll=" + std::to_string(camera_ypr[2])
    );
    out << YAML::Comment("board horizontal distance m: " + std::to_string(board_distance));
    out << YAML::Comment(
        "board world offset deg yaw=" + std::to_string(board_ypr[0])
        + " pitch=" + std::to_string(board_ypr[1])
        + " roll=" + std::to_string(board_ypr[2])
    );
    out << YAML::Key << "R_camera2gimbal";
    out << YAML::Value << YAML::Flow << mat_to_vector(R_camera2gimbal);
    out << YAML::Key << "t_camera2gimbal";
    out << YAML::Value << YAML::Flow << mat_to_vector(t_camera2gimbal);
    out << YAML::Key << "camera_in_gimbal";
    out << YAML::Value;
    emit_isometry(out, R_camera2gimbal, t_camera2gimbal);
    out << YAML::Key << "R_board2world";
    out << YAML::Value << YAML::Flow << mat_to_vector(R_board2world);
    out << YAML::Key << "t_board2world";
    out << YAML::Value << YAML::Flow << mat_to_vector(t_board2world);
    out << YAML::EndMap;

    print_yaml_result(out);
    return 0;
}
