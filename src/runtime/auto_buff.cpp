#include "angles.h"
#include "ascii_banner.hpp"
#include "daedalus_interface/shm_layout.hpp"
#include "tasks/auto_aim/armor_track/motion_model.hpp"
#include "tasks/auto_buff/rune_detect/rune_detector.hpp"
#include "tasks/base/ballistic_trajectory.hpp"
#include "tasks/base/wheel_odometry.hpp"
#include "utils/drivers/mv_camera.hpp"
#include "utils/io/video_save.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <optional>
#include <string>
#include <utility>
#ifdef USE_ROS2
    #include "_rcl/node.hpp"
    #include "_rcl/tf.hpp"
    #include "_rcl/visual/armor.hpp"
    #include "_rcl/visual/armor_target.hpp"
    #include "sensor_msgs/msg/camera_info.hpp"
    #include "sensor_msgs/msg/image.hpp"
    #include <rclcpp/qos.hpp>
#endif
#include "backward-cpp/backward.hpp"
#include "daedalus_interface/shm_client.hpp"
#include "param_deliver.h"
#include "runtime/config.hpp"
#include "tasks/base/common.hpp"
#include "tasks/base/packet_typedef_receive.hpp"
#include "tasks/base/packet_typedef_send.hpp"
#include "tasks/base/web.hpp"
#include "utils/buffer.hpp"
#include "utils/common/image.hpp"
#include "utils/common/type_common.hpp"
#include "utils/drivers/hik_camera.hpp"
#include "utils/drivers/serial_driver.hpp"
#include "utils/logger.hpp"
#include "utils/runtime_tf.hpp"
#include "utils/scheduler/scheduler.hpp"
#include "utils/semaphore_guard.hpp"
#include "utils/signal_guard.hpp"
#include "utils/utils.hpp"
namespace backward {
static backward::SignalHandling sh;
}
using namespace awakening;

enum class SimpleFrame : int { ODOM, GIMBAL_ODOM, GIMBAL, CAMERA, CAMERA_CV, SHOOT, N };

using SimpleRobotTF = utils::tf::RobotTF<SimpleFrame, static_cast<size_t>(SimpleFrame::N), false>;

std::string SimpleFrame_to_str(int frame) {
    constexpr const char* details[] = { "odom",   "gimbal_odom", "gimbal",
                                        "camera", "camera_cv",   "shoot" };
    return std::string(details[frame]);
}
std::string SimpleFrame_to_str(SimpleFrame frame) {
    return SimpleFrame_to_str(std::to_underlying(frame));
}
struct CameraTag {};
struct SerialTag {};
struct DetectTag {};
struct FrameTag {};

using CameraIO = IOPair<CameraTag, ImageFrame>;
using SerialIO = IOPair<SerialTag, std::vector<uint8_t>>;
using CommonFrameIo = IOPair<FrameTag, CommonFrame>;
// using DetIo = IOPair<DetectTag, std::vector<auto_aim::Armors>>;
struct LogCtx {
    int camera_count = 0;
    int detect_count = 0;
    int track_count = 0;
    int solve_count = 0;
    int serial_count = 0;
    int found_count = 0;
    double latency_ms_total = 0.0;
    void reset() {
        *this = {};
    }
};
struct AutoExposureCfg {
    double target_brightness = 0.0;
    double step_gain = 0.0;
    double decay_step = 0.0;
    double tolerance = 0.0;
    double exposure_min = 0.0;
    double exposure_max = 0.0;
    double control_interval_ms = 0.0;

    explicit AutoExposureCfg(const YAML::Node& c):
        target_brightness(c["target_brightness"].as<double>()),
        step_gain(c["step_gain"].as<double>()),
        decay_step(c["decay_step"].as<double>()),
        tolerance(c["tolerance"].as<double>()),
        exposure_min(c["exposure_min"].as<double>()),
        exposure_max(c["exposure_max"].as<double>()),
        control_interval_ms(c["control_interval_ms"].as<double>()) {}
};
bool is_web_running() {
    static std::atomic<bool> cached { true };
    utils::dt_once(
        [&]() {
            const int ret = std::system("pgrep -x wust_vision_web > /dev/null 2>&1");
            cached = (ret == 0);
        },
        std::chrono::duration<double>(1.0)
    );
    return cached.load();
}
static constexpr auto RECORD_FOLDER_PATH_ARR = utils::concat(ROOT_DIR, "/record/auto_aim");
static constexpr std::string_view RECORD_FOLDER_PATH(RECORD_FOLDER_PATH_ARR.data());

int main(int argc, char** argv) {
    auto start_tp = Clock::now();
    print_banner();
    auto& signal = utils::SignalGuard::instance();
    logger::init(spdlog::level::trace);

    bool debug = false;
    std::string config_path;
    std::string robot_name;
    auto first_arg = utils::get_arg(1, argc, argv);
    if (first_arg) {
        robot_name = first_arg.value();
        config_path = get_robot_config_path(robot_name).value_or(robot_name);
    } else {
        return 1;
    }
    auto second_arg = utils::get_arg(2, argc, argv);
    if (second_arg) {
        debug = second_arg.value() == "true";
    }
    auto config = YAML::LoadFile(config_path);
    Scheduler s;
    EnemyColor enemy_color = enemy_color_from_string(config["enemy_color"].as<std::string>());
    double bullet_speed = config["bullet_speed"].as<double>();
#ifdef USE_ROS2
    rcl::RclcppNode rcl_node("auto_aim");
    rcl::TF rcl_tf(rcl_node);
#endif
    std::unique_ptr<talos::ipc::ShmClient> daedalus_shm_client;
    if (config["use_sim"].as<bool>()) {
        auto client = talos::ipc::ShmClient::connect();
        if (!client) {
            AWAKENING_ERROR("Failed to connect to talos::ipc::ShmClient");
            return 1;
        } else {
            daedalus_shm_client = std::make_unique<talos::ipc::ShmClient>(std::move(*client));
        }
    }
    std::unique_ptr<SerialDriver> serial;
    if (!daedalus_shm_client && config["serial"]["enable"].as<bool>()) {
        serial = std::make_unique<SerialDriver>(config["serial"], s);
    }

    auto camera_config = config["camera"];
    std::unique_ptr<HikCamera> camera;
    utils::SignalGuard::add_callback([&]() {
        if (camera) {
            camera->stop();
        }
    });
    if (!daedalus_shm_client) {
        camera = std::make_unique<HikCamera>(camera_config["hik_camera"], s);
        camera->init();
        if (!camera->running_) {
            return 0;
        }
    }
    std::unique_ptr<VideoSaver> video_saver;
    if (config["record"]["enable"].as<bool>()) {
        video_saver = std::make_unique<VideoSaver>(
            VideoSaver::generate_record_filename(RECORD_FOLDER_PATH.data()),
            VideoSaver::Mode::NonBlocking
        );
    }
    CameraInfo camera_info;
    camera_info.load(camera_config["camera_info"]);
    auto_buff::RuneDetector rune_detector(config["rune_detector"]);
    BulletPickUp bullet_pick_up(config["bullet_pick_up"]);
    LogCtx log_ctx;

    std::pair<double, double> operator_offset = std::make_pair(0, 0);

    WheelOdometry wheel_odometry(config["wheel_odometry"], Clock::now());
    auto tf = SimpleRobotTF::create();
    {
        for (auto [from, to]: std::array {
                 std::pair { SimpleFrame::ODOM, SimpleFrame::GIMBAL_ODOM },
                 std::pair { SimpleFrame::GIMBAL_ODOM, SimpleFrame::GIMBAL },
                 std::pair { SimpleFrame::GIMBAL, SimpleFrame::CAMERA },
                 std::pair { SimpleFrame::GIMBAL, SimpleFrame::SHOOT },
                 std::pair { SimpleFrame::CAMERA, SimpleFrame::CAMERA_CV },
             })
        {
            tf->add_edge(from, to);
        }
        ISO3 cv_in_camera = ISO3::Identity();
        cv_in_camera.linear() = R_CV2PHYSICS;
        tf->push(SimpleFrame::CAMERA, SimpleFrame::CAMERA_CV, Clock::now(), cv_in_camera);
        tf->push(
            SimpleFrame::GIMBAL,
            SimpleFrame::CAMERA,
            Clock::now(),
            utils::load_isometry3(config["tf"]["camera_in_gimbal"])
        );
        tf->push(
            SimpleFrame::GIMBAL,
            SimpleFrame::SHOOT,
            Clock::now(),
            utils::load_isometry3(config["tf"]["shoot_in_gimbal"])
        );
    }
    auto serial_send_to_image_microseconds = config["serial_send_to_image_microseconds"].as<int>();
    if (daedalus_shm_client) {
        auto daedalus_imgs = s.register_source<CameraIO>("daedalus_img");
        s.add_rate_source<>("daedalus_tick", 300.0, [&]() {
            static bool has_camera_info = false;
            if (!has_camera_info) {
                auto daedalus_camera_info = daedalus_shm_client->camera_info();
                has_camera_info = true;
                camera_info.camera_matrix = cv::Mat::eye(3, 3, CV_64F);

                camera_info.camera_matrix.at<double>(0, 0) = daedalus_camera_info.fx;
                camera_info.camera_matrix.at<double>(1, 1) = daedalus_camera_info.fy;
                camera_info.camera_matrix.at<double>(0, 2) = daedalus_camera_info.cx;
                camera_info.camera_matrix.at<double>(1, 2) = daedalus_camera_info.cy;

                camera_info.distortion_coefficients = cv::Mat(1, 5, CV_64F);
                std::memcpy(
                    camera_info.distortion_coefficients.ptr<double>(),
                    daedalus_camera_info.distortion,
                    5 * sizeof(double)
                );
            }
            if (auto frame = daedalus_shm_client->recv_image()) {
                ImageFrame img_frame {
                    .src_img = std::move(frame->image),
                    .format = PixelFormat::RGB,
                    .timestamp = TimePoint(std::chrono::nanoseconds(frame->timestamp_ns)),
                };
                s.runtime_push_source<CameraIO>(daedalus_imgs, [f = std::move(img_frame)]() {
                    return std::make_tuple(std::optional<typename CameraIO::second_type>(std::move(f
                    )));
                });
            }
            if (auto pose = daedalus_shm_client->recv_pose(talos::ipc::PoseIndex::POSE_GIMBAL)) {
                ISO3 gimbal_2_gimbal_odom = ISO3::Identity();
                gimbal_2_gimbal_odom.linear() =
                    Quaternion { pose->qw, pose->qx, pose->qy, pose->qz }.toRotationMatrix();
                tf->push(
                    SimpleFrame::GIMBAL_ODOM,
                    SimpleFrame::GIMBAL,
                    TimePoint(std::chrono::nanoseconds(pose->timestamp_ns)),
                    gimbal_2_gimbal_odom
                );
            }
        });
    }

    if (video_saver) {
        s.register_task<CameraIO>("save_video", [&](CameraIO::second_type&& f) {
            if (!f.src_img.empty()) {
                video_saver->write_frame(f.src_img);
            }
            return std::make_tuple(std::optional<CameraIO::second_type>(std::nullopt));
        });
    }
    s.register_task<CameraIO, CommonFrameIo>("push_common_frame", [&](CameraIO::second_type&& f) {
        static int current_id = 0;
        if (f.src_img.empty()) {
            return std::make_tuple(std::optional<CommonFrameIo::second_type>(std::nullopt));
        }
        log_ctx.camera_count++;
        CommonFrame frame {
            .img_frame = std::move(f),
            .id = current_id++,
            .frame_id = std::to_underlying(SimpleFrame::CAMERA_CV),
        };

        return std::make_tuple(std::optional<CommonFrameIo::second_type>(std::move(frame)));
    });

    if (serial) {
        s.register_task<SerialIO>("receive_serial", [&](SerialIO::second_type&& data) {
            static std::mutex mutex;
            std::lock_guard<std::mutex> lock(mutex);
            auto now = Clock::now();

            log_ctx.serial_count++;
            if (auto robo_opt = ReceiveRobotData::create(data); robo_opt.has_value()) {
                auto robo = robo_opt.value();
                static uint32_t last_pc = -1, delay = 0, last_bullet_count = 0;
                if (robo.time_stamp_pc != last_pc) {
                    last_pc = robo.time_stamp_pc;
                    delay = (std::chrono::duration_cast<std::chrono::microseconds>(now - start_tp)
                                 .count()
                             - robo.time_stamp_pc
                             - (robo.time_stamp_send_micro - robo.time_stamp_receive_micro))
                        / 2.0;
                }

                auto packet_time =
                    now - std::chrono::microseconds(serial_send_to_image_microseconds);
                ISO3 gimbal_2_gimbal_odom = ISO3::Identity();
                gimbal_2_gimbal_odom.linear() = utils::rpy2matrix(Vec3(
                    angles::from_degrees(robo.roll),
                    angles::from_degrees(robo.pitch),
                    angles::from_degrees(robo.yaw)
                ));
                tf->push(
                    SimpleFrame::GIMBAL_ODOM,
                    SimpleFrame::GIMBAL,
                    packet_time,
                    gimbal_2_gimbal_odom
                );
                operator_offset = { angles::from_degrees(robo.operator_yaw_offset),
                                    angles::from_degrees(robo.operator_pitch_offset) };
                tf->push(
                    SimpleFrame::ODOM,
                    SimpleFrame::GIMBAL_ODOM,
                    packet_time,
                    ISO3::Identity()
                );
                enemy_color = EnemyColor(robo.detect_color);
                bullet_speed = robo.bullet_speed;
                robo.update_log(delay);
                if (robo.bullet_count > last_bullet_count) {
                    bullet_pick_up.push_back(Bullet {
                        .fire_time = Clock::now(),
                        .fire_time_shoot_in_odom =
                            tf->pose_a_in_b(SimpleFrame::SHOOT, SimpleFrame::ODOM, Clock::now()),
                        .speed_in_odom = bullet_speed,
                    });
                }
                last_bullet_count = robo.bullet_count;
            }
        });
    }
    if (camera) {
        s.register_task<CommonFrameIo>("auto_exposure", [&](CommonFrameIo::second_type&& frame) {
            static std::mutex mutex;
            std::lock_guard<std::mutex> lock(mutex);
            static std::optional<AutoExposureCfg> auto_exposure_cfg =
                config["auto_exposure"]["enable"].as<bool>()
                ? std::optional<AutoExposureCfg>(std::in_place, config["auto_exposure"])
                : std::nullopt;
            if (auto_exposure_cfg) { // 平均亮度pid
                auto& cfg = auto_exposure_cfg.value();
                utils::dt_once(
                    [&]() {
                        cv::Mat gray;
                        cv::cvtColor(frame.img_frame.src_img, gray, cv::COLOR_BGR2GRAY);
                        double exposure_time = camera->get_ExposureTime();
                        static double last_exposure_time = 0.0;
                        const double diff = cv::mean(gray)[0] - cfg.target_brightness;
                        if (std::fabs(diff) > cfg.tolerance && exposure_time > 0.0) {
                            exposure_time -= diff * cfg.step_gain;
                        } else {
                            exposure_time -= cfg.decay_step;
                        }
                        exposure_time =
                            std::clamp(exposure_time, cfg.exposure_min, cfg.exposure_max);
                        if (std::abs(exposure_time - last_exposure_time) > 10) {
                            camera->set_ExposureTime(exposure_time);
                            last_exposure_time = exposure_time;
                        }
                    },
                    std::chrono::milliseconds((int)cfg.control_interval_ms)
                );
            }
        });
    }

    s.register_task<CommonFrameIo>("detector", [&](CommonFrameIo::second_type&& frame) {
        static std::unique_ptr<std::counting_semaphore<>> detector_sem;
        if (!detector_sem) {
            detector_sem =
                std::make_unique<std::counting_semaphore<>>(config["max_infer_num"].as<int>());
        }
        auto rune_detection = rune_detector.detect(
            frame,
            cv::Rect(0, 0, frame.img_frame.src_img.cols, frame.img_frame.src_img.rows),
            enemy_color
        );
        auto& img = frame.img_frame.src_img;
        rune_detection.draw(img);
        cv::imshow("Rune Detection", img);
        cv::waitKey(1);
    });

    s.add_rate_source<>("logger", 1.0, [&]() {
        double avg_latency_ms = log_ctx.latency_ms_total / log_ctx.track_count;
        AWAKENING_INFO(
            "detect: {} track: {} found: {} solve: {} serial: {} camera: {} avg_latency: {:.3} ms",
            log_ctx.detect_count,
            log_ctx.track_count,
            log_ctx.found_count,
            log_ctx.solve_count,
            log_ctx.serial_count,
            log_ctx.camera_count,
            avg_latency_ms
        );

        log_ctx.reset();
    });

    if (camera) {
        camera->start<CameraTag>("hik");
    }
    if (serial) {
        serial->start<SerialTag>("serial");
    }
    s.build();
    s.run();

#ifdef USE_ROS2
    std::thread([&]() { rcl_node.spin(); }).detach();
#endif
    utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    s.stop();
#ifdef USE_ROS2
    rcl_node.shutdown();
#endif
    for (int i = 0; i < 10; ++i) {
        AWAKENING_CRITICAL("改了东西记得同步其他有关的exe的src");
    }
    return 0;
}
