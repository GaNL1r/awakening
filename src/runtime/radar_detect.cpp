#include "_rcl/node.hpp"
#include "_rcl/tf.hpp"
#include "ascii_banner.hpp"
#include "tasks/base/common.hpp"
#include "tasks/radar_detect/car_pool.hpp"
#include "tasks/radar_detect/detector.hpp"
#include "tasks/radar_detect/lidar_location.hpp"
#include "tasks/radar_detect/pixel_to_world.hpp"
#include "tasks/radar_detect/tracker.hpp"
#include "tasks/radar_detect/type.hpp"
#include "utils/buffer.hpp"
#include "utils/common/image.hpp"
#include "utils/common/type_common.hpp"
#include "utils/drivers/hik_camera.hpp"
#include "utils/drivers/video_player.hpp"
#include "utils/logger.hpp"
#include "utils/runtime_tf.hpp"
#include "utils/semaphore_guard.hpp"
#include "utils/signal_guard.hpp"
#include "utils/utils.hpp"
#include <iostream>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <optional>
#include <rclcpp/qos.hpp>
#include <rclcpp/utilities.hpp>
#include <sensor_msgs/msg/detail/point_cloud2__struct.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <string>
#include <utility>
#include <vector>
#include <yaml-cpp/node/parse.h>
using namespace awakening;
struct CameraTag {};
struct SerialTag {};
struct DetectTag {};
struct FrameTag {};
using CameraIO = IOPair<CameraTag, ImageFrame>;
using SerialIO = IOPair<SerialTag, std::vector<uint8_t>>;
using CommonFrameIo = IOPair<FrameTag, CommonFrame>;
using DetIo = IOPair<DetectTag, std::vector<radar_detect::Cars>>;
enum class RadarFrame : int { TARGET_MAP, CAMERA, CAMERA_CV, N };
using RadarTF = utils::tf::RobotTF<RadarFrame, static_cast<size_t>(RadarFrame::N), false>;
std::string RadarFrame_to_str(int f) {
    constexpr const char* details[] = { "target_map", "camera_cv" };
    return std::string(details[f]);
}
std::string RadarFrame_to_str(RadarFrame f) {
    return RadarFrame_to_str(std::to_underlying(f));
}
int main(int argc, char** argv) {
    print_banner();
    auto& signal = utils::SignalGuard::instance();
    logger::init(spdlog::level::trace);
    auto get_arg = [&](int i) -> std::optional<std::string> {
        if (i < argc) {
            AWAKENING_INFO("get args {} ", std::string(argv[i]));
            return std::make_optional(std::string(argv[i]));
        }
        return std::nullopt;
    };
    bool debug = false;
    std::string config_path;
    auto first_arg = get_arg(1);
    if (first_arg) {
        config_path = first_arg.value();
    } else {
        return 1;
    }
    auto config = YAML::LoadFile(config_path);
    auto camera_config = config["camera"];
    CameraInfo camera_info;
    camera_info.load(camera_config["camera_info"]);
    Scheduler s;
    rcl::RclcppNode rcl_node("radar_detect");
    rcl::TF rcl_tf(rcl_node);
    RadarTF tf;
    {
        tf.add_edge(RadarFrame::TARGET_MAP, RadarFrame::CAMERA);
        tf.add_edge(RadarFrame::CAMERA, RadarFrame::CAMERA_CV);
        ISO3 camera_cv_in_camera = ISO3::Identity();
        camera_cv_in_camera.linear() = R_CV2PHYSICS;
        tf.push(RadarFrame::CAMERA, RadarFrame::CAMERA_CV, Clock::now(), camera_cv_in_camera);
        ISO3 camera_in_target_map = utils::load_isometry3(config["tf"]["camera_in_target_map"]);

        tf.push(RadarFrame::TARGET_MAP, RadarFrame::CAMERA, Clock::now(), camera_in_target_map);
    }
    std::unique_ptr<VideoPlayer> video;
    std::unique_ptr<HikCamera> camera;
    utils::SignalGuard::add_callback([&]() {
        if (camera) {
            camera->stop();
        }
    });
    std::string camera_type = camera_config["type"].as<std::string>();
    camera_type = utils::to_upper(camera_type);
    if (camera_type == "VIDEO") {
        video = std::make_unique<VideoPlayer>(camera_config["video"], s);
    } else {
        camera = std::make_unique<HikCamera>(camera_config["hik_camera"], s);
    }
    if (camera) {
        camera->init();
        if (!camera->running_) {
            return 0;
        }
    }

    utils::OrderedQueue<radar_detect::Cars> cars_queue;
    radar_detect::SelfColor self_color =
        radar_detect::SelfColor_from_str(config["self_color"].as<std::string>());
    radar_detect::Detector detector(config["detector"]);
    radar_detect::Tracker tracker(config["tracker"]);
    radar_detect::RMUC2026Map map(config["map"], self_color);
    utils::SWMR<std::unordered_map<int, radar_detect::TheOnlyCar>> fin_cars;
    radar_detect::CarPool car_pool(config["car_pool"], map);
    ISO3 camera_cv_in_target_map =
        tf.pose_a_in_b(RadarFrame::CAMERA_CV, RadarFrame::TARGET_MAP, Clock::now());
    radar_detect::PixelToWorld pixel_to_world(
        config["pixel_to_world"],
        camera_cv_in_target_map,
        camera_info
    );
    radar_detect::RadarDebugCtx debug_ctx;
    debug_ctx.map = map.image.clone();
    auto outpost_bbox = utils::load_rect2f(config["outpost_bbox"]);
    s.register_task<CameraIO, CommonFrameIo>("push_common_frame", [&](CameraIO::second_type&& f) {
        static int current_id = 0;
        int x = 0;
        int y = 0;
        int w = f.src_img.cols;
        int h = f.src_img.rows;
        CommonFrame frame {
            .img_frame = std::move(f),
            .id = current_id++,
            .frame_id = 0,
            .expanded = cv::Rect(x, y, w, h),
            .offset = cv::Point2f(x, y),
        };

        return std::make_tuple(std::optional<CommonFrameIo::second_type>(std::move(frame)));
    });
    s.register_task<CommonFrameIo, DetIo>("detector", [&](CommonFrameIo::second_type&& frame) {
        static std::unique_ptr<std::counting_semaphore<>> detector_sem;
        if (!detector_sem) {
            detector_sem =
                std::make_unique<std::counting_semaphore<>>(config["max_infer_num"].as<int>());
        }
        static std::vector<radar_detect::Armor> outpost_armors;
        auto img = frame.img_frame.src_img;

        radar_detect::Cars cars { .t = frame.img_frame.timestamp, .id = frame.id };
        {
            bool got = detector_sem->try_acquire();
            utils::SemaphoreGuard guard(*detector_sem, got);
            if (got) {
                auto start = Clock::now();
                cars.cars = detector.detect(frame);
                utils::dt_once(
                    [&]() {
                        CommonFrame f = frame;
                        f.expanded = outpost_bbox;
                        f.offset = outpost_bbox.tl();
                        outpost_armors = detector.detect_armors(f);
                    },
                    std::chrono::duration<double>(1.0)
                );
                auto end = Clock::now();
                std::cout
                    << "cost : "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                    << " ms" << std::endl;
            }
        }

        std::vector<radar_detect::Car> valid_cars;
        for (auto& car: cars.cars) {
            auto p_opt = pixel_to_world.pixel_to_world(car.get_key_point());
            if (p_opt) {
                car.point_in_uwb = p_opt.value();
                if (self_color == radar_detect::SelfColor::BLUE) {
                    car.point_in_uwb.x() = radar_detect::FIELD_LONGTH - car.point_in_uwb.x();
                    car.point_in_uwb.y() = radar_detect::FIELD_WIDTH - car.point_in_uwb.y();
                }
                // std::cout << car.point_in_uwb.transpose() << std::endl;
                valid_cars.push_back(car);
            }
        }
        debug_ctx.cars.set(cars.cars);
        debug_ctx.outpost.set(outpost_armors);
        cars_queue.enqueue(cars);
        auto batch_cars = cars_queue.dequeue_batch();
        debug_ctx.img_frame.set(frame.img_frame);
        return std::make_tuple(std::optional<DetIo::second_type>(std::move(batch_cars)));
    });
    s.register_task<DetIo>("tracker", [&](DetIo::second_type&& io) {
        for (const auto& cars: io) {
            tracker.update(cars.cars, cars.t);
            car_pool.update(tracker.get_targets());
        }
        auto _fin_cars = car_pool.get_fin_cars();
        fin_cars.write(_fin_cars);
    });
    cv::namedWindow("Video Frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video Frame", 800, 600);
    cv::namedWindow("Map", cv::WINDOW_NORMAL);
    cv::resizeWindow("Map", 800, 600);
    s.add_rate_source<>("debug", 10.0, [&]() {
        auto image = debug_ctx.img_frame.get().src_img;
        if (image.empty()) {
            return;
        }
        auto cars = debug_ctx.cars.get();
        auto outpost = debug_ctx.outpost.get();
        auto map = debug_ctx.map.clone();
        if (map.image.empty()) {
            return;
        }
        cv::resize(map.image, map.image, cv::Size(map.image.cols * 3.0, map.image.rows * 3.0));
        auto _fin_cars = fin_cars.read();
        for (const auto& car: cars) {
            car.draw(image);
        }
        for (const auto& o: outpost) {
            o.draw(image);
        }
        for (const auto& car: _fin_cars) {
            auto img_point = radar_detect::uwb_to_image(map, car.second.uwb_state.state.pos().head<2>());
            cv::putText(
                map.image,
                radar_detect::CarClass_to_str(car.second.car_class),
                img_point,
                cv::FONT_HERSHEY_SIMPLEX,
                5.0,
                cv::Scalar(0, 255, 0),
                2
            );
            cv::circle(map.image, img_point, 10, cv::Scalar(0, 255, 0), -1);
        }
        cv::imshow("Map", map.image);
        cv::imshow("Video Frame", image);
        cv::waitKey(1);
    });

    if (camera) {
        camera->start<CameraTag>("hik");
    }
    if (video) {
        video->start<CameraTag>("video");
    }
    s.build();
    s.run();
    std::thread([&]() { rcl_node.spin(); }).detach();

    utils::SignalGuard::spin(std::chrono::milliseconds(1000));

    s.stop();
    rcl_node.shutdown();
    cv::destroyAllWindows();

    for (int i = 0; i < 10; ++i) {
        AWAKENING_CRITICAL("改了东西记得同步其他有关的exe的src");
    }
    return 0;
}