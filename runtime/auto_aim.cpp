#include "base/robot.hpp"
#include "common.hpp"
#include "tasks/auto_aim/armor_detect/armor_detector.hpp"
#include "tasks/auto_aim/type.hpp"
#include "utils/logger.hpp"
#include "utils/scheduler/scheduler.hpp"
#include "utils/signal_guard.hpp"
#include <opencv2/highgui.hpp>
#include <optional>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <utility>
#include <vector>
#include <yaml-cpp/yaml.h>

using namespace awakening;

struct CameraFrameTag {};
struct DetectTag {};

struct MainCameraCtx {
    Frame frame_id = Frame::CAMERA;
};
struct LogCtx {
    int camera_count = 0;
    int detect_count = 0;
    int track_count = 0;
    int solve_count = 0;
    std::vector<double> latency_ms;
    void reset() {
        camera_count = 0;
        detect_count = 0;
        track_count = 0;
        solve_count = 0;
        latency_ms.clear();
    }
};
int main() {
    logger::init(spdlog::level::info);
    Scheduler s;

    Camera<MainCameraCtx> camera;

    auto camera_config = YAML::LoadFile("/home/hy/awakening/config/camera.yaml");
    auto auto_aim_config = YAML::LoadFile("/home/hy/awakening/config/auto_aim.yaml");
    auto_aim::ArmorDetector armor_detector(auto_aim_config["armor_detector"]);
    using CamIO = IOPair<CameraFrameTag, CommonFrame>;
    using DetIo = IOPair<DetectTag, auto_aim::Armors>;
    LogCtx log_ctx;
    s.register_task<CamIO, DetIo>("detector", [&](CamIO::second_type&& frame) {
        static std::atomic<int> running_count = 0;
        auto_aim::Armors armors { .timestamp = frame.img_frame.timestamp,
                                  .id = frame.id,
                                  .frame_id = frame.frame_id };
        if (running_count > 5) {
            return std::make_tuple(std::optional<DetIo::second_type>(armors));
        }

        running_count++;
        armors.armors = armor_detector.detect(frame);
        running_count--;
        log_ctx.detect_count++;
        auto& show = frame.img_frame.src_img;
        for (auto& armor: armors.armors) {
            int i = 0;
            for (auto& key_point: armor.key_points.points) {
                if (!key_point.has_value()) {
                    continue;
                }
                cv::circle(show, key_point.value(), 2, cv::Scalar(0, 255, 0), -1);
                cv::putText(
                    show,
                    auto_aim::getStringByArmorKeyPointsIndex(i++),
                    key_point.value(),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 255, 0),
                    1
                );
            }
        }

        cv::imshow("armor_detect", show);
        cv::waitKey(1);
        return std::make_tuple(std::optional<DetIo::second_type>(armors));
    });

    s.register_task<DetIo>("tracker", [&](DetIo::second_type&& io) {
        log_ctx.track_count++;
        auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - io.timestamp
        )
                              .count();
        log_ctx.latency_ms.push_back(latency_ms);
    });
    s.add_rate_source<0>("slover", 1000.0, [&]() { log_ctx.solve_count++; });
    s.add_rate_source<1>("logger", 1.0, [&]() {
        double avg_latency_ms =
            std::accumulate(log_ctx.latency_ms.begin(), log_ctx.latency_ms.end(), 0.0)
            / log_ctx.latency_ms.size();
        AWAKENING_INFO(
            "detect: {} track: {} solve: {} camera: {} avg_latency: {} ms",
            log_ctx.detect_count,
            log_ctx.track_count,
            log_ctx.solve_count,
            log_ctx.camera_count,
            avg_latency_ms
        );
        log_ctx.reset();
    });
    auto cam_source = s.register_source<CamIO>("camera");
    camera.load(camera_config, [&](wust_vl::video::ImageFrame& img_frame) {
        if (img_frame.src_img.empty())
            return;

        log_ctx.camera_count++;
        s.runtime_push_source<CamIO>(cam_source, [f = std::move(img_frame)]() {
            static int current_id = 0;
            CommonFrame frame;
            frame.img_frame = std::move(f);
            frame.detect_color = EnemyColor::BLUE;
            frame.expanded =
                cv::Rect(0, 0, frame.img_frame.src_img.cols, frame.img_frame.src_img.rows);
            frame.offset = cv::Point2f(0, 0);
            frame.id = current_id++;
            frame.frame_id = std::to_underlying(Frame::CAMERA);
            return std::make_tuple(std::optional<CamIO::second_type>(std::move(frame)));
        });
    });

    camera.device_->start();
    s.build();
    s.run();
    SignalGuard::spin(std::chrono::milliseconds(1000));
    s.stop();
    return 0;
}