#include "base/robot.hpp"
#include "common.hpp"
#include "tasks/auto_aim/type.hpp"
#include "utils/scheduler/scheduler.hpp"
#include <opencv2/highgui.hpp>
#include <optional>
#include <utility>
#include <yaml-cpp/yaml.h>
using namespace awakening;

struct CameraFrameTag {};
struct DetectTag {};

struct MainCameraCtx {
    Frame frame_id = Frame::CAMERA;
};
struct RateCount {
    int cam = 0;
    int show = 0;
    int detect = 0;
    int track = 0;
    int solve = 0;
    void reset() {
        cam = 0;
        show = 0;
        detect = 0;
        track = 0;
        solve = 0;
    }
};
int main() {
    Scheduler s;

    Camera<MainCameraCtx> camera;
    auto camera_config = YAML::LoadFile("/home/hy/awakening/config/camera.yaml");

    using CamIO = IOPair<CameraFrameTag, CommonFrame>;
    using DetIo = IOPair<DetectTag, int>;
    RateCount rate_count;
    s.register_task<CamIO>("show", [&](CamIO::second_type&& frame) {
        if (!frame.img_frame.src_img.empty()) {
            cv::imshow("camera", frame.img_frame.src_img);
            cv::waitKey(1);
        }

        rate_count.show++;
    });

    s.register_task<CamIO, DetIo>("detector", [&](CamIO::second_type&& frame) {
        static std::atomic<int> running_count = 0;
        if (running_count > 5) {
            return std::make_tuple(std::optional<DetIo::second_type>(std::nullopt));
        }
        running_count++;

        running_count--;
        rate_count.detect++;
        return std::make_tuple(std::optional<DetIo::second_type>(1));
    });

    s.register_task<DetIo>("tracker", [&](DetIo::second_type&& io) { rate_count.track++; });
    s.add_rate_source<0>("slover", 1000.0, [&]() { rate_count.solve++; });
    s.add_rate_source<1>("logger", 1.0, [&]() {
        std::cout << "detect: " << rate_count.detect << " track: " << rate_count.track
                  << " solve: " << rate_count.solve << " camera: " << rate_count.cam << std::endl;
        rate_count.reset();
    });
    auto cam_source = s.register_source<CamIO>("camera");
    camera.load(camera_config, [&](wust_vl::video::ImageFrame& img_frame) {
        if (img_frame.src_img.empty())
            return;

        rate_count.cam++;
        s.runtime_push_source<CamIO>(cam_source, [f = std::move(img_frame)]() {
            CommonFrame frame;
            frame.img_frame = std::move(f);
            frame.detect_color = EnemyColor::BLUE;
            frame.expanded =
                cv::Rect(0, 0, frame.img_frame.src_img.cols, frame.img_frame.src_img.rows);
            frame.offset = cv::Point2f(0, 0);
            return std::make_tuple(std::optional<CamIO::second_type>(frame));
        });
    });

    camera.device_->start();
    s.build();
    s.run();
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    return 0;
}