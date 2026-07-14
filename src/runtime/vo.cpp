#include "ascii_banner.hpp"
#include "backward-cpp/backward.hpp"
#include "tasks/base/common.hpp"
#include "tasks/vslam/feature/orb.hpp"
#include "tasks/vslam/frame.hpp"
#include "tasks/vslam/type.hpp"
#include "utils/buffer.hpp"
#include "utils/common/image.hpp"
#include "utils/common/type_common.hpp"
#include "utils/drivers/camera.hpp"
#include "utils/drivers/camera_factory.hpp"
#include "utils/runtime_tf.hpp"
#include "utils/scheduler/node.hpp"
#include "utils/scheduler/scheduler.hpp"
#include "utils/semaphore_guard.hpp"
#include "utils/signal_guard.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
using namespace awakening;

namespace backward {
static backward::SignalHandling sh;
}
struct CameraTag {};
struct FrameTag {};
struct DetectTag {};
using CameraIO = IOPair<CameraTag, ImageFrame>;
using CommonFrameIo = IOPair<FrameTag, CommonFrame>;
using DetectIO = IOPair<DetectTag, std::vector<vslam::Frame>>;
enum class VOFrame : int { ODOM, CAMERA, CAMERA_CV, N };
using VOTF = utils::tf::RobotTF<VOFrame, static_cast<size_t>(VOFrame::N), false>;
struct LogCtx {
    int camera_count = 0;
    int detect_count = 0;
    double detect_cost = 0.0;
    int match_count = 0;
    double match_cost = 0.0;

    void reset() {
        *this = {};
    }
};
int main(int argc, char** argv) {
    print_banner();
    auto& signal = utils::SignalGuard::instance();
    logger::init(spdlog::level::trace);
    bool debug = false;
    std::string config_path;
    auto first_arg = utils::get_arg(1, argc, argv);
    if (first_arg) {
        config_path = first_arg.value();
    } else {
        return 1;
    }
    auto second_arg = utils::get_arg(2, argc, argv);
    if (second_arg) {
        debug = second_arg.value() == "true";
    }
    auto config = YAML::LoadFile(config_path);
    Scheduler s;
    auto camera_config = config["camera"];
    std::unique_ptr<Camera> camera;
    utils::SignalGuard::add_callback([&]() {
        if (camera) {
            camera->stop();
        }
    });
    camera = create_camera(camera_config, s, "hik");
    camera->init();
    CameraInfo camera_info;
    camera_info.load(camera_config["camera_info"]);
    vslam::Orb orb(config["orb"]);
    LogCtx log_ctx;
    auto tf = VOTF::create();
    {
        for (auto [from, to]: std::array {
                 std::pair { VOFrame::ODOM, VOFrame::CAMERA },
                 std::pair { VOFrame::CAMERA, VOFrame::CAMERA_CV },
             })
        {
            tf->add_edge(from, to);
        }
        ISO3 cv_in_camera = ISO3::Identity();
        cv_in_camera.linear() = R_CV2PHYSICS;
        tf->push(VOFrame::CAMERA, VOFrame::CAMERA_CV, Clock::now(), cv_in_camera);
    }
    utils::OrderedQueue<vslam::Frame> features_queue;
    s.register_task<CameraIO, CommonFrameIo>("push_common_frame", [&](CameraIO::second_type&& f) {
        static int current_id = 0;
        if (f.src_img.empty()) {
            return std::make_tuple(std::optional<CommonFrameIo::second_type>(std::nullopt));
        }
        log_ctx.camera_count++;
        CommonFrame frame {
            .img_frame = std::move(f),
            .id = current_id++,
            .frame_id = std::to_underlying(VOFrame::CAMERA_CV),
        };

        return std::make_tuple(std::optional<CommonFrameIo::second_type>(std::move(frame)));
    });
    s.register_task<CommonFrameIo, DetectIO>("detector", [&](CommonFrameIo::second_type&& f) {
        static std::unique_ptr<std::counting_semaphore<>> detector_sem;
        if (!detector_sem) {
            detector_sem = std::make_unique<std::counting_semaphore<>>(3);
        }
        cv::Mat gray;
        if(f.img_frame.format!=PixelFormat::GRAY){
            cv::cvtColor(f.img_frame.src_img, gray, cv::COLOR_BGR2GRAY);
        }
        vslam::Frame frame {
            .img_gray = gray.empty() ? f.img_frame.src_img : gray,
            .timestamp = f.img_frame.timestamp,
            .id = f.id,
            
        };
        {
            bool got = detector_sem->try_acquire();
            utils::SemaphoreGuard guard(*detector_sem, got); //并发控制
            if (got) {
                auto start = Clock::now();
                orb.detect(frame);
                log_ctx.detect_count++;
                auto end = Clock::now();
                log_ctx.detect_cost +=
                    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            }
        }

        features_queue.enqueue(frame);
        auto batch_features = features_queue.dequeue_batch();
        return std::make_tuple(std::optional<DetectIO::second_type>(std::move(batch_features)));
    });
    s.register_task<DetectIO>("tracker", [&](DetectIO::second_type&& frames) {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<vslam::Frame> detected;
        for (auto& frame: frames) {
            if (frame.detected) {
                detected.push_back(frame);
            }
        }
        
    });
    s.add_rate_source<>("logger", 1.0, [&]() {
        double detect_avg_cost =
            log_ctx.detect_count > 0 ? log_ctx.detect_cost / log_ctx.detect_count : 0;
        double match_avg_cost =
            log_ctx.match_count > 0 ? log_ctx.match_cost / log_ctx.match_count : 0;
        AWAKENING_INFO(
            "camera: {} detect: {} cost: {:.3} ms match: {} cost: {:.3} ms",
            log_ctx.camera_count,
            log_ctx.detect_count,
            detect_avg_cost,
            log_ctx.match_count,
            match_avg_cost
        );
        log_ctx.reset();
    });
    if (camera) {
        camera->start<CameraTag>("hik");
    }
    s.build();
    s.run();
    utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    s.stop();
    return 0;
}