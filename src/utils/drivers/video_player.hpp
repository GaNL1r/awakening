#pragma once
#include "utils/common/image.hpp"
#include "utils/logger.hpp"
#include "utils/scheduler/scheduler.hpp"
#include <opencv2/videoio.hpp>
#include <string>
#include <thread>
#include <yaml-cpp/node/node.h>
namespace awakening {
class VideoPlayer {
public:
    VideoPlayer(const YAML::Node& config, Scheduler& scheduler): scheduler_(scheduler) {
        path_ = config["path"].as<std::string>();
        fps_ = config["fps"].as<int>();
        loop_ = config["loop"].as<bool>();
        start_frame_ = config["start_frame"].as<int>();
    }
    ~VideoPlayer() {
        stop();
    }
    template<typename Tag>
    void start(std::string source_name) {
        cap_.open(path_);
        if (!cap_.isOpened()) {
            AWAKENING_ERROR("open {} failed", path_);
        }
        cap_.set(cv::CAP_PROP_POS_FRAMES, start_frame_);
        using IO = IOPair<Tag, ImageFrame>;
        source_snapshot_id_ = scheduler_.register_source<IO>(source_name);
        running_ = true;
        worker_ = std::thread(&VideoPlayer::run_loop<IO>, this);
    }
    void stop() {
        running_ = false;
        if (worker_.joinable()) {
            worker_.join();
        }
        cap_.release();
        AWAKENING_INFO("Video closed successfully: {}", path_);
    }
    template<typename IO>
    void run_loop() {
        const auto frame_interval =
            std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / fps_));

        auto next_frame_time = Clock::now();

        while (running_) {
            cv::Mat frame_bgr;
            cap_ >> frame_bgr;

            if (frame_bgr.empty()) {
                if (loop_) {
                    cap_.set(cv::CAP_PROP_POS_FRAMES, start_frame_);
                    continue;
                } else {
                    break;
                }
            }

            ImageFrame frame;
            frame.src_img = std::move(frame_bgr);
            frame.timestamp = Clock::now();
            frame.format = PixelFormat::BGR;
            scheduler_.runtime_push_source<IO>(source_snapshot_id_, [f = std::move(frame)]() {
                return std::make_tuple(std::optional<typename IO::second_type>(std::move(f)));
            });

            next_frame_time += frame_interval;
            std::this_thread::sleep_until(next_frame_time);
        }
    }
    std::string path_;
    int fps_;
    bool loop_;
    int start_frame_;
    std::atomic<bool> running_;
    cv::VideoCapture cap_;
    std::thread worker_;
    Scheduler& scheduler_;
    size_t source_snapshot_id_;
};
} // namespace awakening