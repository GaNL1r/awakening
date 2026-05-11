#include "ascii_banner.hpp"
#include "tasks/eyes_of_blind/decoder.hpp"
#include "tasks/eyes_of_blind/encoder.hpp"
#include "utils/drivers/hik_camera.hpp"
#include "utils/drivers/serial_driver.hpp"
#include "utils/logger.hpp"
#include "utils/semaphore_guard.hpp"
#include "utils/signal_guard.hpp"
#include "utils/utils.hpp"
#include "video_stream.pb.h"
#include <array>
#include <cstring>
#include <mqtt/async_client.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <random>  // 添加随机数支持
#include <yaml-cpp/node/parse.h>

using namespace awakening;

struct CameraTag {};
struct SerialTag {};

using CameraIO = IOPair<CameraTag, ImageFrame>;
using SerialIO = IOPair<SerialTag, std::vector<uint8_t>>;

// ==================== 丢包模拟器 ====================
class PacketLossSimulator {
public:
    PacketLossSimulator(double loss_rate = 0.0, int seed = 42) 
        : loss_rate_(std::clamp(loss_rate, 0.0, 1.0)),
          gen_(seed),
          dist_(0.0, 1.0),
          total_packets_(0),
          dropped_packets_(0),
          last_report_time_(std::chrono::steady_clock::now()) {
        // AWAKENING_INFO("PacketLossSimulator initialized with loss_rate={:.1f}%", loss_rate_ * 100);
    }
    
    // 设置新的丢包率（0.0 ~ 1.0）
    void set_loss_rate(double rate) {
        loss_rate_ = std::clamp(rate, 0.0, 1.0);
        // AWAKENING_INFO("PacketLossSimulator: loss_rate updated to {:.1f}%", loss_rate_ * 100);
    }
    
    // 检查是否应该丢弃当前包
    // 返回 true = 丢弃，false = 保留
    bool should_drop() {
        total_packets_++;
        
        if (loss_rate_ <= 0.0) {
            return false;
        }
        
        // 生成随机数并与丢包率比较
        if (dist_(gen_) < loss_rate_) {
            dropped_packets_++;
            print_stats();
            return true;
        }
        
        // 每5秒打印统计
        print_stats();
        return false;
    }
    
    // 获取统计信息
    void print_stats() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_report_time_).count();
        
        if (elapsed >= 5 && total_packets_ > 0) {
            double actual_loss = (double)dropped_packets_ / total_packets_ * 100.0;
            // AWAKENING_INFO(" Packet Loss Stats: "
            //               "total={}, dropped={}, rate={:.1f}% (target={:.1f}%)",
            //               total_packets_, dropped_packets_, actual_loss, loss_rate_ * 100);
            last_report_time_ = now;
        }
    }
    
    size_t total_packets() const { return total_packets_; }
    size_t dropped_packets() const { return dropped_packets_; }

private:
    double loss_rate_;                              // 丢包率 (0.0~1.0)
    std::mt19937 gen_;                              // 随机数生成器（Mersenne Twister）
    std::uniform_real_distribution<double> dist_;   // [0.0, 1.0) 均匀分布
    size_t total_packets_;
    size_t dropped_packets_;
    std::chrono::steady_clock::time_point last_report_time_;
};
// ==================== 丢包模拟器结束 ====================


int main(int argc, char** argv) {
    print_banner();
    auto& signal = utils::SignalGuard::instance();
    logger::init(spdlog::level::trace);

    // 参数解析
    if (argc < 2) return 1;
    std::string config_path = argv[1];
    auto config = YAML::LoadFile(config_path);

    Scheduler s;

    double target_fps = config["encoder"]["fps"].as<double>(30.0);
    if (target_fps <= 0) target_fps = 30.0;
    std::chrono::microseconds frame_interval(static_cast<int64_t>(1e6 / target_fps));

    // 串口驱动
    std::unique_ptr<SerialDriver> serial;
    if (config["serial"]["enable"].as<bool>())
        serial = std::make_unique<SerialDriver>(config["serial"], s);

    // MQTT 客户端
    std::unique_ptr<mqtt::async_client> mqtt_client;
    std::string mqtt_topic;
    bool use_mqtt = false;
    if (config["mqtt"]["enable"].as<bool>(false)) {
        use_mqtt = true;
        mqtt_client = std::make_unique<mqtt::async_client>(
            config["mqtt"]["server"].as<std::string>("tcp://127.0.0.1:3333"),
            config["mqtt"]["client_id"].as<std::string>("eyes_blind_encoder"));
        mqtt_topic = config["mqtt"]["topic"].as<std::string>("CustomByteBlock");
        mqtt::connect_options opts;
        opts.set_keep_alive_interval(20);
        opts.set_clean_session(true);
        try {
            mqtt_client->connect(opts)->wait();
            AWAKENING_INFO("MQTT connected");
        } catch (...) {
            AWAKENING_ERROR("MQTT connection failed");
            use_mqtt = false;
        }
    }

    PacketLossSimulator loss_sim(config["packet_loss_sim"]["loss_rate"].as<double>(0.0), 42);

    // 相机初始化
    auto camera = std::make_unique<HikCamera>(config["camera"]["hik_camera"], s);
    camera->init();
    if (!camera->running_) return 0;

    // 编码器
    eyes_of_blind::Encoder encoder(config["encoder"]);

    // ========== 帧队列及线程控制 ==========
    std::queue<cv::Mat> frame_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> running{true};
    // 任务1：相机回调（极轻）
    s.register_task<CameraIO>("camera_feeder", [&](CameraIO::second_type&& f) {
        static int cb_count = 0;
        // AWAKENING_INFO("Camera callback triggered, count={}", ++cb_count);
        if (f.src_img.empty()) return;
        cv::Mat frame_copy = f.src_img.clone(); // 必须拷贝，因为原帧会被释放
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (frame_queue.size() > 2) // 队列上限2帧，丢弃旧帧
                frame_queue.pop();
            frame_queue.push(std::move(frame_copy));
        }
        queue_cv.notify_one();
    });

    // 启动相机
    camera->start<CameraTag>("hik");
    if (serial) serial->start<SerialTag>("serial");
    

    // ========== 编码线程 ==========
    // 编码线程
    // 独立推流线程（只推不拉）
    std::thread push_thread([&]() {
        using namespace std::chrono;
        auto next_push = steady_clock::now();
        cv::Mat latest_frame;
        bool has_new = false;

        while (running) {
            // 拿最新帧
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (queue_cv.wait_for(lock, milliseconds(5), [&]{ return !frame_queue.empty() || !running; })) {
                    if (frame_queue.empty()) continue;
                    latest_frame = std::move(frame_queue.front());
                    frame_queue.pop();
                    while (!frame_queue.empty()) {
                        latest_frame = std::move(frame_queue.front());
                        frame_queue.pop();
                    }
                    has_new = true;
                } else {
                    if (!running) break;
                    continue;
                }
            }

            // 等待至预定时间
            auto now = steady_clock::now();
            if (now < next_push) {
                std::this_thread::sleep_until(next_push);
                now = steady_clock::now();
            }

            if (has_new) {
                encoder.push_frame(latest_frame);
                has_new = false;
            }
            next_push = std::max(now, next_push + frame_interval);
        }
    });
    
    // 独立拉流 + 发送线程
    std::thread pull_thread([&]() {
        using namespace std::chrono;
        auto next_send = steady_clock::now();
        while (running) {
            if (!encoder.is_pipeline_alive()) {   // 检测到管线失效，立即退出
                AWAKENING_WARN("Encoder pipeline dead, exiting pull thread.");
                running = false;
                break;
            }
            encoder.pull_and_packetize();
            eyes_of_blind::BlindSend pkg;
            while (encoder.try_pop_packet(pkg)) {
                // 严格 20ms 间隔
                auto now = steady_clock::now();
                if (now < next_send) {
                    std::this_thread::sleep_until(next_send);
                    now = steady_clock::now();
                }
                // 发送（串口 / MQTT）
                if (serial && config["serial"]["enable"].as<bool>()) {
                    eyes_of_blind::SerialSendPacket send{};
                    std::memcpy(&send.data, &pkg, sizeof(eyes_of_blind::BlindSend));
                    serial->write(utils::to_vector(send));
                } else if (use_mqtt && mqtt_client && mqtt_client->is_connected()) {
                    if (loss_sim.should_drop()) {
                    next_send += milliseconds(20);  // 严格间隔，保留
                    continue;}
                    
                    std::array<uint8_t, eyes_of_blind::MAX_PACKET_SIZE> raw{};
                    std::memcpy(raw.data(), &pkg, eyes_of_blind::MAX_PACKET_SIZE);
                    doorlock_sniper::CustomByteBlock block;
                    block.set_data(raw.data(), eyes_of_blind::MAX_PACKET_SIZE);
                    std::string serialized;
                    if (block.SerializeToString(&serialized))
                        mqtt_client->publish(mqtt::make_message(mqtt_topic, serialized));
                }
                next_send = std::max(now, next_send + milliseconds(20));
            }
            std::this_thread::sleep_for(milliseconds(1));
        }
    });
    // 启动调度器
    s.build();
    s.run();

    // 等待退出信号

    utils::SignalGuard::add_callback([&]() {
        running = false;
        queue_cv.notify_all();
        if (camera) camera->stop();
    });

    utils::SignalGuard::spin(std::chrono::milliseconds(100));
    if (push_thread.joinable()) push_thread.join();
    if (pull_thread.joinable()) pull_thread.join();
    // 停止处理
    running = false;
    queue_cv.notify_all();

    s.stop();

    loss_sim.print_stats();

    if (mqtt_client && mqtt_client->is_connected())
        mqtt_client->disconnect()->wait();

    AWAKENING_CRITICAL("changed files, remember to sync other exe");
    return 0;
}
// GTREAMER调试命令：
// GST_DEBUG=3,*appsink*:6,*appsrc*:6,*queue*:6,*caps*:6 ./bin/blind_test config/hero.yaml