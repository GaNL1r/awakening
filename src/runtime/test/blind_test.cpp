#include "ascii_banner.hpp"
#include "tasks/eyes_of_blind/decoder.hpp"
#include "tasks/eyes_of_blind/encoder.hpp"
#include "utils/drivers/hik_camera.hpp"
#include "utils/drivers/serial_driver.hpp"
#include "utils/logger.hpp"
#include "utils/semaphore_guard.hpp"
#include "utils/signal_guard.hpp"
#include "utils/utils.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <mutex>
#include <deque>
#include <yaml-cpp/node/parse.h>
using namespace awakening;

struct CameraTag {};
struct SerialTag {};

using CameraIO = IOPair<CameraTag, ImageFrame>;
using SerialIO = IOPair<SerialTag, std::vector<uint8_t>>;

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
    
    std::string config_path;
    auto first_arg = get_arg(1);
    if (first_arg) {
        config_path = first_arg.value();
    } else {
        return 1;
    }

    Scheduler s;
    auto config = YAML::LoadFile(config_path);
    std::unique_ptr<SerialDriver> serial;

    if (config["serial"]["enable"].as<bool>()) {
        serial = std::make_unique<SerialDriver>(config["serial"], s);
    }

    auto camera_config = config["camera"];
    std::unique_ptr<HikCamera> camera;
    utils::SignalGuard::add_callback([&]() {
        if (camera) {
            camera->stop();
        }
    });

    camera = std::make_unique<HikCamera>(camera_config["hik_camera"], s);
    camera->init();
    if (!camera->running_) {
        return 0;
    }

    eyes_of_blind::Encoder encoder(config["encoder"]);
    
    // ===== 线程安全的包缓冲队列 =====
    static std::mutex send_queue_mutex;
    static std::deque<std::vector<uint8_t>> send_queue;
    static std::atomic<size_t> send_queue_max_size{0};
    // =================================

    // 验证结构体大小
    AWAKENING_INFO("BlindSend size: {}", sizeof(eyes_of_blind::BlindSend));
    AWAKENING_INFO("SerialSendPacket size: {}", sizeof(eyes_of_blind::SerialSendPacket));

    // 任务1：编码（在相机回调中运行）
    s.register_task<CameraIO>("blind_encoder", [&](CameraIO::second_type&& f) {
        if (f.src_img.empty()) {
            return;
        }
        static std::unique_ptr<std::counting_semaphore<>> detector_sem;
        if (!detector_sem) {
            detector_sem = std::make_unique<std::counting_semaphore<>>(1);
        }

        {
            bool got = detector_sem->try_acquire();
            utils::SemaphoreGuard guard(*detector_sem, got);
            if (got) {
                encoder.push_frame(f.src_img);
                eyes_of_blind::BlindSend pkg;
                
                while (encoder.try_pop_packet(pkg)) {
                                                if (serial && config["serial"]["enable"].as<bool>()) {
                            // ===== 直接封装到 SerialSendPacket，不做 Proto 序列化 =====
                        eyes_of_blind::SerialSendPacket send;
                        std::memset(send.data, 0, sizeof(send.data));
                        
                        // 将 BlindSend 的 300 bytes 直接拷贝到 send.data
                        std::memcpy(send.data, &pkg, sizeof(eyes_of_blind::BlindSend));

                        auto* raw = reinterpret_cast<const uint8_t*>(send.data);

                        size_t actual_size = sizeof(eyes_of_blind::BlindSend);  // 或 sizeof(send.data)
                        AWAKENING_INFO(
                            "Encoder seq={}, first 24: {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} "
                            "{:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} "
                            "{:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}",
                            pkg.header.sequence_id,
                            raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
                            raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15],
                            raw[16], raw[17], raw[18], raw[19], raw[20], raw[21], raw[22], raw[23]);

                        AWAKENING_INFO(
                            "Encoder seq={}, last 8: {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}",
                            pkg.header.sequence_id,
                            raw[292], raw[293], raw[294], raw[295], raw[296], raw[297], raw[298], raw[299]);
                        

                        serial->write(std::move(utils::to_vector(send)));
                    }
                }
            }
        }
    });
// 启动驱动
    if (camera) {
        camera->start<CameraTag>("hik");
    }

    if (serial) {
        serial->start<SerialTag>("serial");
    }

    s.build();
    s.run();
    
    utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    s.stop();

    for (int i = 0; i < 10; ++i) {
        AWAKENING_CRITICAL("改了东西记得同步其他有关的exe的src");
    }
    return 0;
}