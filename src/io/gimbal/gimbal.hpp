#pragma once

#include "gimbal_protocol.hpp"
#include "io/core/thread_safe_queue.hpp"
#include "io/protocol/message_base.hpp"
#include "io/serial/async_serial.hpp"
#include "io/serial/sim_serial.hpp"
#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace awakening::io {

enum class GimbalMode {
    IDLE,
    AUTO_AIM,
    SMALL_BUFF,
    BIG_BUFF
};

struct GimbalState {
    float yaw;
    float yaw_vel;
    float pitch;
    float pitch_vel;
    float bullet_speed;
    uint16_t bullet_count;
};

class Gimbal {
public:
    explicit Gimbal(const std::string& config_path);
    ~Gimbal();

    GimbalMode mode() const;
    GimbalState state() const;
    std::string str(GimbalMode mode) const;
    Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

    void send(bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel, float pitch_acc);

private:
    bool read_thread();
    
    std::unique_ptr<AsyncSerial> serial_;
    std::unique_ptr<SimSerial> sim_serial_;
    std::unique_ptr<BaseMessage> message_;
    std::thread thread_;
    std::atomic<bool> quit_{false};
    mutable std::mutex mutex_;
    
    GimbalReceive gimbal_receive_{};
    ShootReceive shoot_receive_{};
    GimbalMode mode_ = GimbalMode::IDLE;
    GimbalState state_;
    
    ThreadSafeQueue<std::pair<Eigen::Quaterniond, std::chrono::steady_clock::time_point>> queue_{1000};
};

} // namespace awakening::io
