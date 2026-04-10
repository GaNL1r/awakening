#include "gimbal.hpp"
#include "io/core/crc.hpp"
#include "utils/logger.hpp"
#include <cmath>
#include <yaml-cpp/yaml.h>

namespace awakening::io {

Gimbal::Gimbal(const std::string& config_path) {
    auto yaml = YAML::LoadFile(config_path);
    auto serial_type = yaml["serial_type"].as<std::string>("physical");

    if (serial_type == "physical") {
        serial_ = std::make_unique<AsyncSerial>();
        if (!serial_->initialize(config_path)) {
            AWAKENING_ERROR("[Gimbal] Failed to initialize serial");
            return;
        }
    } else {
        sim_serial_ = std::make_unique<SimSerial>();
        if (!sim_serial_->initialize(config_path)) {
            AWAKENING_ERROR("[Gimbal] Failed to initialize sim serial");
            return;
        }
    }

    thread_ = std::thread(&Gimbal::read_thread, this);
    
    queue_.pop();
    AWAKENING_INFO("[Gimbal] First quaternion received");
}

Gimbal::~Gimbal() {
    quit_ = true;
    if (thread_.joinable()) {
        thread_.join();
    }
    if (serial_) {
        serial_->close();
    }
}

GimbalMode Gimbal::mode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mode_;
}

GimbalState Gimbal::state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

std::string Gimbal::str(GimbalMode mode) const {
    switch (mode) {
        case GimbalMode::IDLE:
            return "IDLE";
        case GimbalMode::AUTO_AIM:
            return "AUTO_AIM";
        case GimbalMode::SMALL_BUFF:
            return "SMALL_BUFF";
        case GimbalMode::BIG_BUFF:
            return "BIG_BUFF";
        default:
            return "INVALID";
    }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t) {
    while (true) {
        auto [q_a, t_a] = queue_.pop();
        auto [q_b, t_b] = queue_.front();
        
        auto t_ab = std::chrono::duration<double>(t_b - t_a).count();
        auto t_ac = std::chrono::duration<double>(t - t_a).count();
        auto k = t_ac / t_ab;
        
        Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
        
        if (t < t_a) {
            return q_c;
        }
        if (!(t_a < t && t <= t_b)) {
            continue;
        }
        
        return q_c;
    }
}

void Gimbal::send(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, 
    float pitch, float pitch_vel, float pitch_acc) {
    
    (void)control;
    (void)yaw_vel;
    (void)yaw_acc;
    (void)pitch_vel;
    (void)pitch_acc;
    
    yaw *= 180.0 / M_PI;
    pitch *= 180.0 / M_PI;
    
    GimbalSend gimbal_send{yaw, pitch};
    ShootSend shoot_send{fire ? 1 : 0};
    
    Packet packet;
    int16_t id1 = 1, id2 = 2;
    packet.write(id1);
    packet.write(gimbal_send);
    packet.write(id2);
    packet.write(shoot_send);
    
    if (serial_) {
        serial_->write(packet);
    } else if (sim_serial_) {
        sim_serial_->write(packet);
    }
}

bool Gimbal::read_thread() {
    AWAKENING_INFO("[Gimbal] read_thread started");
    mode_ = GimbalMode::AUTO_AIM;

    while (!quit_) {
        auto t = std::chrono::steady_clock::now();
        
        Packet packet;
        bool success = false;
        
        if (serial_) {
            success = serial_->read(packet, 22); // sizeof(GimbalReceive) + sizeof(ShootReceive) + 4 (2 IDs)
        } else if (sim_serial_) {
            success = sim_serial_->read(packet, 22);
        }
        
        if (!success) {
            continue;
        }
        
        int16_t id1, id2;
        packet.read(id1);
        if (id1 == 1) {
            packet.read(gimbal_receive_);
        }
        packet.read(id2);
        if (id2 == 2) {
            packet.read(shoot_receive_);
        }
        
        auto yaw = gimbal_receive_.yaw * M_PI / 180.0;
        auto pitch = gimbal_receive_.pitch * M_PI / 180.0;
        auto roll = gimbal_receive_.roll * M_PI / 180.0;
        
        Eigen::Vector3d ypr(yaw, pitch, roll);
        Eigen::Quaterniond q;
        q = Eigen::AngleAxisd(ypr[0], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(ypr[1], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(ypr[2], Eigen::Vector3d::UnitX());
        
        q.normalize();
        queue_.push({q, t});
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        state_.yaw = static_cast<float>(yaw);
        state_.pitch = static_cast<float>(pitch);
        state_.bullet_speed = shoot_receive_.bullet_speed;
    }

    AWAKENING_INFO("[Gimbal] read_thread stopped");
    return true;
}

} // namespace awakening::io
