#pragma once

#include "tasks/base/packet_typedef_receive.hpp"
#include "tasks/base/packet_typedef_send.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

namespace awakening::control_2026 {

namespace detail {
template<typename T>
void append_raw(std::vector<uint8_t>& out, const T& value) {
    const auto* p = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), p, p + sizeof(T));
}

template<typename T>
bool read_raw(const std::vector<uint8_t>& data, size_t& offset, T& value) {
    if (offset + sizeof(T) > data.size()) {
        return false;
    }
    std::memcpy(&value, data.data() + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}
} // namespace detail

struct GimbalReceive {
    float yaw = 0.0F;
    float pitch = 0.0F;
};

struct ShootReceive {
    int fire_flag = 0;
};

struct GimbalSend {
    float yaw = 0.0F;
    float pitch = 0.0F;
    float roll = 0.0F;
    int mode = 0;
    int color = 0;
};

struct ShootSend {
    float bullet_speed = 0.0F;
};

struct Status {
    ReceiveRobotData robot {};
    std::optional<int> mode;
};

inline std::vector<uint8_t> pack_command(const SendRobotCmdData& cmd, bool fire_advice) {
    GimbalReceive gimbal {
        .yaw = cmd.yaw,
        .pitch = cmd.pitch,
    };
    ShootReceive shoot {
        .fire_flag = fire_advice ? 1 : 0,
    };

    constexpr int16_t payload_size =
        static_cast<int16_t>(sizeof(int16_t) + sizeof(GimbalReceive)
                             + sizeof(int16_t) + sizeof(ShootReceive));
    constexpr int16_t gimbal_id = 1;
    constexpr int16_t shoot_id = 2;

    std::vector<uint8_t> out;
    out.reserve(sizeof(int16_t) + payload_size);
    detail::append_raw(out, payload_size);
    detail::append_raw(out, gimbal_id);
    detail::append_raw(out, gimbal);
    detail::append_raw(out, shoot_id);
    detail::append_raw(out, shoot);
    return out;
}

inline std::optional<Status> unpack_status_with_mode(const std::vector<uint8_t>& data) {
    size_t offset = 0;
    int16_t payload_size = 0;
    if (!detail::read_raw(data, offset, payload_size) || payload_size <= 0) {
        return std::nullopt;
    }

    const size_t packet_size = sizeof(int16_t) + static_cast<size_t>(payload_size);
    if (data.size() < packet_size) {
        return std::nullopt;
    }

    GimbalSend gimbal {};
    ShootSend shoot {};
    bool has_gimbal = false;
    bool has_shoot = false;
    const size_t end = std::min(data.size(), packet_size);

    while (offset + sizeof(int16_t) <= end) {
        int16_t id = -1;
        if (!detail::read_raw(data, offset, id)) {
            break;
        }

        if (id == 1) {
            if (!detail::read_raw(data, offset, gimbal)) {
                return std::nullopt;
            }
            has_gimbal = true;
        } else if (id == 2) {
            if (!detail::read_raw(data, offset, shoot)) {
                return std::nullopt;
            }
            has_shoot = true;
        } else {
            return std::nullopt;
        }
    }

    if (!has_gimbal) {
        return std::nullopt;
    }

    Status out {};
    out.robot.cmd_ID = ReceiveRobotData::ID;
    out.robot.yaw = gimbal.yaw;
    out.robot.pitch = gimbal.pitch;
    out.robot.roll = gimbal.roll;
    out.robot.detect_color = gimbal.color >= 100 ? 0 : 1;
    out.mode = gimbal.mode;
    if (has_shoot) {
        out.robot.bullet_speed = shoot.bullet_speed;
    }
    return out;
}

inline std::optional<ReceiveRobotData> unpack_status(const std::vector<uint8_t>& data) {
    if (auto status = unpack_status_with_mode(data)) {
        return status->robot;
    }
    return std::nullopt;
}

inline std::optional<ReceiveRobotData> unpack_status_or_legacy(const std::vector<uint8_t>& data) {
    if (auto status = unpack_status(data)) {
        return status;
    }
    return ReceiveRobotData::create(data);
}

inline std::vector<uint8_t> pack_command_for_control_2026(
    const SendRobotCmdData& cmd,
    bool fire_advice
) {
    return pack_command(cmd, fire_advice);
}

inline std::vector<uint8_t> pack_command_for_control_2026(const SendRobotCmdData& cmd) {
    return pack_command(cmd, cmd.appear);
}

class StatusStreamParser {
public:
    std::optional<Status> push_status(const std::vector<uint8_t>& data) {
        if (auto legacy = ReceiveRobotData::create(data)) {
            buffer_.clear();
            return Status {
                .robot = *legacy,
                .mode = std::nullopt,
            };
        }

        buffer_.insert(buffer_.end(), data.begin(), data.end());
        while (buffer_.size() >= sizeof(int16_t)) {
            int16_t payload_size = 0;
            std::memcpy(&payload_size, buffer_.data(), sizeof(payload_size));
            if (payload_size <= 0 || payload_size > kMaxPayloadSize) {
                buffer_.erase(buffer_.begin());
                continue;
            }

            const size_t packet_size = sizeof(int16_t) + static_cast<size_t>(payload_size);
            if (buffer_.size() < packet_size) {
                break;
            }

            std::vector<uint8_t> packet(buffer_.begin(), buffer_.begin() + packet_size);
            buffer_.erase(buffer_.begin(), buffer_.begin() + packet_size);
            if (auto status = unpack_status_with_mode(packet)) {
                return status;
            }
        }

        if (buffer_.size() > kMaxBufferedBytes) {
            buffer_.erase(buffer_.begin(), buffer_.end() - kMaxBufferedBytes);
        }
        return std::nullopt;
    }

    std::optional<ReceiveRobotData> push(const std::vector<uint8_t>& data) {
        if (auto status = push_status(data)) {
            return status->robot;
        }
        return std::nullopt;
    }

private:
    static constexpr int16_t kMaxPayloadSize = 128;
    static constexpr size_t kMaxBufferedBytes = 256;
    std::vector<uint8_t> buffer_;
};

} // namespace awakening::control_2026
