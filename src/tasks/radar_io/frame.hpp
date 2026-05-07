#pragma once

#include "tasks/radar_io/crc.hpp"
#include "utils/utils.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
namespace awakening::radar_io {
struct FrameHeader {
    static constexpr uint8_t SOF = 0xA5;
    uint8_t sof = SOF;
    uint16_t data_length;
    uint8_t seq;
    uint8_t crc8;
};
inline void push_u8(std::vector<uint8_t>& out, uint8_t value) {
    out.push_back(value);
}

inline void push_u16(std::vector<uint8_t>& out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
}

inline void push_u32(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
}
inline std::vector<uint8_t> pack_frame(uint16_t cmd_id, const std::vector<uint8_t>& data) {
    static uint8_t seq = 0;

    constexpr size_t header_size = 5;
    constexpr size_t cmd_size = 2;
    constexpr size_t crc16_size = 2;
    std::vector<uint8_t> frame;
    frame.reserve(header_size + cmd_size + data.size() + crc16_size);
    frame.push_back(FrameHeader::SOF);
    push_u16(frame, static_cast<uint16_t>(data.size()));
    frame.push_back(seq);
    frame.push_back(0);
    frame[4] = get_crc8(frame.data(), static_cast<uint32_t>(frame.size() - 1));
    push_u16(frame, cmd_id);
    frame.insert(frame.end(), data.begin(), data.end());

    const uint16_t crc16 = get_crc16(frame.data(), static_cast<uint32_t>(frame.size()));
    push_u16(frame, crc16);

    return frame;
}
template<class T>
inline std::vector<uint8_t> pack_frame(const T& data) {
    return pack_frame(T::CMDID, utils::to_vector(data));
}
struct RoboStatus {
    static constexpr uint16_t CMDID = 0x0201;
    uint8_t robot_id;
    uint8_t robot_level;
    uint16_t current_HP;
    uint16_t maximum_HP;
    uint16_t shooter_barrel_cooling_value;
    uint16_t shooter_barrel_heat_limit;
    uint16_t chassis_power_limit;
    uint8_t power_management_gimbal_output : 1;
    uint8_t power_management_chassis_output : 1;
    uint8_t power_management_shooter_output : 1;
} __attribute__((packed));
struct RobotMarks {
    bool hero;
    bool engineer;
    bool infantry3;
    bool infantry4;
    bool drone;
    bool sentry;
};
struct RadarMark {
    static constexpr uint16_t CMDID = 0x020C;
    RobotMarks enemy;
    RobotMarks ally;
    static RadarMark create(const uint8_t* data, size_t len) {
        RadarMark r;

        if (len < 2)
            return r;

        uint16_t bits = data[0] | (data[1] << 8);

        auto get = [&](int bit) { return bits & (1u << bit); };

        r.enemy.hero = get(0);
        r.enemy.engineer = get(1);
        r.enemy.infantry3 = get(2);
        r.enemy.infantry4 = get(3);
        r.enemy.drone = get(4);
        r.enemy.sentry = get(5);

        r.ally.hero = get(6);
        r.ally.engineer = get(7);
        r.ally.infantry3 = get(8);
        r.ally.infantry4 = get(9);
        r.ally.drone = get(10);
        r.ally.sentry = get(11);

        return r;
    }
} __attribute__((packed));
struct RadarInfo {
    static constexpr uint16_t CMDID = 0x020E;
    uint8_t double_vulnerability_chance = 0;

    bool enemy_is_double_vulnerable = false;

    uint8_t encryption_level = 0;

    bool can_change_key = false;

    static RadarInfo create(const uint8_t* data, size_t len) {
        RadarInfo r;

        if (len < 1)
            return r;

        uint8_t bits = data[0];

        auto get = [&](int bit) { return bits & (1u << bit); };

        auto get_range = [&](int start, int count) {
            return (bits >> start) & ((1u << count) - 1);
        };

        // bit0-1
        r.double_vulnerability_chance = get_range(0, 2);

        // bit2
        r.enemy_is_double_vulnerable = get(2);

        // bit3-4
        r.encryption_level = get_range(3, 2);

        // bit5
        r.can_change_key = get(5);

        return r;
    }

} __attribute__((packed));
enum class CMDID : uint16_t {
    RoboStatus = RoboStatus::CMDID,
    RadarMark = RadarMark::CMDID,
    RadarInfo = RadarInfo::CMDID,
};

struct RadarCmd {
    static constexpr uint16_t CMDID = 0X0121;
    uint8_t radar_cmd;
    uint8_t password_cmd;
    uint8_t password_1;
    uint8_t password_2;
    uint8_t password_3;
    uint8_t password_4;
    uint8_t password_5;
    uint8_t password_6;
} __attribute__((packed));
struct MapRobotData {
    static constexpr uint16_t CMDID = 0x0305;
    uint16_t opponent_hero_position_x;
    uint16_t opponent_hero_position_y;
    uint16_t opponent_engineer_position_x;
    uint16_t opponent_engineer_position_y;
    uint16_t opponent_infantry_3_position_x;
    uint16_t opponent_infantry_3_position_y;
    uint16_t opponent_infantry_4_position_x;
    uint16_t opponent_infantry_4_position_y;
    uint16_t opponent_aerial_position_x;
    uint16_t opponent_aerial_position_y;
    uint16_t opponent_sentry_position_x;
    uint16_t opponent_sentry_position_y;
    uint16_t ally_hero_position_x;
    uint16_t ally_hero_position_y;
    uint16_t ally_engineer_position_x;
    uint16_t ally_engineer_position_y;
    uint16_t ally_infantry_3_position_x;
    uint16_t ally_infantry_3_position_y;
    uint16_t ally_infantry_4_position_x;
    uint16_t ally_infantry_4_position_y;
    uint16_t ally_aerial_position_x;
    uint16_t ally_aerial_position_y;
    uint16_t ally_sentry_position_x;
    uint16_t ally_sentry_position_y;
} __attribute__((packed));
struct CustomInfo {
    static constexpr uint16_t CMDID = 0x0308;
    uint16_t sender_id;
    uint16_t receiver_id;
    uint8_t user_data[30] = { 0 };
};
} // namespace awakening::radar_io