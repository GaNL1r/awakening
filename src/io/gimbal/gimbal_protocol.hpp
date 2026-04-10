#pragma once

#include <cstdint>

namespace awakening::io {

struct GimbalSend {
    float yaw;
    float pitch;
};

struct GimbalReceive {
    float yaw;
    float pitch;
    float roll;
    int mode;
    int color;
};

struct ShootSend {
    int fire_flag;
};

struct ShootReceive {
    float bullet_speed;
};

struct ReceivePacket {
    float yaw;
    float pitch;
    float roll;
    int mode;
    int color;
    float bullet_speed;
};

} // namespace awakening::io
