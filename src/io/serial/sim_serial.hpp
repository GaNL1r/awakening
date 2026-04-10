#pragma once

#include "io/core/packet.hpp"
#include <cstdint>
#include <string>

namespace awakening::io {

class SimSerial {
public:
    SimSerial();
    ~SimSerial() = default;

    bool initialize(const std::string& config_path);
    bool read(Packet& packet, int16_t size);
    bool write(const Packet& packet);

private:
    void init_params();

    static int8_t cdc_receive(uint8_t* buf, uint32_t* len);

    static char buffer_[256];
    static int16_t buffer_size_;
    static int16_t receive_size_;
    static int16_t send_size_;
    static int16_t send_id_list_[32];
    static int16_t send_id_num_;

    struct VisionGimbalReceive {
        float yaw;
        float pitch;
        float roll;
        int mode;
        int color;
    };

    struct VisionShootReceive {
        float bullet_speed;
    };

    struct VisionGimbalSend {
        float yaw;
        float pitch;
    };

    struct VisionShootSend {
        int fire_flag;
    };

    struct Message {
        void* ptr_list[32];
        int16_t size_list[32];
    };

    static VisionGimbalReceive vision_gimbal_recv_;
    static VisionShootReceive vision_shoot_recv_;
    static VisionGimbalSend vision_gimbal_send_;
    static VisionShootSend vision_shoot_send_;

    static Message receive_;
    static Message send_;
};

} // namespace awakening::io
