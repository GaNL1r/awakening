#include "sim_serial.hpp"
#include <cstring>
#include <yaml-cpp/yaml.h>

namespace awakening::io {

char SimSerial::buffer_[256];
int16_t SimSerial::buffer_size_ = 0;
int16_t SimSerial::receive_size_ = 0;
int16_t SimSerial::send_size_ = 0;
int16_t SimSerial::send_id_list_[32];
int16_t SimSerial::send_id_num_ = 0;

SimSerial::VisionGimbalReceive SimSerial::vision_gimbal_recv_;
SimSerial::VisionShootReceive SimSerial::vision_shoot_recv_;
SimSerial::VisionGimbalSend SimSerial::vision_gimbal_send_;
SimSerial::VisionShootSend SimSerial::vision_shoot_send_;

SimSerial::Message SimSerial::receive_;
SimSerial::Message SimSerial::send_;

SimSerial::SimSerial() {
    init_params();
}

void SimSerial::init_params() {
#define REGISTER_ID(data, id, packet) \
    data.ptr_list[id] = &(packet);    \
    data.size_list[id] = sizeof(packet);

    receive_.ptr_list[1] = &vision_gimbal_send_;
    receive_.size_list[1] = sizeof(vision_gimbal_send_);

    receive_.ptr_list[2] = &vision_shoot_send_;
    receive_.size_list[2] = sizeof(vision_shoot_send_);

    send_.ptr_list[1] = &vision_gimbal_recv_;
    send_.size_list[1] = sizeof(vision_gimbal_recv_);

    send_.ptr_list[2] = &vision_shoot_recv_;
    send_.size_list[2] = sizeof(vision_shoot_recv_);
}

bool SimSerial::initialize(const std::string& config_path) {
    (void)config_path;
    init_params();
    return true;
}

bool SimSerial::read(Packet& packet, int16_t size) {
    if (buffer_size_ < sizeof(int16_t)) {
        return false;
    }

    int16_t received_size = *reinterpret_cast<int16_t*>(buffer_);
    if (received_size != size) {
        return false;
    }

    if (buffer_size_ < sizeof(int16_t) + size) {
        return false;
    }

    packet.resize_packet(size);
    std::memcpy(packet.data_ptr(), buffer_ + sizeof(int16_t), size);
    return true;
}

bool SimSerial::write(const Packet& packet) {
    std::vector<char> buffer;
    int16_t size = static_cast<int16_t>(packet.size_packet());

    buffer.insert(buffer.end(), reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + sizeof(int16_t));
    buffer.insert(buffer.end(), reinterpret_cast<const char*>(packet.data_ptr()), reinterpret_cast<const char*>(packet.data_ptr()) + packet.size_packet());

    size_t offset = 0;
    while (offset < buffer.size()) {
        size_t chunk_size = std::min(buffer.size() - offset, static_cast<size_t>(64));
        cdc_receive(reinterpret_cast<uint8_t*>(buffer.data() + offset), nullptr);
        offset += chunk_size;
    }

    return true;
}

int8_t SimSerial::cdc_receive(uint8_t* buf, uint32_t* /*len*/) {
    if (!buf) return -1;

    int16_t buf_pos = 0;
    char* ptr;

    if (receive_size_ == 0) {
        if (buf_pos + sizeof(int16_t) > 64) return -1;
        std::memcpy(&receive_size_, buf + buf_pos, sizeof(int16_t));
        buf_pos += sizeof(int16_t);
        buffer_size_ = 0;
        if (receive_size_ <= 0 || receive_size_ > 1024) {
            receive_size_ = 0;
            return -1;
        }
    }

    int16_t remain_size = receive_size_ - buffer_size_;
    if (remain_size > 0) {
        int16_t copy_size = (remain_size > 64 - buf_pos) ? (64 - buf_pos) : remain_size;
        if (buffer_size_ + copy_size > sizeof(buffer_)) {
            receive_size_ = 0;
            buffer_size_ = 0;
            return -1;
        }
        std::memcpy(buffer_ + buffer_size_, buf + buf_pos, copy_size);
        buffer_size_ += copy_size;
    }

    if (receive_size_ != buffer_size_) return 0;

    ptr = buffer_;
    while (ptr < buffer_ + buffer_size_) {
        if (ptr + sizeof(int16_t) > buffer_ + buffer_size_) break;

        int16_t id;
        std::memcpy(&id, ptr, sizeof(int16_t));
        ptr += sizeof(int16_t);

        if (id == -1) {
            receive_size_ = 0;
            return 0;
        } else if (id == 0) {
            send_id_num_ = 0;
            send_size_ = 0;
            while (ptr < buffer_ + buffer_size_) {
                if (ptr + sizeof(int16_t) > buffer_ + buffer_size_) break;
                std::memcpy(&id, ptr, sizeof(int16_t));
                ptr += sizeof(int16_t);
                if (id < 0 || id >= 32) continue;
                send_id_list_[send_id_num_++] = id;
                send_size_ += send_.size_list[id] + sizeof(int16_t);
                if (send_id_num_ >= 32) break;
            }
        } else {
            if (id < 0 || id >= 32) break;
            if (ptr + receive_.size_list[id] > buffer_ + buffer_size_) break;
            std::memcpy(receive_.ptr_list[id], ptr, receive_.size_list[id]);
            ptr += receive_.size_list[id];
        }
    }

    buffer_size_ = sizeof(int16_t) + send_size_;
    if (buffer_size_ > sizeof(buffer_)) {
        receive_size_ = 0;
        return -1;
    }

    ptr = buffer_;
    std::memcpy(ptr, &send_size_, sizeof(int16_t));
    ptr += sizeof(int16_t);

    for (int i = 0; i < send_id_num_ && i < 32; i++) {
        int16_t id = send_id_list_[i];
        if (id < 0 || id >= 32 || !send_.ptr_list[id]) continue;
        if (ptr + sizeof(int16_t) + send_.size_list[id] > buffer_ + sizeof(buffer_)) break;
        std::memcpy(ptr, &id, sizeof(int16_t));
        ptr += sizeof(int16_t);
        std::memcpy(ptr, send_.ptr_list[id], send_.size_list[id]);
        ptr += send_.size_list[id];
    }

    receive_size_ = 0;
    return 0;
}

} // namespace awakening::io
