#pragma once

#include "io/core/packet.hpp"
#include <memory>
#include <ranges>
#include <string>
#include <unordered_map>

namespace awakening::io {

class BaseMessage {
public:
    BaseMessage() = default;
    virtual ~BaseMessage() = default;

    virtual bool initialize(const std::string& config_path) { return false; }
    virtual bool send() { return false; }
    virtual bool receive() { return false; }
    virtual bool connect(bool flag) { return false; }

    template <typename T>
    bool receive_register(int16_t id);

    template <typename T>
    bool send_register(int16_t id);

    template <typename T>
    bool read_data(T& data);

    template <typename T>
    bool write_data(const T& data);

protected:
    using Registry = std::unordered_map<std::string, std::pair<int16_t, int16_t>>;

    template <typename T>
    bool register_type(Registry& registry, int16_t id);

    template <typename T>
    int16_t get_id(const Registry& registry) const;

    Registry receive_registry_{};
    Registry send_registry_{};
    int16_t receive_size_{};
    Packet receive_buffer_{};
    int16_t send_size_{};
    Packet send_buffer_{};
    std::unordered_map<int16_t, Packet> packets_received_;
};

template <typename T>
bool BaseMessage::receive_register(int16_t id) {
    if (!register_type<T>(receive_registry_, id)) {
        return false;
    }
    packets_received_[id].resize_packet(sizeof(T));
    receive_size_ += sizeof(int16_t) + sizeof(T);
    return true;
}

template <typename T>
bool BaseMessage::send_register(int16_t id) {
    if (!register_type<T>(send_registry_, id)) {
        return false;
    }
    send_size_ += sizeof(int16_t) + sizeof(T);
    return true;
}

template <typename T>
bool BaseMessage::read_data(T& data) {
    const int16_t id = get_id<T>(receive_registry_);
    if (!packets_received_.contains(id)) {
        return false;
    }
    Packet packet = packets_received_[id];
    packet.read(data);
    return true;
}

template <typename T>
bool BaseMessage::write_data(const T& data) {
    const int16_t id = get_id<T>(send_registry_);
    if (!id) {
        return false;
    }
    send_buffer_.write(id);
    send_buffer_.write(data);
    return true;
}

template <typename T>
bool BaseMessage::register_type(Registry& registry, int16_t id) {
    const std::string name = typeid(T).name();
    for (const auto& pair : registry) {
        if (pair.second.first == id) {
            return false;
        }
    }
    if (registry.contains(name)) {
        return false;
    }
    registry[name] = {id, sizeof(T)};
    return true;
}

template <typename T>
int16_t BaseMessage::get_id(const Registry& registry) const {
    const std::string name = typeid(T).name();
    if (registry.contains(name)) {
        return registry.at(name).first;
    }
    return 0;
}

} // namespace awakening::io
