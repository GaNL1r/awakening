#pragma once

#include "tags.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace awakening::io {

class Packet : public std::vector<uint8_t> {
public:
    Packet() = default;
    explicit Packet(size_t n) : std::vector<uint8_t>(n) {}

    template <typename T>
    bool write(T REF_IN data) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&data);
        insert(end(), ptr, ptr + sizeof(data));
        return true;
    }

    template <typename T>
    bool read(T REF_OUT data) {
        if (begin() + read_offset_ + sizeof(data) > end()) {
            return false;
        }

        std::copy_n(begin() + read_offset_, sizeof(data), reinterpret_cast<uint8_t*>(&data));
        read_offset_ += sizeof(data);
        return true;
    }

    void clear_packet() {
        clear();
        read_offset_ = 0;
    }

    void resize_packet(size_t n) {
        resize(n);
        read_offset_ = 0;
    }

    [[nodiscard]] size_t size_packet() const { return size(); }

    [[nodiscard]] bool empty_packet() const { return read_offset_ >= size(); }

    [[nodiscard]] const uint8_t* data_ptr() const { return data(); }

    uint8_t* data_ptr() { return data(); }

private:
    size_t read_offset_{};
};

} // namespace awakening::io
