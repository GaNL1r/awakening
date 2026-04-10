#pragma once

#include "io/core/packet.hpp"
#include <memory>
#include <string>

namespace awakening::io {

class ISerial {
public:
    virtual ~ISerial() = default;

    virtual bool initialize(const std::string& config_path) = 0;
    virtual bool read(Packet& packet, int16_t size) = 0;
    virtual bool write(const Packet& packet) = 0;
    virtual void close() = 0;
};

using SerialPtr = std::unique_ptr<ISerial>;

} // namespace awakening::io
