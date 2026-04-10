#pragma once

#include <cstdint>

namespace awakening::io {

uint8_t get_crc8(const uint8_t* data, uint16_t len);

bool check_crc8(const uint8_t* data, uint16_t len);

uint16_t get_crc16(const uint8_t* data, uint32_t len);

bool check_crc16(const uint8_t* data, uint32_t len);

} // namespace awakening::io
