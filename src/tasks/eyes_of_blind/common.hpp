#pragma once
#include <array>
#include <cstdint>
namespace awakening::eyes_of_blind {
static constexpr std::size_t MAX_PACKET_SIZE = 300;

#pragma pack(push, 1)
struct PacketHeader {
    uint64_t sequence_id;
};
#pragma pack(pop)

static_assert(sizeof(PacketHeader) == 8, "Header must be 8 bytes");
static constexpr std::size_t HEADER_SIZE = sizeof(PacketHeader); // 8 bytes for header
static constexpr std::size_t PAYLOAD_SIZE = 292;

#pragma pack(push, 1)  
struct BlindSend {
    PacketHeader header;
    std::array<uint8_t, PAYLOAD_SIZE> data {};
};
#pragma pack(pop)

struct SerialSendPacket {
    static constexpr uint8_t ID = 0x10;
    uint8_t cmd_ID = ID;
    uint8_t data[300]; 
} __attribute__((packed));

} // namespace awakening::eyes_of_blind
