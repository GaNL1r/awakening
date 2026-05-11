#pragma once

#include <array>
#include <cstdint>

namespace awakening::eyes_of_blind {

static constexpr std::size_t MAX_PACKET_SIZE = 300;

static constexpr uint8_t FLAG_KEYFRAME   = 0x01;   // bit0: IDR 帧
static constexpr uint8_t FLAG_FEC_PACKET = 0x02;   // bit1: 冗余包
static constexpr uint8_t FLAG_MERGED     = 0x04;   // 此帧为合并帧
// flags bits [7:3] 用于存储子帧数（当 FLAG_MERGED 置位时）
#pragma pack(push, 1)
struct PacketHeader
{
    uint32_t frame_id;      // 当前视频帧(AU)编号
    uint16_t frag_idx;      // 当前分片编号 从 0 开始
    uint16_t frag_count;    // 当前帧总共有多少分片
    uint16_t payload_size;  // 当前分片有效载荷大小 最后一个分片可能不足
    uint16_t frame_size;    // 原始帧总字节数
    uint8_t flags;          // 标志位 bit0 = 是否关键帧(IDR)
};
#pragma pack(pop)
static_assert(sizeof(PacketHeader) == 13);
static constexpr std::size_t HEADER_SIZE =
    sizeof(PacketHeader);
static constexpr std::size_t PAYLOAD_SIZE =
    MAX_PACKET_SIZE - HEADER_SIZE;

#pragma pack(push, 1)
struct BlindSend
{
    PacketHeader header;
    std::array<uint8_t, PAYLOAD_SIZE> data {};
};
#pragma pack(pop)
static_assert(sizeof(BlindSend) == 300);

struct SerialSendPacket
{
    static constexpr uint8_t ID = 0x10;
    uint8_t cmd_ID = ID;
    uint8_t data[MAX_PACKET_SIZE];
} __attribute__((packed));

}