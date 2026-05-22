# 视频编解码修复说明

## 问题诊断

解码器出现大量 "error while decoding MB X Y, bytestream -15" 错误，原因是：
1. **冗余包数不匹配**：解码端在尝试错误的 FEC 冗余值时，RS 解码会产生垃圾数据
2. **缺乏确定性恢复**：解码端需要猜测编码端使用的冗余包数，导致频繁失败

## 解决方案

在协议的 `flags` 字段中明确编码 `r` 值（冗余包数量），让解码端准确知道每帧的 FEC 参数。

### 修改 1：common.hpp（协议定义）

flags 字段布局：
- bit [2:0]：标志位（FLAG_KEYFRAME, FLAG_FEC_PACKET, FLAG_MERGED）
- bit [7:3]：当 FLAG_MERGED 置位时存储 merge_count；否则存储冗余包数 r（0-7）

```cpp
// flags bits [7:3]: 当 FLAG_MERGED 置位时，存储子帧数；否则存储冗余包数 r (0-7)
```

### 修改 2：encoder.cpp（编码端）

在 `send_encoded_frame()` 中，将实际的 FEC 冗余包数 `r` 编码到 flags：

```cpp
// 编码 r 值到 flags 的 [7:3] 位（当 FLAG_MERGED 未置位时）
uint8_t flags_with_r = flags;
if (!(flags & FLAG_MERGED)) {
    flags_with_r |= ((r & 0x1F) << 3);
}

// 所有包都使用 flags_with_r 而不是 flags
pkt.header.flags = flags_with_r;
fec.header.flags = flags_with_r | FLAG_FEC_PACKET;
```

### 修改 3：video_decoder.py（解码端）

- 从 flags 中提取 `r_value`
- 优先使用编码端指定的 r 值进行恢复
- 只在编码端 r 值不适用时才尝试其他候选值

```python
def _parse_header(self, pkt):
    # ...
    is_merged = bool(flags & FLAG_MERGED)
    merge_count = (flags >> 3) & 0x1F if is_merged else 0
    r_value = ((flags >> 3) & 0x1F) if not is_merged else 0
    # ...

def _try_recover_frame(self, fid):
    # ...
    r_encoded = buf.get("r_value", 0)
    candidates = []
    if r_encoded > 0:
        candidates.append(r_encoded)  # 优先使用编码端指定的 r 值
    # ...
```

## 编码端编译

```bash
cd /home/auauau/awakening
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## 解码端测试

```bash
cd /home/auauau/doorlock_decoder
python3 video_decoder.py node_params.yaml
```

## 配置验证

确保编码端和解码端配置一致（或让编码端通过 flags 指定）：

**encoder 配置（hero.yaml）**：
- `raid_redundancy: 3` - IDR 帧的 FEC 冗余包数
- `p_redundancy: 2` - P 帧的 FEC 冗余包数

**decoder 配置（node_params.yaml）**：
- `fec_redundancy: 3` - 备用 IDR FEC 冗余值
- `p_redundancy: 2` - 备用 P 帧 FEC 冗余值

现在解码端会：
1. 首先使用编码端在 flags 中指定的 r 值
2. 只有在解析失败时才尝试配置的备用值
3. 这大大提高了恢复成功率并减少错误日志

## 网络约束

- 每个包最大 300 字节（已满足）
- 每秒最多 50 个包（由令牌桶限流实现）
