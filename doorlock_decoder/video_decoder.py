#!/usr/bin/env python3
"""
解码器：接收 BlindSend 包（protobuf → MQTT），完成 RS 解码、合并帧拆分、H.264 解码与显示。
- 适应高丢包场景：延长帧超时、全局 IDR 超时自动重置、防止重复日志。
- 详细包日志、每秒统计、十字准心显示。
"""
# 启动命令：python3 video_decoder.py params.yaml
# 适配本地丢包20%，30%以上无法解码，错误原因未知，建议重写适配编码端协议的解码端（P帧合包的问题较大）
import struct
import time
import argparse
from collections import deque
from queue import Queue
from typing import Optional, Dict, List, Tuple

import numpy as np
import cv2
import yaml
import paho.mqtt.client as mqtt
import video_stream_pb2

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False
    print("PyAV not installed, H.264 decoding disabled. Install with: pip install av")

# ==================== 常量 ====================
MAX_PACKET_SIZE = 300
HEADER_SIZE = 13
PAYLOAD_SIZE = MAX_PACKET_SIZE - HEADER_SIZE

FLAG_KEYFRAME   = 0x01
FLAG_FEC_PACKET = 0x02
FLAG_MERGED     = 0x04

# ==================== GF(2^8) NumPy 向量化 ====================
gf_exp = np.zeros(512, dtype=np.uint8)
gf_log = np.zeros(256, dtype=np.uint8)

def init_gf():
    x = 1
    for i in range(255):
        gf_exp[i] = x
        gf_exp[i + 255] = x
        gf_log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11d
    gf_log[0] = 0

init_gf()

def gf_mul_scalar(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return int(gf_exp[int(gf_log[a]) + int(gf_log[b])])

def gf_div_scalar(a: int, b: int) -> int:
    if a == 0:
        return 0
    return int(gf_exp[(int(gf_log[a]) - int(gf_log[b])) % 255])

def gf_pow_scalar(a: int, p: int) -> int:
    if p == 0:
        return 1
    r = 1
    for _ in range(p):
        r = gf_mul_scalar(r, a)
    return r

def gf_mul_vector(coeff: int, vec: np.ndarray) -> np.ndarray:
    if coeff == 0:
        return np.zeros_like(vec)
    if coeff == 1:
        return vec.copy()
    logs = gf_log[vec]
    out = gf_exp[(logs.astype(np.uint16) + int(gf_log[coeff])) % 255]
    out[vec == 0] = 0
    return out

def invert_matrix(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    aug = np.zeros((n, n * 2), dtype=np.uint8)
    aug[:, :n] = mat
    for i in range(n):
        aug[i, n + i] = 1
    for col in range(n):
        pivot = None
        for r in range(col, n):
            if aug[r, col] != 0:
                pivot = r
                break
        if pivot is None:
            raise RuntimeError("Matrix singular")
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]
        inv_pivot = gf_div_scalar(1, aug[col, col])
        aug[col] = gf_mul_vector(inv_pivot, aug[col])
        for r in range(n):
            if r == col:
                continue
            factor = aug[r, col]
            if factor == 0:
                continue
            aug[r] ^= gf_mul_vector(factor, aug[col])
    return aug[:, n:]

# ==================== 包解析（增强） ====================
HEADER_FORMAT = '<IHHHHB'

def parse_packet(raw: bytes) -> Optional[Dict]:
    if len(raw) != MAX_PACKET_SIZE:
        return None
    header = raw[:HEADER_SIZE]
    payload = raw[HEADER_SIZE:]
    fid, fidx, fcnt, psize, fsize, flags = struct.unpack(HEADER_FORMAT, header)
    return {
        'frame_id': fid,
        'frag_idx': fidx,
        'frag_count': fcnt,
        'payload_size': psize,
        'frame_size': fsize,
        'flags': flags,
        'payload': payload
    }

def log_packet(pkt: Dict, yaml_r_idr: int, yaml_r_p: int):
    flags = pkt['flags']
    is_idr = bool(flags & FLAG_KEYFRAME)
    is_fec = bool(flags & FLAG_FEC_PACKET)
    is_merged = bool(flags & FLAG_MERGED)
    high_val = (flags >> 3) & 0x1F

    if is_merged:
        sub_cnt = high_val
        extra = f"SUB={sub_cnt}"
    else:
        r_from_flags = high_val
        exp_r = yaml_r_idr if is_idr else yaml_r_p
        if r_from_flags != exp_r:
            extra = f"r={r_from_flags} (WARN: YAML={exp_r})"
        else:
            extra = f"r={r_from_flags}"
    print(f"[PKT] fid={pkt['frame_id']:5d} idx={pkt['frag_idx']:2d}/{pkt['frag_count']:2d} "
          f"fsize={pkt['frame_size']:3d} psize={pkt['payload_size']:3d} "
          f"flg=0x{flags:02x} {'IDR' if is_idr else 'P'} "
          f"{'MERGED' if is_merged else ''} {'FEC' if is_fec else 'DATA'} {extra}")

# ==================== 合并帧拆分 ====================
def split_merged_frame(data: bytes) -> List[bytes]:
    if len(data) < 1:
        return [data]
    n = data[0]
    if n < 1 or n > 4:
        return [data]
    header_len = 1 + 2 * n
    if len(data) < header_len:
        return [data]
    lengths = []
    for i in range(n):
        off = 1 + 2 * i
        length = struct.unpack('<H', data[off:off+2])[0]
        lengths.append(length)
    frames = []
    pos = header_len
    for length in lengths:
        if pos + length > len(data):
            break
        frames.append(data[pos:pos+length])
        pos += length
    return frames

# ==================== 帧重组与 RS 解码 ====================
class FrameReassembler:
    def __init__(self, raid_redundancy: int, p_redundancy: int, timeout_ms: float = 2000):
        self.raid_r = raid_redundancy
        self.p_r = p_redundancy
        self.timeout_ms = timeout_ms
        self.buffers: Dict[int, dict] = {}
        self.completed: Dict[int, Tuple[List[bytes], int, bool]] = {}

    def get_r_from_packet(self, flags: int) -> Tuple[int, Optional[int]]:
        is_merged = bool(flags & FLAG_MERGED)
        high = (flags >> 3) & 0x1F
        if is_merged:
            return self.p_r, high
        else:
            return high, None

    def add_packet(self, pkt: Dict):
        fid = pkt['frame_id']
        if fid not in self.buffers:
            self.buffers[fid] = {
                'packets': {},
                'frag_count': pkt['frag_count'],
                'frame_size': pkt['frame_size'],
                'flags': pkt['flags'],
                'ts': time.time()
            }
        buf = self.buffers[fid]
        idx = pkt['frag_idx']
        if idx not in buf['packets']:
            buf['packets'][idx] = pkt
            buf['ts'] = time.time()
        self._try_reconstruct(fid)

    def _try_reconstruct(self, fid: int):
        buf = self.buffers[fid]
        flags = buf['flags']
        frag_count = buf['frag_count']
        frame_size = buf['frame_size']
        packets = buf['packets']

        r_from_flags, sub_cnt = self.get_r_from_packet(flags)
        k = frag_count - r_from_flags
        if k <= 0:
            return

        data_pkts = {i: p for i, p in packets.items() if i < k and not (p['flags'] & FLAG_FEC_PACKET)}
        fec_pkts  = {i: p for i, p in packets.items() if i >= k and (p['flags'] & FLAG_FEC_PACKET)}

        if len(data_pkts) == k:
            sorted_data = [data_pkts[i] for i in range(k)]
            raw = b''.join(p['payload'][:PAYLOAD_SIZE] for p in sorted_data)[:frame_size]
            self._store_completed(fid, raw, flags, sub_cnt)
        elif len(data_pkts) + len(fec_pkts) >= k:
            selected = list(data_pkts.keys())
            fec_keys = sorted(fec_pkts.keys())
            selected += fec_keys[: k - len(selected)]

            A = np.zeros((k, k), dtype=np.uint8)
            fragments = []
            for row, sel_idx in enumerate(selected):
                fragments.append(np.frombuffer(packets[sel_idx]['payload'][:PAYLOAD_SIZE], dtype=np.uint8))
                if sel_idx < k:
                    A[row, sel_idx] = 1
                else:
                    j = sel_idx - k
                    for col in range(k):
                        coeff = gf_pow_scalar(col + 1, j)
                        A[row, col] = coeff
            try:
                A_inv = invert_matrix(A)
            except RuntimeError:
                return
            B = np.stack(fragments, axis=0)
            X = np.zeros_like(B)
            for i in range(k):
                for j in range(k):
                    if A_inv[i, j] != 0:
                        X[i] ^= gf_mul_vector(A_inv[i, j], B[j])
            raw = X.ravel().tobytes()[:frame_size]
            self._store_completed(fid, raw, flags, sub_cnt)

    def _store_completed(self, fid, raw, flags, sub_cnt):
        if flags & FLAG_MERGED:
            sub_frames = split_merged_frame(raw)
            if sub_cnt is not None and len(sub_frames) != sub_cnt:
                print(f"WARN: merged sub count mismatch: expected {sub_cnt}, got {len(sub_frames)}")
            self.completed[fid] = (sub_frames, flags, True)
        else:
            self.completed[fid] = ([raw], flags, False)
        del self.buffers[fid]

    def is_frame_timed_out(self, fid: int) -> bool:
        if fid not in self.buffers:
            return False
        return (time.time() - self.buffers[fid]['ts']) * 1000 > self.timeout_ms

    def cleanup_timeouts(self) -> List[int]:
        now = time.time()
        expired = [fid for fid, buf in self.buffers.items()
                   if (now - buf['ts']) * 1000 > self.timeout_ms]
        for fid in expired:
            del self.buffers[fid]
        return expired

    def reset(self):
        """清空所有缓存（全局重置时调用）"""
        self.buffers.clear()
        self.completed.clear()

# ==================== 顺序输出控制器 ====================
class Sequencer:
    def __init__(self, reassembler: FrameReassembler, max_idr_wait_sec=5.0):
        self.reasm = reassembler
        self.expected_fid = None
        self.waiting_idr = True
        self.output_queue = deque()
        self.callback_on_emit = None
        self.last_idr_wait_start = 0.0        # 进入 waiting_idr 的时间戳
        self.max_idr_wait_sec = max_idr_wait_sec
        self._logged_lost_frames = set()      # 避免重复打印丢失日志

    def update(self):
        if self.waiting_idr:
            idr_fids = [fid for fid, (_, flags, _) in self.reasm.completed.items()
                        if flags & FLAG_KEYFRAME]
            if idr_fids:
                fid = min(idr_fids)
                self._emit(fid)
                self.expected_fid = fid + 1
                self.waiting_idr = False
                self.last_idr_wait_start = 0.0
                self._logged_lost_frames.clear()
            else:
                # 检查 IDR 缓冲区超时丢弃
                for fid in list(self.reasm.buffers.keys()):
                    if (self.reasm.buffers[fid]['flags'] & FLAG_KEYFRAME) and self.reasm.is_frame_timed_out(fid):
                        print(f"[SEQ] IDR frame {fid} lost (timeout), dropping it")
                        del self.reasm.buffers[fid]

                # 全局 IDR 等待超时重置
                if self.last_idr_wait_start == 0.0:
                    self.last_idr_wait_start = time.time()
                elif time.time() - self.last_idr_wait_start > self.max_idr_wait_sec:
                    print("[SEQ] IDR wait timeout, resetting decoder state...")
                    self.reasm.reset()
                    self.waiting_idr = True
                    self.expected_fid = None
                    self.last_idr_wait_start = time.time()
                    self._logged_lost_frames.clear()
        else:
            # 输出连续完成的帧
            while self.expected_fid in self.reasm.completed:
                self._emit(self.expected_fid)
                self.expected_fid += 1

            # 检查期望帧是否真实丢失（在缓冲中超时）
            if self.expected_fid in self.reasm.buffers and self.reasm.is_frame_timed_out(self.expected_fid):
                if self.expected_fid not in self._logged_lost_frames:
                    print(f"[SEQ] Frame {self.expected_fid} lost (timeout), entering IDR wait")
                    self._logged_lost_frames.add(self.expected_fid)
                self.waiting_idr = True
                self.expected_fid = None
                self.last_idr_wait_start = time.time()

    def _emit(self, fid: int):
        data_list, flags, is_merged = self.reasm.completed.pop(fid)
        for h264_bytes in data_list:
            self.output_queue.append((h264_bytes, flags))
        if self.callback_on_emit:
            self.callback_on_emit(len(data_list))

    def get_frame(self) -> Optional[Tuple[bytes, int]]:
        if self.output_queue:
            return self.output_queue.popleft()
        return None

# ==================== 解码器主类 ====================
class BlindDecoder:
    def __init__(self, config: dict):
        vd = config.get('video_decoder', {})
        self.display = vd.get('display', True)
        self.width = vd.get('width', 128)
        self.height = vd.get('height', 128)
        self.display_scale = vd.get('display_scale', 4)
        self.raid_r = vd.get('raid_redundancy', 2)
        self.p_r = vd.get('p_redundancy', 1)
        # 帧超时时间从配置读取，默认 2000 ms（适应高丢包/抖动）
        self.frame_timeout_ms = vd.get('frame_timeout_ms', 2000)
        self.max_inflight = vd.get('max_frames_inflight', 32)   # 适当增大
        self.test_mode = config.get('test_mode', False)

        self.crosshair_offset_x = vd.get('crosshair_offset_x', 0)
        self.crosshair_offset_y = vd.get('crosshair_offset_y', 0)
        self.crosshair_width = vd.get('crosshair_width', 1)

        self.reassembler = FrameReassembler(
            raid_redundancy=self.raid_r,
            p_redundancy=self.p_r,
            timeout_ms=self.frame_timeout_ms
        )
        self.sequencer = Sequencer(self.reassembler)
        self.sequencer.callback_on_emit = lambda n: self._on_frames_emitted(n)

        self.received_packets = 0
        self.decoded_frames = 0
        self.decode_ok = 0
        self.dropped_frames = 0
        self.last_stat_time = time.time()

    def _on_frames_emitted(self, n: int):
        self.decoded_frames += n

    def feed_packet(self, raw: bytes):
        pkt = parse_packet(raw)
        if pkt is None:
            return
        self.received_packets += 1
        log_packet(pkt, self.raid_r, self.p_r)

        self.reassembler.add_packet(pkt)
        expired = self.reassembler.cleanup_timeouts()
        if expired:
            self.dropped_frames += len(expired)

        if len(self.reassembler.buffers) > self.max_inflight:
            oldest = min(self.reassembler.buffers.keys())
            del self.reassembler.buffers[oldest]

    def update_sequencer(self):
        self.sequencer.update()

    def get_frame(self) -> Optional[Tuple[bytes, int]]:
        return self.sequencer.get_frame()

# ==================== 显示与绘图 ====================
def draw_overlay(img: np.ndarray, fps: float, crosshair_offset_x, crosshair_offset_y, crosshair_width):
    h, w = img.shape[:2]
    cx = max(0, min(w - 1, w // 2 + crosshair_offset_x))
    cy = max(0, min(h - 1, h // 2 + crosshair_offset_y))
    cv2.line(img, (0, cy), (w - 1, cy), (220, 180, 235), 1, cv2.LINE_AA)
    cv2.line(img, (cx, 0), (cx, h - 1), (220, 180, 235), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 18, (170, 255, 170), 1, cv2.LINE_AA)
    cv2.putText(img, f"FPS {fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    decoder = BlindDecoder(config)
    vd = config.get('video_decoder', {})

    raw_queue = Queue()

    def on_message(client, userdata, msg):
        try:
            block = video_stream_pb2.CustomByteBlock()
            block.ParseFromString(msg.payload)
            raw_data = block.data
            if len(raw_data) == MAX_PACKET_SIZE:
                raw_queue.put(raw_data)
        except Exception as e:
            print(f"Protobuf parse error: {e}")

    client = mqtt.Client()
    client.on_message = on_message
    try:
        client.connect(vd.get('mqtt_ip', '127.0.0.1'),
                       vd.get('mqtt_port', 3333))
        client.subscribe(vd.get('mqtt_topic', 'CustomByteBlock'))
        client.loop_start()
        print("MQTT connected and subscribed")
    except Exception as e:
        print(f"MQTT connection failed: {e}")
        client.loop_start()

    if decoder.display and not decoder.test_mode:
        win_name = "decoder"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, decoder.width * decoder.display_scale,
                         decoder.height * decoder.display_scale)

    h264_decoder = None
    last_fps_time = time.time()
    fps_counter = 0
    display_fps = 0.0

    try:
        while True:
            while not raw_queue.empty():
                raw_pkt = raw_queue.get_nowait()
                decoder.feed_packet(raw_pkt)

            decoder.update_sequencer()

            item = decoder.get_frame()
            if item:
                h264_bytes, flags = item
                is_idr = bool(flags & FLAG_KEYFRAME)

                if is_idr or h264_decoder is None:
                    if HAS_AV:
                        h264_decoder = av.CodecContext.create('h264', 'r')
                    else:
                        h264_decoder = None

                if h264_decoder is not None and not decoder.test_mode:
                    try:
                        packet = av.Packet(h264_bytes)
                        frames = h264_decoder.decode(packet)
                        for frame in frames:
                            img = frame.to_ndarray(format='bgr24')
                            decoder.decode_ok += 1
                            fps_counter += 1

                            now = time.time()
                            if now - last_fps_time >= 1.0:
                                display_fps = fps_counter / (now - last_fps_time)
                                fps_counter = 0
                                last_fps_time = now

                            display_img = cv2.resize(
                                img,
                                (decoder.width * decoder.display_scale,
                                 decoder.height * decoder.display_scale),
                                interpolation=cv2.INTER_NEAREST
                            )
                            draw_overlay(display_img, display_fps,
                                         decoder.crosshair_offset_x,
                                         decoder.crosshair_offset_y,
                                         decoder.crosshair_width)
                            cv2.imshow(win_name, display_img)
                            cv2.waitKey(1)
                    except Exception:
                        pass
                elif decoder.test_mode:
                    decoder.decode_ok += 1
                    fps_counter += 1
                    now = time.time()
                    if now - last_fps_time >= 1.0:
                        display_fps = fps_counter / (now - last_fps_time)
                        fps_counter = 0
                        last_fps_time = now
            else:
                time.sleep(0.001)

            now = time.time()
            if now - decoder.last_stat_time >= 1.0:
                stat_fps = decoder.decode_ok / (now - decoder.last_stat_time) if decoder.decode_ok > 0 else 0.0
                print(f"FPS={stat_fps:.2f} decoded={decoder.decoded_frames} "
                      f"packets={decoder.received_packets} inflight={len(decoder.reassembler.buffers)} "
                      f"ready={len(decoder.sequencer.output_queue)} drops={decoder.dropped_frames}")
                decoder.decode_ok = 0
                decoder.last_stat_time = now

    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()