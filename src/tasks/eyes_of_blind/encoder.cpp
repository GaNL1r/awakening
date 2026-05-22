#include "encoder.hpp"
#include "utils/logger.hpp"
#include "image_preprocessor.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <stdexcept>
#include <vector>

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video-info.h>
#include <gst/gst.h>
#include <yaml-cpp/node/node.h>

namespace awakening::eyes_of_blind {

// 全局 GF(2^8) 表（生成元 0x03）
static uint8_t gf_exp[512];
static uint8_t gf_log[256];
static bool gf_ready = false;

static void init_gf() {
    if (gf_ready) return;
    gf_ready = true;
    int x = 1;
    for (int i = 0; i < 255; ++i) {
        gf_exp[i] = x;
        gf_exp[i + 255] = x;
        gf_log[x] = i;
        x <<= 1;
        if (x & 0x100) x ^= 0x11d;
    }
    gf_log[0] = 0; // 未使用
}

static uint8_t gf_mul(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) return 0;
    return gf_exp[gf_log[a] + gf_log[b]];
}

// RS 编码：k 个数据分片，生成 r 个冗余包
// 矩阵：编码矩阵为范德蒙矩阵，第 j 行 (0..r-1) 系数为 (1+i)^(j)，i 从 0..k-1
static std::vector<std::vector<uint8_t>> generate_rs_fec(
    const std::vector<std::vector<uint8_t>>& fragments,
    int k, int r)
{
    init_gf();
    std::vector<std::vector<uint8_t>> fec(r, std::vector<uint8_t>(PAYLOAD_SIZE, 0));
    for (int j = 0; j < r; ++j) {
        for (int i = 0; i < k; ++i) {
            uint8_t coeff = 1;
            // 计算 (i+1)^j
            for (int p = 0; p < j; ++p) coeff = gf_mul(coeff, i + 1);
            for (size_t b = 0; b < fragments[i].size(); ++b) {
                fec[j][b] ^= gf_mul(fragments[i][b], coeff);
            }
        }
    }
    return fec;
}

struct Encoder::Impl {
    struct Params {
        int out_w {}, out_h {}, fps {};
        int target_bitrate {};
        int max_packets_per_sec = 30;
        int raid_redundancy = 2;   
        int p_redundancy = 1;   
        bool test_mode = false;

        void load(const YAML::Node& config) {
            out_w = config["output_w"].as<int>();
            out_h = config["output_h"].as<int>();
            fps = config["fps"].as<int>();
            target_bitrate = config["target_bitrate"].as<int>();
            raid_redundancy = config["raid_redundancy"].as<int>(2);
            p_redundancy   = config["p_redundancy"].as<int>(1);
            test_mode = config["test_mode"].as<bool>(false);
            if (config["max_packets_per_sec"])
                max_packets_per_sec = config["max_packets_per_sec"].as<int>();

            if (max_packets_per_sec <= 0)
                max_packets_per_sec = 30;
        }
    } params_;

    struct TokenBucket {
        double tokens = 0.0;
        double rate = 30.0;
        double capacity = 60.0;
        int64_t last_ns = 0;

        static int64_t now_ns() {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch()
            )
                .count();
        }

        void init(double r, double cap) {
            rate = r;
            capacity = cap;
            tokens = cap;
            last_ns = 0;
        }

        bool consume(double n = 1.0) {
            int64_t now = now_ns();

            if (last_ns == 0)
                last_ns = now;

            double dt = (now - last_ns) / 1e9;
            tokens = std::min(capacity, tokens + dt * rate);
            last_ns = now;

            if (tokens < n)
                return false;

            tokens -= n;
            return true;
        }

        bool try_consume_bulk(int n) {
            // 先计算当前可用令牌（不修改任何状态）
            int64_t now = now_ns();
            int64_t last = last_ns ? last_ns : now;
            double dt = (now - last) / 1e9;
            double current = std::min(capacity, tokens + dt * rate);

            if (current < n) return false;   // 不够

            // 够，则正式更新令牌桶
            if (last_ns == 0) last_ns = now;
            dt = (now - last_ns) / 1e9;
            tokens = std::min(capacity, tokens + dt * rate) - n;
            last_ns = now;
            return true;
        }
    };

    TokenBucket bucket_;

    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    GstElement* appsink_ = nullptr;
    GstBus* bus_ = nullptr;

    std::mutex pkg_mutex_;
    std::condition_variable pkg_cv_;
    std::deque<BlindSend> pkg_queue_;
    size_t max_queue_packets_ = 0;

    uint32_t frame_id_ = 0;
    std::unique_ptr<ImagePreprocessor> preprocessor_;
    std::vector<uint8_t> gop_buffer_;
    int p_frames_in_gop_ = 0;
    static const int MAX_P_FRAMES = 9;
    int gop_idr_frame_id_ = 0;
    uint32_t gop_send_id_ = 0;
    std::chrono::steady_clock::time_point last_send_time_;

    // 合并 P 帧 相关
    std::vector<uint8_t> merge_buffer_;          // 合并后的原始数据
    std::vector<size_t> merge_frame_lengths_;    // 每个子帧的长度
    int merge_frame_count_ = 0;                  // 当前缓存的子帧数量
    static constexpr int MERGE_MAX_FRAMES = 4;   // 最多合并 4 个 P 帧

    bool test_mode_;
    int test_frame_counter_ = 0;


    std::chrono::steady_clock::time_point last_merge_time_;
    std::atomic<bool> pipeline_alive_{true}; 

    // 声明辅助函数
    void flush_merged_pframes() {
        if (merge_frame_count_ == 0) return;

        if (merge_frame_count_ == 1) {
            // 只有一帧，当作普通单 P 帧发送（带 FEC）
            uint8_t flags = 0;   // 无合并标志
            send_encoded_frame(merge_buffer_.data(), merge_frame_lengths_[0],
                            frame_id_++, flags);
        } else {
            // 多帧合并
            std::vector<uint8_t> payload;
            payload.push_back(static_cast<uint8_t>(merge_frame_count_));
            for (size_t len : merge_frame_lengths_) {
                uint16_t l = static_cast<uint16_t>(len);
                payload.push_back(l & 0xFF);
                payload.push_back((l >> 8) & 0xFF);
            }
            payload.insert(payload.end(), merge_buffer_.begin(), merge_buffer_.end());

            uint8_t flags = FLAG_MERGED | ((merge_frame_count_ & 0x1F) << 3);
            send_encoded_frame(payload.data(), payload.size(), frame_id_++, flags);
        }

        merge_buffer_.clear();
        merge_frame_lengths_.clear();
        merge_frame_count_ = 0;
    }

    void send_encoded_frame(const uint8_t* data, size_t size, uint32_t frame_id, uint8_t flags) {
        const size_t frag_payload = PAYLOAD_SIZE;
        uint16_t frag_count = (size + frag_payload - 1) / frag_payload;
        if (frag_count == 0) frag_count = 1;

        // 数据分片（所有分片填充到 PAYLOAD_SIZE 以保证 RS 编码正确）
        std::vector<std::vector<uint8_t>> fragments(frag_count);
        for (uint16_t i = 0; i < frag_count; ++i) {
            size_t offset = i * frag_payload;
            size_t copy   = std::min(frag_payload, size - offset);
            fragments[i].resize(PAYLOAD_SIZE, 0);
            std::memcpy(fragments[i].data(), data + offset, copy);
        }

        int r = 0;
        bool is_idr = (flags & FLAG_KEYFRAME);
        if (is_idr && frag_count >= 2) {
            r = params_.raid_redundancy;
            if (r < 2) r = 2;
        } else {
            r = params_.p_redundancy;
            if (r < 0) r = 0;
        }

        uint16_t total_frags = frag_count + r;

        if (!bucket_.try_consume_bulk(total_frags)) {
            AWAKENING_WARN("Dropping frame#{}: rate limit", frame_id);
            return;
        }

        std::vector<BlindSend> packets;

        // 编码 r 值到 flags 的 [7:3] 位（当 FLAG_MERGED 未置位时）
        uint8_t flags_with_r = flags;
        if (!(flags & FLAG_MERGED)) {
            flags_with_r |= ((r & 0x1F) << 3);
        }

        // 计算最后一个分片的有效大小
        size_t last_frag_size = size - (frag_count - 1) * frag_payload;

        // 发送数据分片（总是发送完整的 PAYLOAD_SIZE，包括填充零）
        for (uint16_t i = 0; i < frag_count; ++i) {
            BlindSend pkt{};
            pkt.header.frame_id     = frame_id;
            pkt.header.frag_idx     = i;
            pkt.header.frag_count   = total_frags;
            // 发送完整大小（包括填充），这样 RS 编解码一致
            pkt.header.payload_size = PAYLOAD_SIZE;
            pkt.header.frame_size   = static_cast<uint16_t>(size);
            pkt.header.flags        = flags_with_r;
            std::memcpy(pkt.data.data(), fragments[i].data(), PAYLOAD_SIZE);
            packets.push_back(pkt);
        }

        // 生成 r 个冗余包
        if (r > 0) {
            if (frag_count == 1) {
                // 单 P 帧：冗余包直接拷贝 r 份
                for (int j = 0; j < r; ++j) {
                    BlindSend fec{};
                    fec.header.frame_id     = frame_id;
                    fec.header.frag_idx     = frag_count + j;
                    fec.header.frag_count   = total_frags;
                    fec.header.payload_size = static_cast<uint16_t>(fragments[0].size());
                    fec.header.frame_size   = static_cast<uint16_t>(size);
                    fec.header.flags        = flags_with_r | FLAG_FEC_PACKET;
                    std::memcpy(fec.data.data(), fragments[0].data(), fragments[0].size());
                    packets.push_back(fec);
                }
            } else {
                // 多分片：使用 RS 编码生成 r 个冗余包
                auto fecs = generate_rs_fec(fragments, frag_count, r);
                for (int j = 0; j < r; ++j) {
                    BlindSend fec{};
                    fec.header.frame_id     = frame_id;
                    fec.header.frag_idx     = frag_count + j;
                    fec.header.frag_count   = total_frags;
                    fec.header.payload_size = PAYLOAD_SIZE;
                    fec.header.frame_size   = static_cast<uint16_t>(size);
                    fec.header.flags        = flags_with_r | FLAG_FEC_PACKET;
                    std::memcpy(fec.data.data(), fecs[j].data(), PAYLOAD_SIZE);
                    packets.push_back(fec);
                }
            }
        }

        // 入队（同前）
        {
            std::lock_guard<std::mutex> qlock(pkg_mutex_);
            for (auto& p : packets) {
                if (pkg_queue_.size() >= max_queue_packets_) pkg_queue_.pop_front();
                pkg_queue_.push_back(p);
            }
        }

        std::string fec_str = r > 0 ? ("RS(" + std::to_string(frag_count) + "+" + std::to_string(r) + ")") : "NO";
        AWAKENING_INFO("Frame#{} {} : {} bytes, data_frags={}, total_frags={}, FEC={}",
                    frame_id, is_idr ? "IDR" : "P", size, frag_count, total_frags, fec_str);
    }

    Impl(const YAML::Node& config) {
        params_.load(config);

        test_mode_ = params_.test_mode;
        if (test_mode_) {
            // 不初始化 GStreamer
            bucket_.init(params_.max_packets_per_sec, params_.max_packets_per_sec);
            max_queue_packets_ = params_.max_packets_per_sec * 8;
            preprocessor_ = std::make_unique<ImagePreprocessor>(config);
            return;
        }

        bucket_.init(params_.max_packets_per_sec, params_.max_packets_per_sec);

        max_queue_packets_ = params_.max_packets_per_sec * 8;

        preprocessor_ = std::make_unique<ImagePreprocessor>(config);

        initialize_gstreamer();
    }

    ~Impl() {
        shutdown_gstreamer();
    }

    void initialize_gstreamer() {
        gst_init(nullptr, nullptr);

        pipeline_ = gst_pipeline_new("encoder_pipe");
        appsrc_ = gst_element_factory_make("appsrc", "source");
        appsink_ = gst_element_factory_make("appsink", "sink");

        GstElement* convert = gst_element_factory_make("videoconvert", "convert");
        GstElement* encoder = gst_element_factory_make("x264enc", "encoder");
        GstElement* parser = gst_element_factory_make("h264parse", "parser");

        if (!pipeline_ || !appsrc_ || !appsink_ || !convert || !encoder || !parser) {
            AWAKENING_ERROR("GStreamer element creation failed");
            return;
        }

        // 输入 caps 为 BGR
        GstCaps* caps_in = gst_caps_new_simple(
            "video/x-raw",
            "format", G_TYPE_STRING, "BGR",      
            "width", G_TYPE_INT, params_.out_w,
            "height", G_TYPE_INT, params_.out_h,
            "framerate", GST_TYPE_FRACTION, params_.fps, 1,
            nullptr
        );

        g_object_set(
            appsrc_,
            "caps", caps_in,
            "stream-type", 0,
            "format", GST_FORMAT_TIME,
            "is-live", TRUE,
            "block", TRUE,                         // 设为 TRUE，确保 pipeline 准备好
            "do-timestamp", TRUE,
            nullptr
        );
        gst_caps_unref(caps_in);


        g_object_set(
            encoder,
            "bitrate",          params_.target_bitrate,   // 目标码率 (kbps)
            "speed-preset",     9,                         
            "tune",             0x00000001,                
            "bframes",          0,
            "ref",              1,
            "key-int-max",      30,
            "rc-lookahead",     0,
            "sync-lookahead",   0,
            "sliced-threads",   FALSE,
            "byte-stream",      TRUE,
            "aud",              TRUE,
            "vbv-buf-capacity", 100,    // ms
            "option-string",            
            "repeat-headers=1:"
            "vbv-maxrate=50:"           // kbps
            "force-cfr=1:"
            "scenecut=0:"
            "open-gop=0:"
            "me=hex:"
            "me-range=16:"
            "subme=7:"
            "trellis=2:"
            "deblock=0,0:"
            // "aq-mode=2:"
            // "aq-strength=1.2:"
            "psy-rd=0.4,0.0",
            nullptr
        );
        
        // h264parse 配置
        // config-interval=-1: 仅在 IDR 帧前插入 SPS/PPS（默认行为）
        g_object_set(parser, "config-interval", -1, "disable-passthrough", TRUE, nullptr);

        // 输出 caps：H.264 byte-stream（Au 对齐）
        GstCaps* h264_caps = gst_caps_new_simple(
            "video/x-h264",
            "stream-format", G_TYPE_STRING, "byte-stream",
            "alignment", G_TYPE_STRING, "au",
            nullptr
        );

        g_object_set(
            appsink_,
            "caps", h264_caps,
            "max-buffers", 5,
            "drop", FALSE,
            "emit-signals", FALSE,
            "sync", FALSE,
            nullptr
        );
        gst_caps_unref(h264_caps);
        
        // 添加元素到 pipeline
        gst_bin_add_many(GST_BIN(pipeline_), appsrc_, convert, encoder, parser, appsink_, nullptr);

        // 链接元素
        if (!gst_element_link_many(appsrc_, convert, encoder, parser, appsink_, nullptr)) {
            AWAKENING_ERROR("GStreamer pipeline link failed");
            return;
        }
        GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            AWAKENING_ERROR("Failed to start pipeline");
            return;
        }


        // 修改 bus watch：捕获 ERROR 和 EOS
        bus_ = gst_element_get_bus(pipeline_);
        gst_bus_add_watch(bus_, [](GstBus* bus, GstMessage* msg, gpointer data) -> gboolean {
            auto* impl = static_cast<Impl*>(data);
            switch (GST_MESSAGE_TYPE(msg)) {
                case GST_MESSAGE_ERROR: {
                    GError* err;
                    gchar* debug;
                    gst_message_parse_error(msg, &err, &debug);
                    AWAKENING_ERROR("GStreamer ERROR: {} ({})", err->message, debug ? debug : "");
                    g_error_free(err);
                    g_free(debug);
                    impl->pipeline_alive_.store(false);   // 标记管线失效
                    break;
                }
                case GST_MESSAGE_EOS: {
                    AWAKENING_WARN("GStreamer EOS received, marking pipeline as dead.");
                    impl->pipeline_alive_.store(false);
                    break;
                }
                case GST_MESSAGE_WARNING: {
                    // 保留原有警告打印
                    GError* warn;
                    gchar* debug;
                    gst_message_parse_warning(msg, &warn, &debug);
                    AWAKENING_WARN("GStreamer WARNING: {} ({})", warn->message, debug ? debug : "");
                    g_error_free(warn);
                    g_free(debug);
                    break;
                }
                default: break;
            }
            return TRUE;
        }, this);   // 传递 this 指针
    }
    void shutdown_gstreamer() {
        if (!pipeline_)
            return;

        gst_element_set_state(pipeline_, GST_STATE_NULL);

        if (bus_) {
            gst_object_unref(bus_);
            bus_ = nullptr;
        }

        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        appsrc_ = nullptr;
        appsink_ = nullptr;
    }

    // ---------- 推流 ----------
    void push_frame_to_gstreamer(const cv::Mat& frame) {
        if (!appsrc_ || frame.empty()) return;

        cv::Mat cont = frame.isContinuous() ? frame : frame.clone();
        size_t size = cont.total() * cont.elemSize();   // 紧密打包总字节数

        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
            std::memcpy(map.data, cont.data, size);
            gst_buffer_unmap(buffer, &map);

            // ---------- 关键：添加 video meta，指定 BGR 紧密步幅 ----------
            GstVideoInfo info;
            gst_video_info_init(&info);
            gst_video_info_set_format(&info, GST_VIDEO_FORMAT_BGR, 
                                    params_.out_w, params_.out_h);
            // 紧密打包：每行步幅 = 宽度 * 3 字节
            info.stride[0] = params_.out_w * 3;
            info.offset[0] = 0;

            gst_buffer_add_video_meta_full(buffer, GST_VIDEO_FRAME_FLAG_NONE,
                                        GST_VIDEO_FORMAT_BGR,
                                        params_.out_w, params_.out_h,
                                        1,              // BGR 只有一个平面
                                        info.offset, info.stride);
            // ------------------------------------------------

            GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);
            if (ret != GST_FLOW_OK) {
                AWAKENING_WARN("gst_app_src_push_buffer failed: %d", ret);
            }
        } else {
            gst_buffer_unref(buffer);
        }
    }

    void pull_stream_and_packetize() {
        if (test_mode_) return;
        static int pull_count = 0;
        // AWAKENING_INFO("pull_stream_and_packetize called (#{})", ++pull_count);
        if (appsink_ && gst_app_sink_is_eos(GST_APP_SINK(appsink_))) {
            pipeline_alive_.store(false);
            return;
        }

        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink_), 0);
        if (!sample) {
            // 每秒最多打印一次
            static int no_sample_count = 0;
            static auto last_warn = std::chrono::steady_clock::now();
            no_sample_count++;
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_warn).count() >= 1) {
                if (no_sample_count > 0) {
                    AWAKENING_DEBUG("No sample for {} polls in last second", no_sample_count);
                }
                no_sample_count = 0;
                last_warn = now;
            }
            if (merge_frame_count_ > 0) {
                auto now = std::chrono::steady_clock::now();
                auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_merge_time_).count();
                if (dur > 100) {   // 100ms 无新 P 帧
                    flush_merged_pframes();
                    last_merge_time_ = now;
                }
            }
            return;
        }

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        if (!buffer) { gst_sample_unref(sample); return; }

        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            gst_sample_unref(sample);
            return;
        }

        const uint8_t* data = map.data;
        const size_t  frame_size = map.size;

        // 检测是否为 IDR
        bool is_idr = false;
        for (size_t i = 0; i + 4 < frame_size; ++i) {
            if (data[i]==0x00 && data[i+1]==0x00 && data[i+2]==0x00 && data[i+3]==0x01 && i+4<frame_size) {
                if ((data[i+4] & 0x1F) == 5) { is_idr = true; break; }
            }
        }

        // IDR 帧：先清空当前合并缓冲区
        if (is_idr && merge_frame_count_ > 0) {
            flush_merged_pframes();
        }
        
        if (!is_idr) {
            // ---- P 帧处理 ----
            // 计算如果加入当前帧，新合并包的 payload 大小
            size_t new_count = merge_frame_count_ + 1;
            size_t header_size = 1 + new_count * 2;   // 1字节帧数 + 2*N个长度
            size_t total_data = merge_buffer_.size() + frame_size;

            if (new_count <= MERGE_MAX_FRAMES && (header_size + total_data) <= PAYLOAD_SIZE) {
                // 可以合并
                merge_buffer_.insert(merge_buffer_.end(), data, data + frame_size);
                merge_frame_lengths_.push_back(frame_size);
                merge_frame_count_ = new_count;
            } else {
                // 无法继续合并，先把旧合并帧发出（如果有）
                if (merge_frame_count_ > 0) {
                    flush_merged_pframes();
                }
                // 然后判断当前帧是否太大（连单独一个包都放不下）
                size_t single_header = 1 + 2;   // 单一帧的合并结构：1字节帧数+2字节长度
                if (single_header + frame_size <= PAYLOAD_SIZE) {
                    // 当前帧可以作为新的合并帧的开始（暂时只有它自己）
                    merge_buffer_.assign(data, data + frame_size);
                    merge_frame_lengths_.push_back(frame_size);
                    merge_frame_count_ = 1;
                } else {
                    // 单帧已经超过一个包，必须立即发送（带 FEC 保护）
                    uint8_t flags = 0;
                    send_encoded_frame(data, frame_size, frame_id_++, flags);
                }
            }
        } else {
            // IDR 帧直接发送（不带合并标志）
            uint8_t flags = FLAG_KEYFRAME;
            send_encoded_frame(data, frame_size, frame_id_++, flags);
        }

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
    }

    cv::Mat preprocess(const cv::Mat& frame) {
        if (!preprocessor_)
            return frame;

        return preprocessor_->process(frame);
    }
    
    void push_frame(const cv::Mat& frame) {
        if (test_mode_) {
            // 生成固定大小的测试帧，内容为 0,1,2...
            constexpr int TEST_SIZE = 100;   // 可调整
            std::vector<uint8_t> data(TEST_SIZE);
            for (int i = 0; i < TEST_SIZE; ++i)
                data[i] = static_cast<uint8_t>(test_frame_counter_ + i);
            test_frame_counter_++;

            uint8_t flags = 0;
            // 每 10 帧设置一个 IDR 标志方便观察
            if (test_frame_counter_ % 10 == 0)
                flags = FLAG_KEYFRAME;

            send_encoded_frame(data.data(), data.size(), frame_id_++, flags);
            return;
        }
        if (frame.empty())
            return;

        auto img = preprocess(frame);
        push_frame_to_gstreamer(img);
    }

    bool try_pop_packet(BlindSend& out) {
        std::lock_guard<std::mutex> lock(pkg_mutex_);
        if (pkg_queue_.empty())
            return false;

        out = pkg_queue_.front();
        pkg_queue_.pop_front();
        pkg_cv_.notify_one();
        return true;
    }
};

Encoder::Encoder(const YAML::Node& config) {
    _impl = std::make_unique<Impl>(config);
}

Encoder::~Encoder() = default;

void Encoder::push_frame(const cv::Mat& frame) {
    _impl->push_frame(frame);
}

void Encoder::pull_and_packetize() {
    _impl->pull_stream_and_packetize();
}

bool Encoder::try_pop_packet(BlindSend& out) {
    return _impl->try_pop_packet(out);
}

bool Encoder::is_pipeline_alive() const {
    return _impl->pipeline_alive_.load();
}

} // namespace awakening::eyes_of_blind