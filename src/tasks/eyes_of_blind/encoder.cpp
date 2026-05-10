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

struct Encoder::Impl {
    struct Params {
        int out_w {}, out_h {}, fps {};
        int target_bitrate {};
        int max_packets_per_sec = 30;

        void load(const YAML::Node& config) {
            out_w = config["output_w"].as<int>();
            out_h = config["output_h"].as<int>();
            fps = config["fps"].as<int>();
            target_bitrate = config["target_bitrate"].as<int>();

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

    Impl(const YAML::Node& config) {
        params_.load(config);

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

        // ----- 低延迟配置（延迟低，清晰度稍差）-----
        g_object_set(
            encoder,
            "bitrate",          params_.target_bitrate,   // 目标码率 (kbps)
            "speed-preset",     9,                         
            "tune",             0x00000001,                
            "bframes",          0,
            "ref",              2,
            "key-int-max",      8,
            "rc-lookahead",     0,
            "sync-lookahead",   0,
            "sliced-threads",   FALSE,
            "byte-stream",      TRUE,
            "aud",              TRUE,
            "option-string",
            "repeat-headers=1:"
            "force-cfr=1:"
            "scenecut=0:"
            "open-gop=0:"
            "b-adapt=2:"
            "me=hex:"
            "me-range=32:"
            "subme=7:"
            "trellis=2:"
            "deblock=0,0:"
            "aq-mode=2:"
            "aq-strength=1.2:"
            "psy-rd=0.4,0.0",
            nullptr
        );

        // h264parse 配置
        // config-interval=-1: 仅在 IDR 帧前插入 SPS/PPS（默认行为）
        g_object_set(parser, "config-interval", 1, "disable-passthrough", TRUE, nullptr);

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


        bus_ = gst_element_get_bus(pipeline_);
        gst_bus_add_watch(bus_, [](GstBus* bus, GstMessage* msg, gpointer data) -> gboolean {
            switch (GST_MESSAGE_TYPE(msg)) {
                case GST_MESSAGE_ERROR: {
                    GError* err;
                    gchar* debug;
                    gst_message_parse_error(msg, &err, &debug);
                    AWAKENING_ERROR("GStreamer ERROR: {} ({})", err->message, debug ? debug : "");
                    g_error_free(err);
                    g_free(debug);
                    break;
                }
                case GST_MESSAGE_WARNING: {
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
        }, nullptr);
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
        static int pull_count = 0;
        // AWAKENING_INFO("pull_stream_and_packetize called (#{})", ++pull_count);

        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink_), 0);
        if (!sample) {
            // 不刷屏，每秒最多打印一次
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
        const size_t   size = map.size;

        // ----- 判断是否为 IDR (NAL type 5) -----
        bool is_idr = false;
        for (size_t i = 0; i + 4 < size; ++i) {
            if (data[i]==0x00 && data[i+1]==0x00 &&
                data[i+2]==0x00 && data[i+3]==0x01 && i+4<size) {
                if ((data[i+4] & 0x1F) == 5) { is_idr = true; break; }
            }
        }
        
        // ----- 分片准备 -----
        const size_t frag_payload = PAYLOAD_SIZE;
        uint16_t frag_count = (size + frag_payload - 1) / frag_payload;
        if (frag_count == 0) frag_count = 1;

        std::vector<std::vector<uint8_t>> fragments(frag_count);
        for (uint16_t i = 0; i < frag_count; ++i) {
            size_t offset = i * frag_payload;
            size_t copy   = std::min(frag_payload, size - offset);
            fragments[i].assign(data + offset, data + offset + copy);
        }

        // ----- FEC 决策 -----
        bool use_fec = ( is_idr ||frag_count >= 2);
        uint16_t total_frags = use_fec ? (frag_count + 1) : frag_count;

        uint32_t frame_id = frame_id_++;

        // ----- 令牌批量申请 -----
        if (!bucket_.try_consume_bulk(total_frags)) {
            // 令牌不足，整帧丢弃，直接返回（解绑缓冲区后返回）
            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);
            return;
        }
        // 日志
        const char* frame_type = is_idr ? "IDR" : "P";
        AWAKENING_INFO("Frame#{} {} : {} bytes, data_frags={}, total_frags={}, FEC={}",
                    frame_id, frame_type, size, frag_count, total_frags,
                    use_fec ? "YES" : "NO");

        // ----- 发送数据分片（直接入队，不再单独扣令牌）-----
        std::vector<BlindSend> packets;
        for (uint16_t i = 0; i < frag_count; ++i) {
            BlindSend pkt{};
            pkt.header.frame_id     = frame_id;
            pkt.header.frag_idx     = i;
            pkt.header.frag_count   = total_frags;
            pkt.header.payload_size = static_cast<uint16_t>(fragments[i].size());
            pkt.header.frame_size   = static_cast<uint16_t>(size);
            pkt.header.flags        = is_idr ? FLAG_KEYFRAME : 0;
            std::memcpy(pkt.data.data(), fragments[i].data(), fragments[i].size());
            packets.push_back(pkt);
        }

        // ----- 生成 FEC 冗余包（若需要）-----
        if (use_fec) {
            BlindSend fec_pkt{};
            fec_pkt.header.frame_id     = frame_id;
            fec_pkt.header.frag_idx     = frag_count;           // 索引固定为分片数
            fec_pkt.header.frag_count   = total_frags;
            fec_pkt.header.frame_size   = static_cast<uint16_t>(size);
            fec_pkt.header.flags        = FLAG_FEC_PACKET | (is_idr ? FLAG_KEYFRAME : 0);

            std::vector<uint8_t> fec_data(PAYLOAD_SIZE, 0);
            if (frag_count == 1) {
                fec_data = fragments[0];                 // 1+1：复制内容
            } else {
                for (uint16_t i = 0; i < frag_count; ++i)
                    for (size_t j = 0; j < fragments[i].size(); ++j)
                        fec_data[j] ^= fragments[i][j];
            }
            fec_pkt.header.payload_size = (frag_count == 1) ? fragments[0].size() : PAYLOAD_SIZE;
            std::memcpy(fec_pkt.data.data(), fec_data.data(), fec_pkt.header.payload_size);
            packets.push_back(fec_pkt);
        }

        // 所有包一次性入队（队列满时丢弃最旧的包）
        {
            std::lock_guard<std::mutex> qlock(pkg_mutex_);
            for (auto& pkt : packets) {
                if (pkg_queue_.size() >= max_queue_packets_) {
                    pkg_queue_.pop_front();      // 队列溢出，丢弃最旧包
                }
                pkg_queue_.push_back(pkt);
            }
        }

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
    }

    // void send_as_frame(const uint8_t* data, size_t size, bool is_keyframe) {
    //     const size_t frag_payload = PAYLOAD_SIZE;
    //     const uint16_t frag_count = (size + frag_payload - 1) / frag_payload;
    //     uint32_t current_frame_id = frame_id_++;
        
    //     AWAKENING_INFO("  Frame#{}: {} bytes -> {} fragments", 
    //         current_frame_id, size, frag_count);
        
    //     for (uint16_t i = 0; i < frag_count; ++i) {
    //         if (!bucket_.consume()) {
    //             AWAKENING_WARN("  Token bucket exhausted at frag {}/{}", i, frag_count);
    //             break;
    //         }
            
    //         BlindSend pkt{};
    //         pkt.header.frame_id = current_frame_id;
    //         pkt.header.frag_idx = i;
    //         pkt.header.frag_count = frag_count;
    //         size_t offset = i * frag_payload;
    //         size_t remain = size - offset;
    //         size_t copy_size = std::min(remain, frag_payload);
    //         pkt.header.payload_size = static_cast<uint16_t>(copy_size);
    //         pkt.header.flags = is_keyframe ? 0x01 : 0x00;
    //         std::memcpy(pkt.data.data(), data + offset, copy_size);
            
    //         {
    //             std::lock_guard<std::mutex> qlock(pkg_mutex_);
    //             if (pkg_queue_.size() < max_queue_packets_) {
    //                 pkg_queue_.push_back(pkt);
    //             }
    //         }
    //     }
    // }

    cv::Mat preprocess(const cv::Mat& frame) {
        if (!preprocessor_)
            return frame;

        return preprocessor_->process(frame);
    }
    
    void push_frame(const cv::Mat& frame) {
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

} // namespace awakening::eyes_of_blind