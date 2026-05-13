#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace awakening {

class VideoSaver {
public:
    // 只需要路径初始化
    VideoSaver(const std::string& filename): save_path(filename), is_opened(false) {}

    ~VideoSaver() {
        close();
    }

    // 写入一帧，如果是第一帧则自动初始化 VideoWriter
    bool write_frame(
        const cv::Mat& frame,
        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        double fps = 30.0
    ) {
        if (frame.empty())
            return false;

        // 第一帧时初始化 VideoWriter
        if (!is_opened) {
            writer.open(save_path, codec, fps, frame.size(), frame.channels() == 3);
            is_opened = writer.isOpened();
            if (!is_opened)
                return false;
        }

        writer.write(frame);
        return true;
    }

    bool opened() const {
        return is_opened;
    }

    void close() {
        if (is_opened) {
            writer.release();
            is_opened = false;
        }
    }

private:
    std::string save_path;
    cv::VideoWriter writer;
    bool is_opened;
};

} // namespace awakening