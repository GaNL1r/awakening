#include "_rcl/node.hpp"
#include "ascii_banner.hpp"
#include "tasks/base/common.hpp"
#include "tasks/radar_detect/rmuc_2026_map.hpp"
#include "utils/common/type_common.hpp"
#include "utils/drivers/hik_camera.hpp"
#include "utils/drivers/video_player.hpp"
#include "utils/io/pcd_io.h"
#include "utils/signal_guard.hpp"
#include "utils/utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <chrono>
#include <memory>
#include <mutex>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <thread>
#include <vector>

using namespace awakening;
struct CameraTag {};
struct SerialTag {};
struct DetectTag {};
struct FrameTag {};
using CameraIO = IOPair<CameraTag, ImageFrame>;
using SerialIO = IOPair<SerialTag, std::vector<uint8_t>>;
using CommonFrameIo = IOPair<FrameTag, CommonFrame>;
using DetIo = IOPair<DetectTag, std::vector<radar_detect::Cars>>;
int main(int argc, char** argv) {
    print_banner();
    auto& signal = utils::SignalGuard::instance();
    logger::init(spdlog::level::trace);

    auto get_arg = [&](int i) -> std::optional<std::string> {
        if (i < argc) {
            AWAKENING_INFO("get args {} ", std::string(argv[i]));
            return std::make_optional(std::string(argv[i]));
        }
        return std::nullopt;
    };

    std::string config_path;
    auto first_arg = get_arg(1);
    if (first_arg) {
        config_path = first_arg.value();
    } else {
        return 1;
    }
    int cal_type = 0;
    auto second_arg = get_arg(2);
    if (second_arg) {
        cal_type = std::stoi(second_arg.value());
    }
    auto config = YAML::LoadFile(config_path);
    if (cal_type == 1) {
        radar_detect::RMUC2026Map map(config["map"], radar_detect::SelfColor::RED);
        map.edit();
        map.dump_yaml("guess.yaml");
    } else {
        Scheduler s;
        std::unique_ptr<VideoPlayer> video;
        std::unique_ptr<HikCamera> camera;
        auto camera_config = config["camera"];
        CameraInfo camera_info;
        camera_info.load(camera_config["camera_info"]);
        std::string camera_type = camera_config["type"].as<std::string>();
        camera_type = utils::to_upper(camera_type);
        if (camera_type == "VIDEO") {
            video = std::make_unique<VideoPlayer>(camera_config["video"], s);
        } else {
            camera = std::make_unique<HikCamera>(camera_config["hik_camera"], s);
        }
        if (camera) {
            camera->init();
            if (!camera->running_) {
                return 0;
            }
        }
        cv::namedWindow("Image", cv::WINDOW_NORMAL);
        cv::resizeWindow("Image", 640, 480);
        if (cal_type == 2) {
            s.register_task<CameraIO>("save_img", [&](CameraIO::second_type&& f) {
                cv::Mat img = f.src_img;
                cv::imshow("Image", img);
                auto key = cv::waitKey(1);
                if (key == 's') {
                    cv::imwrite("out.png", img);
                }
                return;
            });
        } else if (cal_type == 3) {
            struct BBoxState {
                cv::Mat current_img;
                cv::Rect current_box;
                bool drawing = false;
                bool has_box = false;
                cv::Point start_pt;

                std::vector<cv::Rect> boxes;
            };

            auto state = std::make_shared<BBoxState>();

            cv::setMouseCallback(
                "Image",
                [](int event, int x, int y, int, void* userdata) {
                    auto* s = static_cast<BBoxState*>(userdata);

                    if (event == cv::EVENT_LBUTTONDOWN) {
                        s->drawing = true;
                        s->start_pt = cv::Point(x, y);
                        s->current_box = cv::Rect(x, y, 0, 0);
                    }

                    else if (event == cv::EVENT_MOUSEMOVE && s->drawing)
                    {
                        int x0 = std::min(s->start_pt.x, x);
                        int y0 = std::min(s->start_pt.y, y);
                        int w = std::abs(x - s->start_pt.x);
                        int h = std::abs(y - s->start_pt.y);

                        s->current_box = cv::Rect(x0, y0, w, h);
                    }

                    else if (event == cv::EVENT_LBUTTONUP)
                    {
                        s->drawing = false;

                        int x0 = std::min(s->start_pt.x, x);
                        int y0 = std::min(s->start_pt.y, y);
                        int w = std::abs(x - s->start_pt.x);
                        int h = std::abs(y - s->start_pt.y);

                        s->current_box = cv::Rect(x0, y0, w, h);

                        if (w > 5 && h > 5) {
                            s->boxes.push_back(s->current_box);

                            AWAKENING_INFO(
                                "BBox: x={} y={} w={} h={}",
                                s->current_box.x,
                                s->current_box.y,
                                s->current_box.width,
                                s->current_box.height
                            );
                        }
                    }
                },
                state.get()
            );

            s.register_task<CameraIO>("get_bbox", [state](CameraIO::second_type&& f) {
                state->current_img = f.src_img.clone();

                cv::Mat show = state->current_img.clone();

                // 已确认 bbox
                for (auto& box: state->boxes) {
                    cv::rectangle(show, box, cv::Scalar(0, 255, 0), 2);
                }

                // 正在拖拽 bbox
                if (state->drawing) {
                    cv::rectangle(show, state->current_box, cv::Scalar(0, 0, 255), 2);
                }

                cv::imshow("Image", show);

                auto key = cv::waitKey(1);

                // 保存截图
                if (key == 's') {
                    cv::imwrite("bbox_image.png", state->current_img);

                    YAML::Node node;

                    for (size_t i = 0; i < state->boxes.size(); ++i) {
                        auto& b = state->boxes[i];

                        YAML::Node item;
                        item["x"] = b.x;
                        item["y"] = b.y;
                        item["w"] = b.width;
                        item["h"] = b.height;

                        node["bboxes"].push_back(item);
                    }

                    std::ofstream fout("bbox.yaml");
                    fout << node;

                    AWAKENING_INFO("Saved bbox_image.png and bbox.yaml");
                }

                // 清空
                if (key == 'c') {
                    state->boxes.clear();
                    AWAKENING_INFO("Clear bboxes");
                }

                return;
            });
        }
        if (camera) {
            camera->start<CameraTag>("hik");
        }
        if (video) {
            video->start<CameraTag>("video");
        }
        s.build();
        s.run();
        utils::SignalGuard::spin(std::chrono::milliseconds(1000));
        s.stop();
    }

    return 0;
}