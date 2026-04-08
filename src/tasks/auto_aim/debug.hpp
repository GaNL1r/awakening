#pragma once
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "utils/buffer.hpp"
#include "utils/common/type_common.hpp"
#include <mutex>
#include <opencv2/core/types.hpp>
#include <utility>
#include <vector>
namespace awakening::auto_aim {
struct AutoAimDebugCtx {
    CameraInfo camera_info_;
    mutable std::mutex armors_mutex;
    Armors armors_buffer;
    mutable std::mutex armor_target_mutex;
    ArmorTarget armor_target_buffer;
    mutable std::mutex img_frame_mutex;
    ImageFrame img_frame_buffer;
    mutable std::mutex expanded_mutex;
    cv::Rect expanded_buffer;
    mutable std::mutex avg_latency_ms_mutex;
    double avg_latency_ms_buffer;
    mutable std::mutex gimbal_cmd_mutex;
    GimbalCmd gimbal_cmd_buffer;
    mutable std::mutex fsm_state_mutex;
    AutoAimFsm fsm_state_buffer;
    mutable std::mutex gimbal_yaw_pitch_mutex;
    std::pair<double, double> gimbal_yaw_pitch_buffer;
    mutable std::mutex bullet_positions_mutex;
    std::vector<Vec3> bullet_positions_buffer;
    void set_armors(const Armors& armors) {
        std::lock_guard<std::mutex> lock(armors_mutex);
        armors_buffer = armors;
    }
    Armors armors() const noexcept {
        std::lock_guard<std::mutex> lock(armors_mutex);
        return armors_buffer;
    }
    void set_armor_target(const ArmorTarget& armor_target) {
        std::lock_guard<std::mutex> lock(armor_target_mutex);
        armor_target_buffer = armor_target;
    }
    ArmorTarget armor_target() const noexcept {
        std::lock_guard<std::mutex> lock(armor_target_mutex);
        return armor_target_buffer;
    }
    void set_img_frame(const ImageFrame& img_frame) {
        std::lock_guard<std::mutex> lock(img_frame_mutex);
        img_frame_buffer = img_frame;
    }
    ImageFrame img_frame() const noexcept {
        std::lock_guard<std::mutex> lock(img_frame_mutex);
        return img_frame_buffer;
    }
    CameraInfo camera_info() const noexcept {
        return camera_info_;
    }
    void set_expanded(const cv::Rect& expanded) {
        std::lock_guard<std::mutex> lock(expanded_mutex);
        expanded_buffer = expanded;
    }
    cv::Rect expanded() const noexcept {
        std::lock_guard<std::mutex> lock(expanded_mutex);
        return expanded_buffer;
    }
    void set_avg_latency_ms(double ms) noexcept {
        std::lock_guard<std::mutex> lock(avg_latency_ms_mutex);
        avg_latency_ms_buffer = ms;
    }
    double avg_latency_ms() const noexcept {
        std::lock_guard<std::mutex> lock(avg_latency_ms_mutex);
        return avg_latency_ms_buffer;
    }
    void set_gimbal_cmd(const GimbalCmd& gimbal_cmd) noexcept {
        std::lock_guard<std::mutex> lock(gimbal_cmd_mutex);
        gimbal_cmd_buffer = gimbal_cmd;
    }
    GimbalCmd gimbal_cmd() const noexcept {
        std::lock_guard<std::mutex> lock(gimbal_cmd_mutex);
        return gimbal_cmd_buffer;
    }
    void set_fsm_state(AutoAimFsm fsm_state) noexcept {
        std::lock_guard<std::mutex> lock(fsm_state_mutex);
        fsm_state_buffer = fsm_state;
    }
    AutoAimFsm fsm_state() const noexcept {
        std::lock_guard<std::mutex> lock(fsm_state_mutex);
        return fsm_state_buffer;
    }
    void set_gimbal_yaw_pitch(const std::pair<double, double>& ps) noexcept {
        std::lock_guard<std::mutex> lock(gimbal_yaw_pitch_mutex);
        gimbal_yaw_pitch_buffer = ps;
    }
    std::pair<double, double> gimbal_yaw_pitch() const noexcept {
        std::lock_guard<std::mutex> lock(gimbal_yaw_pitch_mutex);
        return gimbal_yaw_pitch_buffer;
    }
    void set_bullet_positions(const std::vector<Vec3>& ps) noexcept {
        std::lock_guard<std::mutex> lock(bullet_positions_mutex);
        bullet_positions_buffer = ps;
    }
    std::vector<Vec3> bullet_positions() const noexcept {
        std::lock_guard<std::mutex> lock(bullet_positions_mutex);
        return bullet_positions_buffer;
    }
};
void draw_auto_aim(cv::Mat& img, const AutoAimDebugCtx& ctx);
void write_debug_data(const AutoAimDebugCtx& ctx);

} // namespace awakening::auto_aim