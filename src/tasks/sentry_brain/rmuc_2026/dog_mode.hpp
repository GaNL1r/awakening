#pragma once
#include "_rcl/tf.hpp"
#include "mode_base.hpp"
#include "tasks/base/packet_typedef_send.hpp"
#include "utils/drivers/serial_driver.hpp"
#include "utils/impl.hpp"
namespace awakening::sentry_brain {
class DogMode: public ModeBase {
public:
    struct Params {
        double go_home_hp_ratio;
        int home_bullet_num;
        void load(const YAML::Node& config) {
            go_home_hp_ratio = config["go_home_hp_ratio"].as<double>();
            home_bullet_num = config["home_bullet_num"].as<int>();
        }
    } params_;
    DogMode(
        rcl::RclcppNode& rcl_node,
        rcl::TF& rcl_tf,
        const YAML::Node& config,
        std::shared_ptr<SerialDriver> serial
    ):
        ModeBase(rcl_node, rcl_tf, config, serial) {
        params_.load(config);
    }
    void tick_callback() override {
        if (serial_) {
            SentryRefereeSend send;
            send.cmd_ID = SentryRefereeSend::ID;
            send.set_current_pose = std::to_underlying(sentry_pose);
        }
        auto& map = RMUC2026Map::instance();
        if (state_.current_game_time_ < 0) {
            AWAKENING_INFO("waiting for game start... current_time: {}", state_.current_game_time_);
            return;
        }
        if (in_home()) {
            state_.home_allowance_bullets_ = 0;
        }
        if (target_in_big_yaw_.check()) {
            sentry_pose = GobalState::Pose::Attack;
        }
        double cur_hp_ratio = double(state_.current_hp) / state_.max_hp;
        if (cur_hp_ratio < params_.go_home_hp_ratio || state_.current_hp < 60) {
            sentry_pose = GobalState::Pose::Move;
            go<home_t>();
            wait_until(
                [&]() {
                    if (std::abs(state_.current_hp - state_.max_hp) < 50) {
                        AWAKENING_INFO("hp is enough: {}", state_.current_hp);
                        return true;
                    }
                    AWAKENING_INFO("waiting for hp to recover: {}", state_.current_hp);
                    return false;
                },
                std::chrono::duration<double>(1.0)
            );
            return;
        }
        if (state_.current_bullets_ < params_.home_bullet_num) {
            if (state_.home_allowance_bullets_ > 10) {
                go<home_t>();
                if (!is_reached<home_t>()) {
                    sentry_pose = GobalState::Pose::Move;
                }
            } else {
                go<ally_fort_t>();
                if (!is_reached<ally_fort_t>()) {
                    sentry_pose = GobalState::Pose::Move;
                }
            }

            return;
        }
        if (state_.enemy_outpost_active_) {
            go<ally_highlands_gain_t>();
            if (is_reached<ally_highlands_gain_t>()) {
                sentry_pose = GobalState::Pose::Attack;
            } else {
                sentry_pose = GobalState::Pose::Move;
            }
            return;
        }
        if (state_.remain_rebuild_outpost_chance_ > 0) {
            go<ally_outpost_t>();
            if (is_reached<ally_outpost_t>()) {
                sentry_pose = GobalState::Pose::Defend;
            } else {
                sentry_pose = GobalState::Pose::Move;
            }
            return;
        }
        patrol<ally_beijing_tunnel_top_t, enemy_jiansudai_tunnel_top_t>(5.0);
        if (state_.current_hp < 10) {
            sentry_pose = GobalState::Pose::Defend;
        }
    }
};
} // namespace awakening::sentry_brain