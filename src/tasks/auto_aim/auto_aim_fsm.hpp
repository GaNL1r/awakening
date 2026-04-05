#pragma once

#include <string>
#include <yaml-cpp/node/node.h>
namespace awakening {
namespace auto_aim {
    enum class AutoAimFsm : int {
        AIM_SINGLE_ARMOR,
        AIM_WHOLE_CAR_ARMOR,
        AIM_WHOLE_CAR_PAIR,
        AIM_WHOLE_CAR_CENTER,
    };

    inline std::string string_by_auto_aim_fsm(AutoAimFsm state) {
        constexpr const char* details[] = {
            "AIM_SINGLE_ARMOR",
            "AIM_WHOLE_CAR_ARMOR",
            "AIM_WHOLE_CAR_PAIR",
            "AIM_WHOLE_CAR_CENTER",

        };
        return std::string(details[std::to_underlying(state)]);
    }

    class AutoAimFsmController {
    public:
        struct Params {
            int transfer_thresh;
            double single_whole_up;
            double single_whole_down;
            double whole_pair_up;
            double whole_pair_down;
            double pair_center_up;
            double pair_center_down;
            void load(const YAML::Node& config) {
                transfer_thresh = config["transfer_thresh"].as<int>();
                single_whole_up = config["single_whole_up"].as<double>();
                single_whole_down = config["single_whole_down"].as<double>();
                whole_pair_up = config["whole_pair_up"].as<double>();
                whole_pair_down = config["whole_pair_down"].as<double>();
                pair_center_up = config["pair_center_up"].as<double>();
                pair_center_down = config["pair_center_down"].as<double>();
            }
        } params_;
        AutoAimFsmController(const YAML::Node& config) {
            params_.load(config);
        }
        AutoAimFsm get_state() const {
            return fsm_state_;
        }
        AutoAimFsm fsm_state_ { AutoAimFsm::AIM_SINGLE_ARMOR };

        int overflow_count_ = 0;

        void update(double v_yaw, bool target_jumped) {
            if (!target_jumped) {
                fsm_state_ = AutoAimFsm::AIM_SINGLE_ARMOR;
                overflow_count_ = 0;
                return;
            }

            const double av = std::abs(v_yaw);

            switch (fsm_state_) {
                case AutoAimFsm::AIM_SINGLE_ARMOR: {
                    overflow_count_ = (av > params_.single_whole_up) ? overflow_count_ + 1 : 0;
                    if (overflow_count_ > params_.transfer_thresh) {
                        fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_ARMOR;
                        overflow_count_ = 0;
                    }
                    break;
                }

                case AutoAimFsm::AIM_WHOLE_CAR_ARMOR: {
                    if (av > params_.whole_pair_up)
                        ++overflow_count_;
                    else if (av < params_.single_whole_down)
                        --overflow_count_;
                    else
                        overflow_count_ = 0;

                    if (std::abs(overflow_count_) > params_.transfer_thresh) {
                        fsm_state_ = (overflow_count_ > 0) ? AutoAimFsm::AIM_WHOLE_CAR_PAIR
                                                           : AutoAimFsm::AIM_SINGLE_ARMOR;
                        overflow_count_ = 0;
                    }
                    break;
                }

                case AutoAimFsm::AIM_WHOLE_CAR_PAIR: {
                    if (av > params_.pair_center_up)
                        ++overflow_count_;
                    else if (av < params_.whole_pair_down)
                        --overflow_count_;
                    else
                        overflow_count_ = 0;

                    if (std::abs(overflow_count_) > params_.transfer_thresh) {
                        fsm_state_ = (overflow_count_ > 0) ? AutoAimFsm::AIM_WHOLE_CAR_CENTER
                                                           : AutoAimFsm::AIM_WHOLE_CAR_ARMOR;
                        overflow_count_ = 0;
                    }
                    break;
                }

                case AutoAimFsm::AIM_WHOLE_CAR_CENTER: {
                    overflow_count_ = (av < params_.pair_center_down) ? overflow_count_ + 1 : 0;
                    if (overflow_count_ > params_.transfer_thresh) {
                        fsm_state_ = AutoAimFsm::AIM_WHOLE_CAR_PAIR;
                        overflow_count_ = 0;
                    }
                    break;
                }

                default:
                    fsm_state_ = AutoAimFsm::AIM_SINGLE_ARMOR;
                    overflow_count_ = 0;
                    break;
            }
        }
    };
} // namespace auto_aim
} // namespace awakening