#include "armor_tracker.hpp"
#include "angles.h"
#include "utils/logger.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <mutex>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <utility>
#include <vector>
namespace awakening::auto_aim {
struct ArmorTracker::Impl {
    Impl(const YAML::Node& config) {
        cfg_.load(config);
    }
    ArmorTarget track(
        Armors& armors,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom,
        int frame_id
    ) {
        auto& cur = target_buf_[cur_target_idx_];
        auto& pre = target_buf_[pre_target_idx_];
        auto process = [&](int idx) {
            auto& t = target_buf_[idx];
            bool found = (t.track_state.tracker_state == ArmorTarget::TrackState::LOST)
                ? init_target(t, armors, frame_id, camera_info, camera_cv_in_odom)
                : update_target(t, armors, camera_info, camera_cv_in_odom);
            update_fsm(found, idx, armors.timestamp);
            if (found) {
                found_count_++;
            }
            return found;
        };
        //双缓冲，方便异常丢失恢复，方便操作手换目标
        process(cur_target_idx_);

        if (cur.track_state.tracker_state == ArmorTarget::TrackState::TEMP_LOST) {
            process(pre_target_idx_);

            if (pre.track_state.tracker_state == ArmorTarget::TrackState::TRACKING) {
                // if (cur.target_number != ArmorClass::OUTPOST)
                // { //给4mm英雄用（，太远我都看不清装甲板
                std::swap(cur, pre);
                pre.track_state.tracker_state = ArmorTarget::TrackState::LOST;
                // }
            }
        } else if (cur.track_state.tracker_state == ArmorTarget::TrackState::TRACKING) {
            pre.track_state.tracker_state = ArmorTarget::TrackState::LOST; //cur恢复就重置
        }

        return target_buf_[cur_target_idx_].fast_copy_without_ekf(); //下游不让用ekf
    }
    bool init_target(
        ArmorTarget& target,
        Armors& armors,
        int frame_id,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom,
        std::vector<ArmorClass> ignore = {}
    ) noexcept {
        if (armors.armors.empty()) {
            return false;
        }
        bool found = false;
        Armor init_target;
        for (auto& a: armors.armors) {
            if (!(a.color == ArmorColor::NONE || a.color == ArmorColor::PURPLE) && !found) {
                if (!(target_buf_[cur_target_idx_].target_number == ArmorClass::OUTPOST
                      && a.number != ArmorClass::OUTPOST && target_buf_[cur_target_idx_].check()))
                {
                    init_target = a;
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            return false;
        }
        // if (iam_sentry) {
        //     ArmorTarget::armor_pnp(init_target, camera_info, camera_cv_in_odom);//逆天散布不让超远击打！
        //     if (init_target.pose.translation().norm() > 5) {
        //         return false;
        //     }
        // }

        AWAKENING_INFO("init target: {}", string_by_armor_class(init_target.number));
        target.reset(init_target, cfg_, armors.timestamp, frame_id, camera_info, camera_cv_in_odom);
        target.track_state.tracker_state = ArmorTarget::TrackState::DETECTING;
        return true;
    }
    bool update_target(
        ArmorTarget& target,
        Armors& armors,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom,
        std::vector<ArmorClass> ignore = {}
    ) noexcept {
        if (armors.armors.empty())
            return false;
        target.predict_ekf(armors.timestamp);
        std::vector<Armor> candidates;
        candidates.reserve(armors.armors.size());
        for (const auto& a: armors.armors) {
            if (a.number == target.target_number) {
                if (a.color == ArmorColor::NONE || a.color == ArmorColor::PURPLE) {
                    continue;
                }
                candidates.emplace_back(a);
            }
        }

        if (candidates.empty())
            return false;
        std::vector<Light> lights;
        auto matched_armors = target.match_armor(candidates, camera_info, camera_cv_in_odom);
        auto matched_lights =
            target.match_light(armors.lights, matched_armors, camera_info, camera_cv_in_odom);
        int updated = target.update(
            matched_armors,
            matched_lights,
            armors.timestamp,
            camera_info,
            camera_cv_in_odom
        );
        return updated > 0;
    }
    void update_fsm(bool found, size_t i, const TimePoint& now) noexcept {
        auto& target = target_buf_[i];
        auto& s = target.track_state;

        switch (s.tracker_state) {
            case ArmorTarget::TrackState::DETECTING:
                if (!found) {
                    s.detect_count = 0;
                    s.tracker_state = ArmorTarget::TrackState::LOST;
                    return;
                }
                if (++s.detect_count > cfg_.tracking_thres) {
                    s.detect_count = 0;
                    s.tracker_state = ArmorTarget::TrackState::TRACKING;
                }
                return;

            case ArmorTarget::TrackState::TRACKING:
                if (!found) {
                    s.tracker_state = ArmorTarget::TrackState::TEMP_LOST;
                }
                return;

            case ArmorTarget::TrackState::TEMP_LOST:
                if (found) {
                    s.tracker_state = ArmorTarget::TrackState::TRACKING;
                    return;
                }
                if (lost_time(target, now) > lost_time_thres(target)) {
                    s.tracker_state = ArmorTarget::TrackState::LOST;
                }
                return;

            default:
                return;
        }

        if (found)
            ++found_count_;
    }
    double lost_time(const ArmorTarget& target, const TimePoint& now) const noexcept {
        return std::max(0.0, std::chrono::duration<double>(now - target.last_update).count());
    }
    double lost_time_thres(const ArmorTarget& target) const noexcept {
        return (target.target_number == ArmorClass::OUTPOST) ? cfg_.lost_time_thres_outpost
                                                             : cfg_.lost_time_thres;
    }
    void set_sentry(bool is_sentry) {
        iam_sentry = is_sentry;
    }

    int is_none_purple_count_ = 0;
    int found_count_ = 0;

    size_t cur_target_idx_ = 0;
    size_t pre_target_idx_ = 1;
    std::array<ArmorTarget, 2> target_buf_;
    ArmorTrackerCfg cfg_;
    bool iam_sentry = false;
};
ArmorTracker::ArmorTracker(const YAML::Node& config) {
    _impl = std::make_unique<Impl>(config);
}
ArmorTracker::~ArmorTracker() noexcept {
    _impl.reset();
}

ArmorTarget ArmorTracker::track(
    Armors& armors,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom,
    int frame_id
) {
    return _impl->track(armors, camera_info, camera_cv_in_odom, frame_id);
}
int ArmorTracker::get_count() {
    return _impl->found_count_;
}
void ArmorTracker::reset_count() {
    _impl->found_count_ = 0;
}
void ArmorTracker::set_sentry(bool is_sentry) {
    _impl->set_sentry(is_sentry);
}
} // namespace awakening::auto_aim
