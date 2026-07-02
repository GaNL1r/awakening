#pragma once

#include "tasks/base/common.hpp"
#include "tasks/base/traj.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace awakening::dta_utils {
struct ControlPoint {
    double yaw;
    double pitch;
    int aim_id;
    AimPoint aim_point;
    bool valid;
};
struct GimbalState {
    struct State {
        double p;
        double v;
        double a;
        bool on_traj;
    };
    State yaw_state;
    State pitch_state;
    int aim_id = 0;
    GimbalState() = default;
    GimbalState(const GimbalState::State& y, const GimbalState::State& p):
        yaw_state(y),
        pitch_state(p) {}
    static GimbalState lerp(const GimbalState& s0, const GimbalState& s1, double a) noexcept {
        GimbalState r;
        r.aim_id = (a < 0.5) ? s0.aim_id : s1.aim_id;
        r.yaw_state =
            GimbalState::State { .p = utils::lerp_angle(s0.yaw_state.p, s1.yaw_state.p, a),
                                 .v = std::lerp(s0.yaw_state.v, s1.yaw_state.v, a),
                                 .a = std::lerp(s0.yaw_state.a, s1.yaw_state.a, a) };
        r.pitch_state =
            GimbalState::State { .p = utils::lerp_angle(s0.pitch_state.p, s1.pitch_state.p, a),
                                 .v = std::lerp(s0.pitch_state.v, s1.pitch_state.v, a),
                                 .a = std::lerp(s0.pitch_state.a, s1.pitch_state.a, a) };

        return r;
    }
};
template<class Scale>
struct QuinticSegment {
    Scale T = 0.0;
    Eigen::Matrix<Scale, 6, 1> c;
    GimbalState::State head;
    GimbalState::State tail;
    bool on_traj;

    static inline Eigen::Matrix<Scale, 6, 1> solve1d_closed_form(
        Scale p0,
        Scale v0,
        Scale a0,
        Scale p1,
        Scale v1,
        Scale a1,
        Scale T
    ) noexcept {
        Eigen::Matrix<Scale, 6, 1> c;
        c.setZero();

        if (T <= static_cast<Scale>(1e-9)) {
            c[0] = p0;
            return c;
        }

        const Scale invT = static_cast<Scale>(1.0) / T;
        const Scale invT2 = invT * invT;
        const Scale invT3 = invT2 * invT;
        const Scale invT4 = invT3 * invT;
        const Scale invT5 = invT4 * invT;

        // c0, c1, c2
        c[0] = p0;
        c[1] = v0;
        c[2] = static_cast<Scale>(0.5) * a0;

        // boundary mismatch
        const Scale dp = p1 - (p0 + v0 * T + static_cast<Scale>(0.5) * a0 * T * T);
        const Scale dv = v1 - (v0 + a0 * T);
        const Scale da = a1 - a0;

        // quintic coefficients
        c[3] = (static_cast<Scale>(10.0) * dp - static_cast<Scale>(4.0) * dv * T
                + static_cast<Scale>(0.5) * da * T * T)
            * invT3;

        c[4] = (static_cast<Scale>(-15.0) * dp + static_cast<Scale>(7.0) * dv * T
                - static_cast<Scale>(1.0) * da * T * T)
            * invT4;

        c[5] = (static_cast<Scale>(6.0) * dp - static_cast<Scale>(3.0) * dv * T
                + static_cast<Scale>(0.5) * da * T * T)
            * invT5;

        return c;
    }

    [[nodiscard]] static inline QuinticSegment build(
        const GimbalState::State& s0,
        const GimbalState::State& s1,
        Scale T,
        bool on_traj
    ) noexcept {
        QuinticSegment seg;
        seg.head = s0;
        seg.tail = s1;
        seg.T = T;
        seg.c = solve1d_closed_form(s0.p, s0.v, s0.a, s1.p, s1.v, s1.a, T);
        seg.on_traj = on_traj;
        return seg;
    }

    static inline Scale max_abs_acc(const Eigen::Matrix<Scale, 6, 1>& c, Scale T) noexcept {
        if (T <= static_cast<Scale>(0))
            return static_cast<Scale>(0);

        auto acc = [&](Scale t) {
            Scale t2 = t * t;
            return 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t2 + 20 * c[5] * t2 * t;
        };

        Scale max_acc = std::max(std::abs(acc(0.0)), std::abs(acc(T)));

        const Scale eps = static_cast<Scale>(1e-9);

        const Scale A = 60.0 * c[5];
        const Scale B = 24.0 * c[4];
        const Scale C = 6.0 * c[3];

        auto update = [&](Scale t) {
            if (t > 0.0 && t < T) {
                max_acc = std::max(max_acc, std::abs(acc(t)));
            }
        };

        if (std::abs(A) < eps) {
            if (std::abs(B) > eps) {
                update(-C / B);
            }
            // else: jerk ~ constant → only endpoints matter
        } else {
            Scale D = B * B - 4 * A * C;

            if (D > eps) {
                Scale sqrtD = std::sqrt(D);
                Scale inv2A = static_cast<Scale>(0.5) / A;

                update((-B + sqrtD) * inv2A);
                update((-B - sqrtD) * inv2A);
            }
        }

        return std::isfinite(max_acc) ? max_acc : static_cast<Scale>(0);
    }

    [[nodiscard]] Scale inline duration() const noexcept {
        return T;
    }

    [[nodiscard]] Scale inline max_acc() const noexcept {
        return QuinticSegment::max_abs_acc(c, T);
    }
    [[nodiscard]] GimbalState::State inline eval(Scale t) const noexcept {
        GimbalState::State s;
        if (T <= 0.0)
            return s;
        t = std::clamp<Scale>(t, 0.0, T);
        Scale t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
        s.p = c[0] + c[1] * t + c[2] * t2 + c[3] * t3 + c[4] * t4 + c[5] * t5;
        s.v = c[1] + 2 * c[2] * t + 3 * c[3] * t2 + 4 * c[4] * t3 + 5 * c[5] * t4;
        s.a = 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t2 + 20 * c[5] * t3;
        s.on_traj = on_traj;
        return s;
    }
};
class LimitTrajectory: public Trajectory<GimbalState, double> {
public:
    using Seg = QuinticSegment<double>;
    struct Traj {
        std::vector<Seg> segs;
        std::vector<int> seg_start_idx;
        std::vector<int> seg_end_idx;
        std::vector<double> seg_prefix_time;
        std::optional<std::pair<int, int>> limit_interval;
        void reserve(size_t size) {
            segs.reserve(size);
            seg_start_idx.reserve(size);
            seg_end_idx.reserve(size);
            seg_prefix_time.reserve(size + 1);
        }
        void clear() {
            segs.clear();
            seg_start_idx.clear();
            seg_end_idx.clear();
            seg_prefix_time.clear();
            limit_interval.reset();
        }
        void push_seg(Seg seg, int start_idx, int end_idx) {
            segs.push_back(std::move(seg));
            seg_start_idx.push_back(start_idx);
            seg_end_idx.push_back(end_idx);
        }
        void rebuild_prefix(double first_time) {
            seg_prefix_time.resize(segs.size() + 1);
            seg_prefix_time[0] = first_time;
            for (size_t i = 0; i < segs.size(); ++i) {
                seg_prefix_time[i + 1] = seg_prefix_time[i] + segs[i].duration();
            }
        }
    };

    Traj yaw_traj;
    Traj pitch_traj;
    static inline double angle_diff(double a, double b) noexcept {
        double d = a - b;
        while (d > M_PI)
            d -= 2 * M_PI;
        while (d < -M_PI)
            d += 2 * M_PI;
        return d;
    }

    static inline double unwrap_angle(double prev, double curr) noexcept {
        return prev + angle_diff(curr, prev);
    }

    void unwrap_states_from(std::vector<GimbalState>& s, size_t first) const noexcept {
        if (s.size() < 2)
            return;
        first = std::max<size_t>(first, 1);
        for (size_t i = first; i < s.size(); ++i) {
            s[i].yaw_state.p = unwrap_angle(s[i - 1].yaw_state.p, s[i].yaw_state.p);
            s[i].pitch_state.p = unwrap_angle(s[i - 1].pitch_state.p, s[i].pitch_state.p);
        }
    }
    void unwrap_states(std::vector<GimbalState>& s) const noexcept {
        unwrap_states_from(s, 1);
    }
    void clear() {
        Trajectory::clear();
        yaw_traj.clear();
        pitch_traj.clear();
    }

    struct SegmentDesc {
        int l = 0;
        int r = 0;
        bool on_traj = true;
    };

    [[nodiscard]] static inline GimbalState::State average_motion_state(
        const GimbalState::State& s0,
        const GimbalState::State& s1,
        double T
    ) noexcept {
        GimbalState::State s = s0;
        s.v = T > 1e-9 ? (s1.p - s0.p) / T : 0.0;
        s.a = 0.0;
        return s;
    }

    [[nodiscard]] static inline double segment_avg_v(
        const std::vector<GimbalState::State>& s,
        const std::vector<double>& prefix,
        const SegmentDesc& d
    ) noexcept {
        const double T = prefix[d.r] - prefix[d.l];
        return T > 1e-9 ? (s[d.r].p - s[d.l].p) / T : 0.0;
    }

    [[nodiscard]] static inline Seg build_sampled_center_seg(
        const std::vector<GimbalState::State>& s,
        const std::vector<double>& prefix,
        const std::vector<SegmentDesc>& descs,
        size_t center
    ) noexcept {
        const auto& d = descs[center];
        const double T = prefix[d.r] - prefix[d.l];
        if (!d.on_traj || center == 0 || center + 1 == descs.size()) {
            auto head = s[d.l];
            auto tail = s[d.r];
            head.v = head.a = 0.0;
            tail.v = tail.a = 0.0;
            return Seg::build(head, tail, T, d.on_traj);
        }

        const bool has_left = d.on_traj && center > 0 && descs[center - 1].on_traj;
        const bool has_right = d.on_traj && center + 1 < descs.size() && descs[center + 1].on_traj;
        const double left_v = has_left ? segment_avg_v(s, prefix, descs[center - 1]) : 0.0;
        const double right_v = has_right ? segment_avg_v(s, prefix, descs[center + 1]) : 0.0;
        const double center_a = (has_left && has_right && T > 1e-9) ? (right_v - left_v) / T : 0.0;

        auto head = s[d.l];
        auto tail = s[d.r];
        head.v = has_left ? left_v : 0.0;
        tail.v = has_right ? right_v : 0.0;
        head.a = center_a;
        tail.a = center_a;
        return Seg::build(head, tail, T, d.on_traj);
    }

    [[nodiscard]] std::vector<SegmentDesc>
    make_segment_descs(int N, const std::optional<std::pair<int, int>>& interval) const {
        std::vector<SegmentDesc> descs;
        if (N <= 1)
            return descs;

        descs.reserve(N - 1);
        for (int i = 0; i < N - 1; ++i) {
            if (interval && i == interval->first) {
                descs.push_back({ interval->first, interval->second, false });
                i = interval->second - 1;
            } else {
                descs.push_back({ i, i + 1, true });
            }
        }
        return descs;
    }

    [[nodiscard]] size_t seed_desc_idx(
        const std::vector<SegmentDesc>& descs,
        const std::optional<std::pair<int, int>>& interval
    ) const noexcept {
        if (descs.empty())
            return 0;

        if (interval) {
            for (size_t i = 0; i < descs.size(); ++i) {
                if (descs[i].l <= interval->first && descs[i].r >= interval->second) {
                    return i;
                }
            }
        }

        return descs.size() / 2;
    }

    [[nodiscard]] std::optional<std::pair<int, int>> find_nearest_change_interval(
        const std::vector<GimbalState>& cp_vec,
        const std::vector<double>& prefix,
        double current_time
    ) const noexcept {
        std::optional<std::pair<int, int>> interval;
        double best_dist = std::numeric_limits<double>::max();
        for (size_t i = 0; i + 1 < cp_vec.size(); ++i) {
            if (cp_vec[i].aim_id == cp_vec[i + 1].aim_id)
                continue;

            const double seg_mid = 0.5 * (prefix[i] + prefix[i + 1]);
            const double dist = std::abs(seg_mid - current_time);
            if (dist < best_dist) {
                best_dist = dist;
                interval.emplace(static_cast<int>(i), static_cast<int>(i + 1));
            }
        }
        return interval;
    }

    [[nodiscard]] std::optional<std::pair<int, int>> expand_limit_interval(
        const std::vector<GimbalState>& cp_vec,
        const std::vector<GimbalState::State>& s,
        const std::vector<double>& prefix,
        std::optional<std::pair<int, int>> interval,
        double max_acc
    ) const noexcept {
        const int N = static_cast<int>(s.size());
        if (!interval)
            return interval;

        auto buildSeg = [&](int l, int r) -> Seg {
            double dur = prefix[r] - prefix[l];
            return Seg::build(s[l], s[r], dur, false);
        };
        const int base_l = interval->first;
        const int base_r = interval->second;

        int left_run_start = base_l;
        while (left_run_start > 0
               && cp_vec[left_run_start - 1].aim_id == cp_vec[left_run_start].aim_id) {
            --left_run_start;
        }

        int right_run_end = base_r;
        while (right_run_end + 1 < N
               && cp_vec[right_run_end].aim_id == cp_vec[right_run_end + 1].aim_id) {
            ++right_run_end;
        }

        const double left_mid_time = 0.5 * (prefix[left_run_start] + prefix[base_l]);
        const auto left_limit_it = std::lower_bound(
            prefix.begin() + left_run_start,
            prefix.begin() + base_l + 1,
            left_mid_time
        );
        const int left_limit = static_cast<int>(std::distance(prefix.begin(), left_limit_it));

        const double right_mid_time = 0.5 * (prefix[base_r] + prefix[right_run_end]);
        const auto right_limit_it = std::upper_bound(
            prefix.begin() + base_r,
            prefix.begin() + right_run_end + 1,
            right_mid_time
        );
        const int right_limit = static_cast<int>(std::distance(prefix.begin(), right_limit_it)) - 1;

        auto radius_interval = [&](int radius) -> std::pair<int, int> {
            return { std::max(left_limit, base_l - radius),
                     std::min(right_limit, base_r + radius) };
        };

        auto acc_at_radius = [&](int radius) -> double {
            const auto [l, r] = radius_interval(radius);
            return buildSeg(l, r).max_acc();
        };

        if (acc_at_radius(0) > max_acc) {
            const int max_radius = std::max(base_l - left_limit, right_limit - base_r);
            int best_radius = max_radius;

            for (int radius = 1; radius <= max_radius; ++radius) {
                if (acc_at_radius(radius) <= max_acc) {
                    best_radius = radius;
                    break;
                }
            }

            interval = radius_interval(best_radius);
        }
        return interval;
    }

    void build_continuous_centered_traj(
        Traj& traj,
        const std::vector<GimbalState::State>& s,
        const std::vector<double>& prefix,
        const std::vector<SegmentDesc>& descs,
        size_t seed
    ) const noexcept {
        traj.segs.clear();
        traj.seg_start_idx.clear();
        traj.seg_end_idx.clear();
        traj.seg_prefix_time.clear();
        if (descs.empty())
            return;

        std::vector<Seg> segs(descs.size());
        const size_t center = std::min(seed, descs.size() - 1);

        auto duration = [&](const SegmentDesc& d) { return prefix[d.r] - prefix[d.l]; };

        segs[center] = build_sampled_center_seg(s, prefix, descs, center);

        for (size_t i = center + 1; i < descs.size(); ++i) {
            const auto& d = descs[i];
            auto head = segs[i - 1].tail;
            head.p = s[d.l].p;
            auto tail = average_motion_state(s[d.l], s[d.r], duration(d));
            tail.p = s[d.r].p;
            segs[i] = Seg::build(head, tail, duration(d), d.on_traj);
        }

        for (size_t i = center; i-- > 0;) {
            const auto& d = descs[i];
            auto head = average_motion_state(s[d.l], s[d.r], duration(d));
            head.p = s[d.l].p;
            auto tail = segs[i + 1].head;
            tail.p = s[d.r].p;
            segs[i] = Seg::build(head, tail, duration(d), d.on_traj);
        }

        traj.segs.reserve(descs.size());
        traj.seg_start_idx.reserve(descs.size());
        traj.seg_end_idx.reserve(descs.size());
        for (size_t i = 0; i < descs.size(); ++i) {
            traj.push_seg(std::move(segs[i]), descs[i].l, descs[i].r);
        }
        traj.rebuild_prefix(prefix[descs.front().l]);
    }

    template<typename ProjectState>
    void limit_traj(
        Traj& traj,
        const std::vector<GimbalState>& cp_vec,
        const std::vector<double>& prefix,
        std::optional<std::pair<int, int>> change_interval,
        double max_acc,
        ProjectState&& project
    ) const noexcept {
        traj.clear();

        const int N = static_cast<int>(cp_vec.size());
        if (N <= 1)
            return;

        std::vector<GimbalState::State> s;
        s.resize(cp_vec.size());
        for (size_t i = 0; i < cp_vec.size(); ++i) {
            s[i] = project(cp_vec[i]);
            s[i].v = 0.0;
            s[i].a = 0.0;
        }

        const auto interval = expand_limit_interval(cp_vec, s, prefix, change_interval, max_acc);
        traj.limit_interval = interval;
        const auto descs = make_segment_descs(N, interval);
        build_continuous_centered_traj(traj, s, prefix, descs, seed_desc_idx(descs, change_interval));
    }

    void build_limit(double max_yaw_acc, double max_pitch_acc, double current_time) noexcept {
        auto& cp_vec = get_cp_vec();
        const auto& prefix = get_prefix();
        unwrap_states(cp_vec);
        const int N = static_cast<int>(cp_vec.size());
        if (N < 2)
            return;
        const auto change_interval = find_nearest_change_interval(cp_vec, prefix, current_time);
        limit_traj(
            yaw_traj,
            cp_vec,
            prefix,
            change_interval,
            max_yaw_acc,
            [](const GimbalState& s) { return s.yaw_state; }
        );
        limit_traj(
            pitch_traj,
            cp_vec,
            prefix,
            change_interval,
            max_pitch_acc,
            [](const GimbalState& s) { return s.pitch_state; }
        );
    }
    [[nodiscard]] inline GimbalState::State state_at(double t, const Traj& traj) const noexcept {
        if (traj.segs.empty())
            return {};
        if (t <= traj.seg_prefix_time[0])
            return traj.segs.front().eval(0.0);

        if (t >= traj.seg_prefix_time.back())
            return traj.segs.back().eval(traj.segs.back().T);

        const auto it =
            std::upper_bound(traj.seg_prefix_time.begin(), traj.seg_prefix_time.end(), t);

        size_t i = std::distance(traj.seg_prefix_time.begin(), it) - 1;
        i = std::min(i, traj.segs.size() - 1);

        const double t0 = traj.seg_prefix_time[i];
        return traj.segs[i].eval(t - t0);
    }
    [[nodiscard]] inline GimbalState state_at(double t) const noexcept {
        GimbalState::State yaw = state_at(t, yaw_traj);
        GimbalState::State pitch = state_at(t, pitch_traj);
        return GimbalState(yaw, pitch);
    }
};
template<typename TrackState>
inline void update_fsm(
    bool found,
    TrackState& state,
    int tracking_thres,
    double lost_time,
    double lost_time_thres
) noexcept {
    switch (state.tracker_state) {
        case TrackState::DETECTING:
            if (!found) {
                state.detect_count = 0;
                state.tracker_state = TrackState::LOST;
                return;
            }
            if (++state.detect_count > tracking_thres) {
                state.detect_count = 0;
                state.tracker_state = TrackState::TRACKING;
            }
            return;

        case TrackState::TRACKING:
            if (!found) {
                state.tracker_state = TrackState::TEMP_LOST;
            }
            return;

        case TrackState::TEMP_LOST:
            if (found) {
                state.tracker_state = TrackState::TRACKING;
                return;
            }
            if (lost_time > lost_time_thres) {
                state.tracker_state = TrackState::LOST;
            }
            return;

        default:
            return;
    }
}

inline double elapsed_sec(const TimePoint& from, const TimePoint& to) noexcept {
    return std::max(0.0, std::chrono::duration<double>(to - from).count());
}

template<typename CostMatrix>
inline std::vector<std::pair<int, int>>
greedy_match(const CostMatrix& cost, int n_obs, int n_ids, double max_cost) {
    std::vector<std::pair<int, int>> result;
    std::vector<bool> used_obs(n_obs, false);
    std::vector<bool> used_id(n_ids, false);

    while (true) {
        double best = max_cost;
        int best_obs = -1;
        int best_id = -1;

        for (int obs = 0; obs < n_obs; ++obs) {
            if (used_obs[obs]) {
                continue;
            }
            for (int id = 0; id < n_ids; ++id) {
                if (used_id[id]) {
                    continue;
                }
                if (cost[obs][id] < best) {
                    best = cost[obs][id];
                    best_obs = obs;
                    best_id = id;
                }
            }
        }

        if (best_obs < 0 || best_id < 0) {
            break;
        }

        used_obs[best_obs] = true;
        used_id[best_id] = true;
        result.emplace_back(best_obs, best_id);
    }

    return result;
}

} // namespace awakening::dta_utils
