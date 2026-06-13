#pragma once

#include "tasks/base/common.hpp"
#include "tasks/base/traj.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
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
struct QuinticSegment {
    double T = 0.0;
    Eigen::Matrix<double, 6, 1> c;
    GimbalState::State head;
    GimbalState::State tail;
    bool on_traj;

    static inline Eigen::Matrix<double, 6, 1> solve1d_closed_form(
        double p0,
        double v0,
        double a0,
        double p1,
        double v1,
        double a1,
        double T
    ) noexcept {
        Eigen::Matrix<double, 6, 1> c;
        double T2 = T * T;
        double T3 = T2 * T;
        double T4 = T3 * T;
        double T5 = T4 * T;

        // known low-order coefficients
        double c0 = p0;
        double c1 = v0;
        double c2 = a0 * 0.5;

        // closed-form for c3, c4, c5 (derived from boundary conditions at t=T)
        double c3 =
            (-3.0 * T2 * a0 + T2 * a1 - 12.0 * T * v0 - 8.0 * T * v1 - 20.0 * p0 + 20.0 * p1)
            / (2.0 * T3);
        double c4 =
            (1.5 * T2 * a0 - T2 * a1 + 8.0 * T * v0 + 7.0 * T * v1 + 15.0 * p0 - 15.0 * p1) / T4;
        double c5 =
            (-T2 * a0 + T2 * a1 - 6.0 * T * v0 - 6.0 * T * v1 - 12.0 * p0 + 12.0 * p1) / (2.0 * T5);

        c << c0, c1, c2, c3, c4, c5;
        return c;
    }

    [[nodiscard]] static inline QuinticSegment build(
        const GimbalState::State& s0,
        const GimbalState::State& s1,
        double T,
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
    static inline double max_abs_acc(const Eigen::Matrix<double, 6, 1>& c, double T) noexcept {
        if (T <= 0.0)
            return 0.0;

        auto acc = [&](double t) {
            double t2 = t * t;
            return 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t2 + 20 * c[5] * t2 * t;
        };

        double max_acc = std::max(std::abs(acc(0.0)), std::abs(acc(T)));

        // jerk = 6c3 + 24c4 t + 60c5 t^2
        double A = 60.0 * c[5];
        double B = 24.0 * c[4];
        double C = 6.0 * c[3];

        const double eps = 1e-9;

        if (std::abs(A) < eps) {
            if (std::abs(B) > eps) {
                double t = -C / B;
                if (t > 0.0 && t < T)
                    max_acc = std::max(max_acc, std::abs(acc(t)));
            }
        } else {
            double D = B * B - 4 * A * C;
            if (D >= 0.0) {
                double sqrtD = std::sqrt(D);
                double inv2A = 1.0 / (2 * A);

                double t1 = (-B + sqrtD) * inv2A;
                double t2 = (-B - sqrtD) * inv2A;

                if (t1 > 0.0 && t1 < T)
                    max_acc = std::max(max_acc, std::abs(acc(t1)));
                if (t2 > 0.0 && t2 < T)
                    max_acc = std::max(max_acc, std::abs(acc(t2)));
            }
        }

        return std::isfinite(max_acc) ? max_acc : 0.0;
    }

    [[nodiscard]] double inline duration() const noexcept {
        return T;
    }

    [[nodiscard]] double inline max_acc() const noexcept {
        return QuinticSegment::max_abs_acc(c, T);
    }
    [[nodiscard]] GimbalState::State inline eval(double t) const noexcept {
        GimbalState::State s;
        if (T <= 0.0)
            return s;
        t = std::clamp(t, 0.0, T);
        double t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
        s.p = c[0] + c[1] * t + c[2] * t2 + c[3] * t3 + c[4] * t4 + c[5] * t5;
        s.v = c[1] + 2 * c[2] * t + 3 * c[3] * t2 + 4 * c[4] * t3 + 5 * c[5] * t4;
        s.a = 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t2 + 20 * c[5] * t3;
        s.on_traj = on_traj;
        return s;
    }
};
class LimitTrajectory: public Trajectory<GimbalState, double> {
public:
    struct Traj {
        std::vector<QuinticSegment> segs;
        std::vector<double> seg_prefix_time;
        void reserve(size_t size) {
            segs.reserve(size);
            seg_prefix_time.reserve(size + 1);
        }
        void clear() {
            segs.clear();
            seg_prefix_time.clear();
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

    void unwrap_states(std::vector<GimbalState>& s) const noexcept {
        for (size_t i = 1; i < s.size(); ++i) {
            s[i].yaw_state.p = unwrap_angle(s[i - 1].yaw_state.p, s[i].yaw_state.p);
            s[i].pitch_state.p = unwrap_angle(s[i - 1].pitch_state.p, s[i].pitch_state.p);
        }
    }
    void clear() {
        Trajectory::clear();
        yaw_traj.clear();
        pitch_traj.clear();
    }

    [[nodiscard]] std::pair<std::vector<GimbalState::State>, std::vector<GimbalState::State>>
    compute_node_states(const std::vector<GimbalState>& gp, const std::vector<double>& prefix)
        const noexcept {
        const size_t N = gp.size();
        std::vector<GimbalState::State> yaw(N), pitch(N);
        for (size_t i = 0; i < N; ++i) {
            yaw[i] = gp[i].yaw_state;
            pitch[i] = gp[i].pitch_state;
        }
        if (N < 2)
            return { yaw, pitch };
        auto compute_va = [&](std::vector<GimbalState::State>& s) {
            // 边界
            s.front().v = s.back().v = 0.0;
            s.front().a = s.back().a = 0.0;

            for (size_t i = 1; i + 1 < N; ++i) {
                const double dt0 = prefix[i] - prefix[i - 1];
                const double dt1 = prefix[i + 1] - prefix[i];
                const double denom = dt0 + dt1;

                if (denom < 1e-6) {
                    s[i].v = s[i].a = 0.0;
                    continue;
                }

                const double w0 = dt1 / denom;
                const double w1 = dt0 / denom;

                s[i].v = w0 * (s[i].p - s[i - 1].p) / dt0 + w1 * (s[i + 1].p - s[i].p) / dt1;
                s[i].a = 2.0 * ((s[i + 1].p - s[i].p) / dt1 - (s[i].p - s[i - 1].p) / dt0) / denom;
            }
        };

        compute_va(yaw);
        compute_va(pitch);

        return { yaw, pitch };
    }

    void limit_traj(
        Traj& traj,
        const std::vector<GimbalState::State>& s,
        const std::vector<double>& prefix,
        int near_change_idx,
        double max_acc
    ) const noexcept {
        traj.segs.clear();
        traj.seg_prefix_time.clear();

        const int N = static_cast<int>(s.size());
        if (N <= 1)
            return;

        auto buildSeg = [&](int l, int r) -> QuinticSegment {
            double dur = prefix[r] - prefix[l];
            return QuinticSegment::build(s[l], s[r], dur, false);
        };

        std::optional<std::pair<int, int>> interval;
        if (near_change_idx >= 0) {
            int l = std::clamp(near_change_idx, 0, N - 1);
            int r = std::clamp(near_change_idx + 1, 0, N - 1);
            if (l < r)
                interval.emplace(l, r);
        }

        if (!interval) {
            traj.segs.reserve(N - 1);
            for (int i = 0; i < N - 1; ++i) {
                traj.segs.push_back(
                    QuinticSegment::build(s[i], s[i + 1], prefix[i + 1] - prefix[i], true)
                );
            }

            traj.rebuild_prefix(prefix[0]);
            return;
        }
        //向两个调整多项式时间长度，直到满足加速度约束
        {
            int& l = interval->first;
            int& r = interval->second;
            QuinticSegment seg = buildSeg(l, r);

            auto try_candidate = [&](int nl, int nr) -> bool {
                nl = std::max(0, nl);
                nr = std::min(N - 1, nr);
                if (nl == l && nr == r)
                    return false;

                QuinticSegment cand = buildSeg(nl, nr);
                if (cand.max_acc() <= seg.max_acc()) {
                    l = nl;
                    r = nr;
                    seg = std::move(cand);
                    return true;
                }
                return false;
            };

            while (seg.max_acc() > max_acc) {
                bool expanded = false;

                if (l > 0 || r < N - 1) {
                    expanded = try_candidate(l - 1, r + 1);
                }

                if (!expanded)
                    break;

                if (l == 0 && r == N - 1 && seg.max_acc() > max_acc)
                    break;
            }
        }

        traj.segs.reserve(N - 1);
        for (int i = 0; i < N - 1; ++i) {
            if (interval && i == interval->first) {
                traj.segs.push_back(buildSeg(interval->first, interval->second));
                i = interval->second - 1; // skip covered indices
            } else {
                traj.segs.push_back(
                    QuinticSegment::build(s[i], s[i + 1], prefix[i + 1] - prefix[i], true)
                );
            }
        }

        traj.rebuild_prefix(prefix[0]);
    }
    void build_limit(double max_yaw_acc, double max_pitch_acc, double current_time) noexcept {
        auto& cp_vec = get_cp_vec();
        auto prefix = get_prefix();
        unwrap_states(cp_vec);
        auto [yaw_states, pitch_states] = compute_node_states(cp_vec, prefix); //粗解va
        int best_idx = -1;
        double best_dist = std::numeric_limits<double>::max();
        const int N = static_cast<int>(cp_vec.size());
        if (N < 2)
            return;

        for (size_t i = 0; i + 1 < cp_vec.size(); ++i) { //找到最近的换板点
            if (cp_vec[i].aim_id == cp_vec[i + 1].aim_id)
                continue;

            const double seg_mid = 0.5 * (prefix[i] + prefix[i + 1]);
            const double dist = std::abs(seg_mid - current_time);

            if (dist < best_dist) {
                best_dist = dist;
                best_idx = static_cast<int>(i);
            }
        }
        limit_traj(yaw_traj, yaw_states, prefix, best_idx, max_yaw_acc);
        limit_traj(pitch_traj, pitch_states, prefix, best_idx, max_pitch_acc);
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
