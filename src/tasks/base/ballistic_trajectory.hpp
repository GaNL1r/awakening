#pragma once
#include "traj.hpp"
#include "utils/common/type_common.hpp"
#include "utils/utils.hpp"
#include <deque>
#include <optional>
#include <vector>
#include <yaml-cpp/node/node.h>
namespace awakening {

class BallisticTrajectory {
public:
    using Ptr = std::shared_ptr<BallisticTrajectory>;
    struct Params {
        double gravity = 9.8;
        double resistance = 0.092;
        int max_iter = 10;
        void load(const YAML::Node& config) {
            gravity = config["gravity"].as<double>();
            resistance = config["resistance"].as<double>();
            max_iter = config["max_iter"].as<int>();
        }
    } params_;

    BallisticTrajectory(const YAML::Node& config) {
        params_.load(config);
    }
    static Ptr create(const YAML::Node& config) {
        return std::make_shared<BallisticTrajectory>(config);
    }
    std::optional<double> solve_pitch(const Vec3& target_pos, double v0) const {
        const double target_height = target_pos.z();
        const double distance =
            std::sqrt(target_pos.x() * target_pos.x() + target_pos.y() * target_pos.y());

        if (distance < 1e-6 || v0 < 1e-3) {
            return std::nullopt;
        }

        // 二分法边界 [-45°, 60°]
        double left = -M_PI / 4.0;
        double right = M_PI / 3.0;

        auto f = [&](double angle) -> double {
            double t;
            if (params_.resistance < 1e-6) {
                t = distance / (v0 * std::cos(angle));
            } else {
                double r = std::max(params_.resistance, 1e-6);
                t = (std::exp(r * distance) - 1) / (r * v0 * std::cos(angle));
            }

            return v0 * std::sin(angle) * t - 0.5 * params_.gravity * t * t - target_height;
        };

        double f_left = f(left);
        double f_right = f(right);

        if (f_left * f_right > 0) {
            return std::nullopt; // 没有解
        }

        double mid = 0;
        for (int i = 0; i < params_.max_iter; ++i) {
            mid = 0.5 * (left + right);
            double f_mid = f(mid);

            if (std::abs(f_mid) < 1e-3 || (right - left) < 1e-6) {
                return std::make_optional(mid);
            }

            if (f_left * f_mid < 0) {
                right = mid;
                f_right = f_mid;
            } else {
                left = mid;
                f_left = f_mid;
            }
        }

        return std::make_optional(mid);
    }
    double solve_flytime(const Vec3& target_pos, const double v0) {
        double r = params_.resistance < 1e-4 ? 1e-4 : params_.resistance;
        double distance =
            std::sqrt(target_pos.x() * target_pos.x() + target_pos.y() * target_pos.y());
        double angle = std::atan2(target_pos.z(), distance);
        double t = (std::exp(r * distance) - 1) / (r * v0 * std::cos(angle));

        return t;
    }
    std::pair<double, double> solve_distance_height(double pitch, double v0, double t) const {
        double r = params_.resistance < 1e-4 ? 1e-4 : params_.resistance;
        double g = params_.gravity;

        double cos_theta = std::cos(pitch);
        double sin_theta = std::sin(pitch);

        if (v0 < 1e-6 || std::abs(cos_theta) < 1e-6) {
            return { 0.0, 0.0 };
        }
        double distance = std::log(1 + r * v0 * cos_theta * t) / r;
        double height = v0 * sin_theta * t - 0.5 * g * t * t;

        return { distance, height };
    }
};
struct Bullet {
    TimePoint fire_time;
    ISO3 fire_time_shoot_in_odom;
    double speed_in_odom;
    std::optional<Vec3>
    get_pos_at(TimePoint t, BallisticTrajectory::Ptr b, const std::pair<double, double>& offset)
        const {
        double dt = std::chrono::duration<double>(t - fire_time).count();
        if (dt <= 0) {
            return std::nullopt;
        }
        auto euler = utils::matrix2euler(fire_time_shoot_in_odom.linear(), utils::EulerOrder::ZYX);
        double yaw = euler[0] - offset.first;
        double pitch = -euler[1] - offset.second;
        auto [dis, height] = b->solve_distance_height(pitch, speed_in_odom, dt);
        double x = dis * std::cos(yaw);
        double y = dis * std::sin(yaw);
        double z = height;
        return fire_time_shoot_in_odom.translation() + Vec3(x, y, z);
    }
};
class BulletPickUp {
public:
    mutable std::mutex mtx;
    std::deque<Bullet> bullets;
    BulletPickUp(const YAML::Node& config) {
        b = BallisticTrajectory::create(config["ballistic_trajectory"]);
    }
    void push_back(const Bullet& bullet) {
        std::lock_guard<std::mutex> lock(mtx);
        bullets.push_back(std::move(bullet));
    }
    void update(TimePoint t, double max_fly_time) {
        if (bullets.empty())
            return;
        std::lock_guard<std::mutex> lock(mtx);
        while (!bullets.empty()
               && std::chrono::duration<double>(t - bullets.front().fire_time).count()
                   > max_fly_time)
        {
            bullets.pop_front();
        }
    }
    std::vector<Vec3>
    get_bullet_positions(TimePoint t, const std::pair<double, double>& offset) const {
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<Vec3> positions;
        for (const auto& bullet: bullets) {
            auto p_opt = bullet.get_pos_at(t, b, offset);
            if (p_opt) {
                positions.push_back(*p_opt);
            }
        }
        return positions;
    }
    BallisticTrajectory::Ptr b;
};
} // namespace awakening
