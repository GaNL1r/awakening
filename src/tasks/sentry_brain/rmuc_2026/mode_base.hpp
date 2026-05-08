#pragma once
#include "_rcl/node.hpp"
#include "_rcl/tf.hpp"
#include "gobal_state.hpp"
#include "map.hpp"
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "tasks/base/packet_typedef_receive.hpp"
#include "utils/drivers/serial_driver.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/subscription.hpp>
namespace awakening::sentry_brain {
class ModeBase {
public:
    ModeBase(
        rcl::RclcppNode& rcl_node,
        rcl::TF& rcl_tf,
        const YAML::Node& config,
        std::shared_ptr<SerialDriver> serial
    ):
        rcl_node_(rcl_node),
        rcl_tf_(rcl_tf) {
        serial_ = serial;
        goal_pub_ =
            rcl_node_.make_pub<geometry_msgs::msg::PoseStamped>("rose_goal", rclcpp::QoS(10));
        odom_sub_ = rcl_node_.make_sub<nav_msgs::msg::Odometry>(
            "Odometry",
            rclcpp::QoS(10),
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                const auto& odom_in = *msg;

                static Eigen::Isometry3d T;
                if (auto opt = rcl_tf_.get_transform<double>(
                        "map",
                        odom_in.header.frame_id,
                        odom_in.header.stamp,
                        rclcpp::Duration::from_seconds(0.1)
                    ))
                {
                    T = *opt;
                } else {
                }
                Eigen::Vector3d p(
                    odom_in.pose.pose.position.x,
                    odom_in.pose.pose.position.y,
                    odom_in.pose.pose.position.z
                );
                p = T * p;
                current_pos_ = p.head<2>();
            }
        );
    }
    virtual void tick_callback() = 0;
    ~ModeBase() {
        stop();
    }
    void start() {
        running_ = true;
        pub_goal_thread_ = std::thread([&]() {
            auto next_tp = std::chrono::steady_clock::now();
            while (rclcpp::ok()) {
                next_tp += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(3.0)
                );
                pub_goal_callback();
                std::this_thread::sleep_until(next_tp);
            }
        });
        tick_thread_ = std::thread([&]() {
            auto next_tp = std::chrono::steady_clock::now();
            while (rclcpp::ok()) {
                next_tp += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(1 / 2.0)
                );
                tick_callback();
                std::this_thread::sleep_until(next_tp);
            }
        });
    }

    void stop() {
        running_ = false;
        if (pub_goal_thread_.joinable()) {
            pub_goal_thread_.join();
        }
        if (tick_thread_.joinable()) {
            tick_thread_.join();
        }
    }
    void update_gobal_state(const SentryRefereeReceive& packet) noexcept {
        state_.update(packet);
    }
    void update_armor_target(const auto_aim::ArmorTarget& t) noexcept {
        target_in_big_yaw_ = t;
    }
    template<typename Func>
    void wait_until(Func&& func, std::chrono::duration<double> check_dt) const noexcept {
        while (running_) {
            if (func())
                break;
            std::this_thread::sleep_for(check_dt);
        }
    }

    bool in_home() {
        auto& map = RMUC2026Map::instance();
        return (current_pos_ - map.get<home_t>().head<2>()).norm() < 0.5;
    }
    template<typename Key>
    bool is_reached() {
        auto& map = RMUC2026Map::instance();
        if ((current_pos_ - map.get<Key>().template head<2>()).norm() < 0.5) {
            AWAKENING_INFO("{} has reached", Key::name);
            return true;
        }
        return false;
    }
    template<typename... Keys>
    void patrol(double change_dt) {
        static size_t idx = 0;
        static TimePoint last_time = Clock::now();

        auto now = Clock::now();

        if (now - last_time < std::chrono::seconds((int)change_dt)) {
            return;
        }

        last_time = now;

        using Func = void (*)(decltype(this));

        std::array<std::function<void()>, sizeof...(Keys)> funcs = { [this]() { go<Keys>(); }... };

        funcs[idx]();

        idx = (idx + 1) % funcs.size();
    }
    template<typename Key>
    void go() noexcept {
        auto& map = RMUC2026Map::instance();
        go(map.get<Key>(), Key::name);
    }
    void go(const Vec3& goal, std::string name) noexcept {
        current_goal_ = goal.head<2>();
        AWAKENING_INFO("go to {}: x: {} y: {} z: {}", name, goal.x(), goal.y(), goal.z());
    }
    void pub_goal_callback() noexcept {
        if (!current_goal_) {
            return;
        }
        if ((current_goal_.value() - current_pos_).norm() < 0.5) {
            return;
        }
        geometry_msgs::msg::PoseStamped msg;
        msg.header.stamp = rcl_node_.get_node()->now();
        msg.header.frame_id = "map";
        msg.pose.position.x = current_goal_.value().x();
        msg.pose.position.y = current_goal_.value().y();
        msg.pose.position.z = 0.0;
        goal_pub_->publish(msg);
    }
    GobalState::Pose sentry_pose = GobalState::Pose::Attack;
    std::optional<Eigen::Vector2d> current_goal_;
    std::thread pub_goal_thread_;
    std::thread tick_thread_;
    GobalState state_;
    Eigen::Vector2d current_pos_;
    auto_aim::ArmorTarget target_in_big_yaw_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    bool running_ = false;
    rcl::RclcppNode& rcl_node_;
    rcl::TF& rcl_tf_;
    std::shared_ptr<SerialDriver> serial_;
};
} // namespace awakening::sentry_brain