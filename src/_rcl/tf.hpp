#pragma once
#include "node.hpp"
#include "utils/common/type_common.hpp"
#include "utils/runtime_tf.hpp"
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <rclcpp/clock.hpp>
#include <rclcpp/node.hpp>
#include <string>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
namespace awakening::rcl {
class TF {
public:
    TF(RclcppNode& node) {
        node_ = node.get_node();
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node_->get_clock());
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, node_);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
    }
    std::optional<ISO3> lookup_transform(
        const std::string& target_frame,
        const std::string& source_frame,
        rclcpp::Time time
    ) {
        ISO3 pose;
        try {
            auto tf =
                tf_buffer_
                    ->lookupTransform(target_frame, source_frame, time, tf2::durationFromSec(0.1));
            pose.translation() << tf.transform.translation.x, tf.transform.translation.y,
                tf.transform.translation.z;
            auto q = tf.transform.rotation;
            Eigen::Quaterniond Q(q.w, q.x, q.y, q.z);
            pose.linear() = Q.toRotationMatrix();
        } catch (tf2::TransformException& ex) {
            return std::nullopt;
        }
        return pose;
    }
    template<typename FrameEnum, size_t N, bool Static, typename F>
    void pub_robot_tf(const utils::tf::RobotTF<FrameEnum, N, Static>& r_tf, F&& get_frame_name) {
        auto now = node_->now();
        auto edges = r_tf.get_edges();
        for (const auto& edge: edges) {
            auto parent = edge.parent;
            auto child = edge.child;
            ISO3 pose = r_tf.pose_a_in_b(
                static_cast<FrameEnum>(child),
                static_cast<FrameEnum>(parent),
                Clock::now()
            );

            geometry_msgs::msg::TransformStamped t_msg;
            t_msg.header.stamp = now;
            t_msg.header.frame_id = get_frame_name(static_cast<FrameEnum>(parent));
            t_msg.child_frame_id = get_frame_name(static_cast<FrameEnum>(child));

            Eigen::Vector3d trans = pose.translation();
            t_msg.transform.translation.x = trans.x();
            t_msg.transform.translation.y = trans.y();
            t_msg.transform.translation.z = trans.z();

            Eigen::Quaterniond q(pose.linear());
            t_msg.transform.rotation.x = q.x();
            t_msg.transform.rotation.y = q.y();
            t_msg.transform.rotation.z = q.z();
            t_msg.transform.rotation.w = q.w();
            tf_broadcaster_->sendTransform(t_msg);
        }
    }

    std::shared_ptr<rclcpp::Node> node_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
};
} // namespace awakening::rcl