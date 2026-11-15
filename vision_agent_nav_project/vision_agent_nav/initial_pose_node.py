from __future__ import annotations

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
import math


def make_initial_pose_message(
    stamp: Time,
    x: float = -2.0,
    y: float = -0.5,
    yaw: float = 0.0,
) -> PoseWithCovarianceStamped:
    message = PoseWithCovarianceStamped()
    message.header.frame_id = "map"
    message.header.stamp = stamp
    message.pose.pose.position.x = x
    message.pose.pose.position.y = y
    message.pose.pose.orientation.z = math.sin(yaw / 2.0)
    message.pose.pose.orientation.w = math.cos(yaw / 2.0)
    message.pose.covariance[0] = 0.25
    message.pose.covariance[7] = 0.25
    message.pose.covariance[35] = 0.0685
    return message


class InitialPoseNode(Node):
    def __init__(self) -> None:
        super().__init__("initial_pose_node")
        self.declare_parameter("initial_x", -2.0)
        self.declare_parameter("initial_y", -0.5)
        self.declare_parameter("initial_yaw", 0.0)
        self.initial_x = float(self.get_parameter("initial_x").value)
        self.initial_y = float(self.get_parameter("initial_y").value)
        self.initial_yaw = float(self.get_parameter("initial_yaw").value)
        self.declare_parameter("publish_count", 3)
        self.declare_parameter("publish_period_s", 1.0)
        self.publisher = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self.remaining = max(1, int(self.get_parameter("publish_count").value))
        self.done = False
        period_s = max(0.1, float(self.get_parameter("publish_period_s").value))
        self.timer = self.create_timer(period_s, self.publish_initial_pose)

    def publish_initial_pose(self) -> None:
        self.publisher.publish(
            make_initial_pose_message(
                self.get_clock().now().to_msg(),
                self.initial_x,
                self.initial_y,
                self.initial_yaw,
            )
        )
        self.remaining -= 1
        if self.remaining <= 0:
            self.timer.cancel()
            self.done = True


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = InitialPoseNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        if node.context.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
