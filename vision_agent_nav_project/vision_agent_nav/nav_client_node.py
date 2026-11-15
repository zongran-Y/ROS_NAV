from __future__ import annotations

import json
import math

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from vision_agent_nav.core import navigation_goal_available


def yaw_from_quaternion(z: float, w: float) -> float:
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


def quaternion_from_yaw(yaw: float) -> tuple[float, float]:
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class NavClientNode(Node):
    def __init__(self) -> None:
        super().__init__("nav_client_node")
        self.current_odom: Odometry | None = None
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.active_goal_handle = None
        self.cancel_pending = False
        self.goal_request_pending = False
        self.goal_started_at_s: float | None = None
        self.last_feedback: dict[str, float | int] = {}
        self.status_pub = self.create_publisher(String, "/agent/status", 10)
        self.result_pub = self.create_publisher(String, "/agent/nav_result", 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(String, "/agent/nav_goal", self.on_nav_goal, 10)
        self.create_subscription(String, "/agent/command", self.on_command, 10)

    def on_odom(self, msg: Odometry) -> None:
        self.current_odom = msg

    def on_command(self, msg: String) -> None:
        if msg.data.strip().lower() in {"stop", "halt", "cancel"}:
            self.cancel_active_goal()

    def current_map_pose(self) -> tuple[float, float, float] | None:
        for base_frame in ("base_link", "base_footprint"):
            try:
                transform = self.tf_buffer.lookup_transform("map", base_frame, Time())
            except TransformException:
                continue
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            return translation.x, translation.y, yaw_from_quaternion(rotation.z, rotation.w)
        return None

    def on_nav_goal(self, msg: String) -> None:
        if not navigation_goal_available(
            self.active_goal_handle is not None,
            self.cancel_pending,
            self.goal_request_pending,
        ):
            self.publish_result("nav_rejected", "a Nav2 goal is already active")
            return

        forward_m = float(json.loads(msg.data).get("forward_m", 1.0))
        current_pose = self.current_map_pose()
        if current_pose is None:
            self.publish_result("nav_failed", "map pose unavailable; localization or TF is not ready")
            return
        x, y, yaw = current_pose

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x + math.cos(yaw) * forward_m
        goal.pose.pose.position.y = y + math.sin(yaw) * forward_m
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = quaternion_from_yaw(yaw)

        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.publish_result("nav_failed", "navigate_to_pose action server unavailable")
            return

        self.goal_started_at_s = self.get_clock().now().nanoseconds / 1e9
        self.last_feedback = {}
        self.goal_request_pending = True
        future = self.nav_client.send_goal_async(goal, feedback_callback=self.on_feedback)
        future.add_done_callback(self.on_goal_response)
        self.publish_status("nav_goal_sent", f"forward goal {forward_m:.2f}m in map frame")

    def on_goal_response(self, future) -> None:
        goal_handle = future.result()
        self.goal_request_pending = False
        if not goal_handle.accepted:
            self.publish_result("nav_rejected", "Nav2 rejected goal")
            return
        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_goal_result)

    def on_feedback(self, feedback_message) -> None:
        feedback = feedback_message.feedback
        self.last_feedback = {
            "distance_remaining_m": float(feedback.distance_remaining),
            "recoveries": int(feedback.number_of_recoveries),
            "navigation_time_s": float(feedback.navigation_time.sec) + feedback.navigation_time.nanosec / 1e9,
        }

    def on_goal_result(self, future) -> None:
        status = future.result().status
        elapsed = None
        if self.goal_started_at_s is not None:
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self.goal_started_at_s
        self.active_goal_handle = None
        self.goal_started_at_s = None
        self.cancel_pending = False
        detail_payload = {"status": int(status), "elapsed_s": elapsed, **self.last_feedback}
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.publish_result("nav_succeeded", json.dumps(detail_payload, ensure_ascii=False))
        else:
            self.publish_result("nav_failed", f"Nav2 finished with status {status}; details={json.dumps(detail_payload, ensure_ascii=False)}")

    def cancel_active_goal(self) -> None:
        if self.active_goal_handle is None:
            return
        self.cancel_pending = True
        future = self.active_goal_handle.cancel_goal_async()
        future.add_done_callback(self.on_cancel_complete)

    def on_cancel_complete(self, _future) -> None:
        self.active_goal_handle = None
        self.cancel_pending = False
        self.goal_started_at_s = None

    def publish_result(self, state: str, detail: str) -> None:
        payload = json.dumps({"state": state, "detail": detail}, ensure_ascii=False)
        self.result_pub.publish(String(data=payload))

    def publish_status(self, state: str, detail: str) -> None:
        self.status_pub.publish(String(data=json.dumps({"state": state, "detail": detail}, ensure_ascii=False)))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = NavClientNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
