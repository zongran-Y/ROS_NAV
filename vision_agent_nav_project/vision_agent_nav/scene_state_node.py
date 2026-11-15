from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from vision_agent_nav.core import Detection, PlannerConfig, TargetRankConfig, build_scene_state


class SceneStateNode(Node):
    def __init__(self) -> None:
        super().__init__("scene_state_node")
        self.declare_parameter("target_class", "person")
        self.declare_parameter("center_tolerance", 0.25)
        self.declare_parameter("near_area_ratio", 0.18)
        self.declare_parameter("medium_area_ratio", 0.08)
        self.declare_parameter("rank_confidence_weight", 0.35)
        self.declare_parameter("rank_center_weight", 0.30)
        self.declare_parameter("rank_depth_weight", 0.20)
        self.declare_parameter("rank_area_weight", 0.10)
        self.declare_parameter("rank_class_match_weight", 0.05)
        self.declare_parameter("rank_depth_max_m", 6.0)
        self.declare_parameter("rank_max_area_ratio", 0.30)
        self.target_class = self.get_parameter("target_class").get_parameter_value().string_value
        self.config = PlannerConfig(
            center_tolerance=self.get_parameter("center_tolerance").get_parameter_value().double_value,
            near_area_ratio=self.get_parameter("near_area_ratio").get_parameter_value().double_value,
            medium_area_ratio=self.get_parameter("medium_area_ratio").get_parameter_value().double_value,
        )
        self.rank_config = TargetRankConfig(
            confidence_weight=self.get_parameter("rank_confidence_weight").get_parameter_value().double_value,
            center_weight=self.get_parameter("rank_center_weight").get_parameter_value().double_value,
            depth_weight=self.get_parameter("rank_depth_weight").get_parameter_value().double_value,
            area_weight=self.get_parameter("rank_area_weight").get_parameter_value().double_value,
            class_match_weight=self.get_parameter("rank_class_match_weight").get_parameter_value().double_value,
            depth_max_m=self.get_parameter("rank_depth_max_m").get_parameter_value().double_value,
            max_area_ratio=self.get_parameter("rank_max_area_ratio").get_parameter_value().double_value,
        )

        self.publisher = self.create_publisher(String, "/perception/scene_state", 10)
        self.subscription = self.create_subscription(String, "/perception/detections", self.on_detections, 10)
        self.create_subscription(String, "/agent/target_class", self.on_target_class, 10)

    def on_target_class(self, msg: String) -> None:
        target = msg.data.strip().lower()
        if target:
            self.target_class = target
            self.get_logger().info(f"scene target class set to {self.target_class}")

    def on_detections(self, msg: String) -> None:
        payload = json.loads(msg.data)
        detections = [Detection.from_dict(item) for item in payload.get("detections", [])]
        state = build_scene_state(detections, self.target_class, self.config, rank_config=self.rank_config)
        self.publisher.publish(String(data=state.to_json()))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SceneStateNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
