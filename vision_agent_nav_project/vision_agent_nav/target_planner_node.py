from __future__ import annotations

from dataclasses import replace

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from vision_agent_nav.core import (
    PlannerConfig,
    SceneState,
    plan_next_action,
    preferred_scan_angular_z,
)


class TargetPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("target_planner_node")
        self.declare_parameter("center_tolerance", 0.25)
        self.declare_parameter("near_area_ratio", 0.18)
        self.declare_parameter("approach_distance_m", 1.0)
        self.declare_parameter("target_reached_distance_m", 0.8)
        self.declare_parameter("target_far_distance_m", 1.5)
        self.config = PlannerConfig(
            center_tolerance=self.get_parameter("center_tolerance").get_parameter_value().double_value,
            near_area_ratio=self.get_parameter("near_area_ratio").get_parameter_value().double_value,
            approach_distance_m=self.get_parameter("approach_distance_m").get_parameter_value().double_value,
            target_reached_distance_m=self.get_parameter("target_reached_distance_m").get_parameter_value().double_value,
            target_far_distance_m=self.get_parameter("target_far_distance_m").get_parameter_value().double_value,
        )
        self.last_target_class: str | None = None
        self.last_visible_horizontal_offset: float | None = None
        self.publisher = self.create_publisher(String, "/agent/planner_intent", 10)
        self.subscription = self.create_subscription(String, "/perception/scene_state", self.on_scene_state, 10)

    def on_scene_state(self, msg: String) -> None:
        state = SceneState.from_json(msg.data)
        if state.target_class != self.last_target_class:
            self.last_target_class = state.target_class
            self.last_visible_horizontal_offset = None
        if state.visible:
            self.last_visible_horizontal_offset = state.horizontal_offset

        scan_angular_z = preferred_scan_angular_z(
            self.last_visible_horizontal_offset,
            self.config.scan_angular_z,
        )
        config = replace(self.config, scan_angular_z=scan_angular_z)
        intent = plan_next_action(state, config)
        self.publisher.publish(String(data=intent.to_json()))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TargetPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
