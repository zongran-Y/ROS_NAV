from __future__ import annotations

import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from vision_agent_nav.core import MetricsTracker, SceneState


class MetricsNode(Node):
    def __init__(self) -> None:
        super().__init__("metrics_node")
        self.declare_parameter("window_size", 100)
        self.declare_parameter("log_path", "logs/metrics.jsonl")
        self.tracker = MetricsTracker(
            window_size=self.get_parameter("window_size").get_parameter_value().integer_value
        )
        self.log_path = Path(self.get_parameter("log_path").get_parameter_value().string_value).expanduser()
        self.publisher = self.create_publisher(String, "/system/metrics", 10)
        self.create_subscription(String, "/perception/detections", self.on_detections, 10)
        self.create_subscription(String, "/perception/scene_state", self.on_scene_state, 10)
        self.create_subscription(String, "/agent/command", self.on_agent_command, 10)
        self.create_subscription(String, "/agent/status", self.on_agent_status, 10)
        self.timer = self.create_timer(1.0, self.publish_metrics)

    def on_detections(self, msg: String) -> None:
        payload = json.loads(msg.data)
        self.tracker.record_detection_frame(
            timestamp_s=float(payload.get("stamp", self.now_s())),
            detection_count=len(payload.get("detections", [])),
            inference_latency_ms=payload.get("inference_latency_ms"),
            depth_valid=payload.get("depth_valid"),
        )

    def on_scene_state(self, msg: String) -> None:
        state = SceneState.from_json(msg.data)
        self.tracker.record_scene_state(state.visible, self.now_s())

    def on_agent_command(self, msg: String) -> None:
        text = msg.data.strip().lower()
        if text.startswith(("find ", "search ", "go to ", "goto ", "navigate to ", "track ")):
            self.tracker.record_task_started(self.now_s())
        self.tracker.record_agent_tool_call()

    def on_agent_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        state = payload.get("state")
        if state == "reached":
            self.tracker.record_navigation_result(True, self.now_s())
        elif state in {"nav_failed", "nav_rejected", "failed"}:
            self.tracker.record_navigation_result(False, self.now_s())

    def publish_metrics(self) -> None:
        now = self.now_s()
        self.publisher.publish(String(data=self.tracker.to_json(now)))
        self.tracker.write_jsonl(self.log_path, now)

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MetricsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
