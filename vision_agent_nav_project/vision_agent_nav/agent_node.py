from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from vision_agent_nav.core import (
    LightweightStateMachine,
    PlannerIntent,
    SceneState,
    is_strong_visual_reach,
    parse_agent_command,
)


class LightweightAgentNode(Node):
    def __init__(self) -> None:
        super().__init__("lightweight_agent_node")
        self.target_class = "person"
        self.active = False
        self.latest_state: SceneState | None = None
        self.latest_intent: PlannerIntent | None = None
        self.machine = LightweightStateMachine(target_class=self.target_class)
        self.declare_parameter("max_subgoals", 3)
        self.declare_parameter("reach_confirmation_frames", 3)
        self.machine.max_subgoals = max(1, self.get_parameter("max_subgoals").get_parameter_value().integer_value)
        self.reach_confirmation_frames = max(
            1, self.get_parameter("reach_confirmation_frames").get_parameter_value().integer_value
        )
        self.scene_revision = 0
        self.awaiting_replan = False
        self.nav_result_scene_revision = 0
        self.reached_streak = 0

        self.status_pub = self.create_publisher(String, "/agent/status", 10)
        self.target_class_pub = self.create_publisher(String, "/agent/target_class", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.nav_goal_pub = self.create_publisher(String, "/agent/nav_goal", 10)
        self.nav_result_sub = self.create_subscription(String, "/agent/nav_result", self.on_nav_result, 10)

        self.create_subscription(String, "/agent/command", self.on_command, 10)
        self.create_subscription(String, "/perception/scene_state", self.on_scene_state, 10)
        self.create_subscription(String, "/agent/planner_intent", self.on_planner_intent, 10)
        self.timer = self.create_timer(0.5, self.tick)

    def on_command(self, msg: String) -> None:
        command = parse_agent_command(msg.data)
        if command.tool == "search_target" and command.target_class:
            self.target_class = command.target_class
            self.machine.start(self.now_s(), self.target_class)
            self.active = True
            self.awaiting_replan = False
            # A new experiment/task must not reuse a planner intent or scene
            # from the previous task.  Otherwise the 0.5 s timer can send a
            # stale Nav2 goal before the first fresh camera observation.
            self.latest_state = None
            self.latest_intent = None
            self.scene_revision = 0
            self.nav_result_scene_revision = 0
            self.reached_streak = 0
            self.target_class_pub.publish(String(data=self.target_class))
            self.publish_status("searching", f"search target set to {self.target_class}")
        elif command.tool == "stop_robot":
            self.active = False
            self.awaiting_replan = False
            self.machine.stop()
            self.stop_robot()
            self.publish_status("stopped", "robot stopped by command")
        elif command.tool == "get_status":
            self.publish_status("status", self.describe_status())
        else:
            self.publish_status("unknown_command", msg.data)

    def on_scene_state(self, msg: String) -> None:
        self.latest_state = SceneState.from_json(msg.data)
        self.scene_revision += 1
        if self.latest_state.suggested_action == "reached":
            self.reached_streak += 1
        else:
            self.reached_streak = 0
        if self.awaiting_replan and self.scene_revision > self.nav_result_scene_revision:
            self.awaiting_replan = False
        if (
            self.active
            and self.latest_state.target_class == self.target_class
            and is_strong_visual_reach(self.latest_state)
        ):
            self.reached_streak = self.reach_confirmation_frames
            self.machine.update(self.latest_state, self.now_s())
            self.active = False
            self.awaiting_replan = False
            self.stop_robot()
            self.publish_status("reached", "strong RGB-D reach")

    def on_planner_intent(self, msg: String) -> None:
        self.latest_intent = PlannerIntent.from_json(msg.data)

    def on_nav_result(self, msg: String) -> None:
        payload = json.loads(msg.data)
        state = str(payload.get("state", ""))
        detail = str(payload.get("detail", state))
        if state == "nav_succeeded" and self.machine.state == "APPROACHING":
            next_state = (
                self.machine.on_nav_subgoal_succeeded(
                    self.latest_state,
                    self.now_s(),
                    self.reached_streak >= self.reach_confirmation_frames,
                )
                if self.latest_state
                else None
            )
            if next_state is not None and next_state.state == "REACHED":
                self.active = False
                self.awaiting_replan = False
                self.stop_robot()
                self.publish_status("reached", next_state.reason)
            elif next_state is not None and next_state.state == "FAILED":
                self.active = False
                self.awaiting_replan = False
                self.stop_robot()
                self.publish_status("failed", next_state.reason)
            else:
                self.active = True
                self.awaiting_replan = True
                self.nav_result_scene_revision = self.scene_revision
                self.publish_status("nav_subgoal_succeeded", detail)
                self.publish_status("replanning", "waiting for a fresh scene after Nav2 subgoal")
        elif state in {"nav_failed", "nav_rejected"}:
            self.active = False
            self.awaiting_replan = False
            self.stop_robot()
            self.machine.state = "FAILED"
            self.machine.failure_reason = detail
            self.publish_status("failed", detail)

    def tick(self) -> None:
        if (
            self.machine.state == "APPROACHING"
            and self.machine.started_at_s is not None
            and self.now_s() - self.machine.started_at_s > self.machine.timeout_s
        ):
            self.active = False
            self.stop_robot()
            self.machine.state = "FAILED"
            self.publish_status("failed", "task timed out during navigation")
            return
        if not self.active or self.awaiting_replan or self.latest_intent is None:
            return

        intent = self.latest_intent
        if intent.target_class != self.target_class:
            return

        if intent.action in {"scan", "turn_left", "turn_right"}:
            if self.latest_state is not None:
                machine_state = self.machine.update(self.latest_state, self.now_s())
                if machine_state.state == "FAILED":
                    self.active = False
                    self.stop_robot()
                    self.publish_status("failed", machine_state.reason)
                    return
            twist = Twist()
            twist.angular.z = intent.angular_z
            self.cmd_vel_pub.publish(twist)
            self.publish_status(intent.action, intent.reason)
        elif intent.action == "navigate_forward":
            if self.latest_state is not None:
                self.machine.update(self.latest_state, self.now_s())
            self.stop_robot()
            self.nav_goal_pub.publish(String(data=json.dumps({"forward_m": intent.forward_m})))
            self.publish_status("navigate_forward", f"sent Nav2 forward goal {intent.forward_m:.2f}m")
            self.active = False
        elif intent.action == "reached":
            if self.latest_state is not None and self.reached_streak >= self.reach_confirmation_frames:
                self.machine.update(self.latest_state, self.now_s())
            self.stop_robot()
            if self.reached_streak >= self.reach_confirmation_frames:
                self.publish_status("reached", "target is centered and close for stable frames")
                self.active = False
            else:
                self.publish_status(
                    "confirming_reached",
                    f"waiting for {self.reach_confirmation_frames} stable reached frames",
                )

    def stop_robot(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    def describe_status(self) -> str:
        if self.latest_state is None:
            return f"target={self.target_class}, no scene state received"
        return (
            f"target={self.target_class}, visible={self.latest_state.visible}, "
            f"track_id={self.latest_state.track_id}, "
            f"offset={self.latest_state.horizontal_offset:.2f}, "
            f"area={self.latest_state.area_ratio:.3f}, "
            f"distance={self.latest_state.estimated_distance_m}"
        )

    def publish_status(self, state: str, detail: str) -> None:
        machine_state = self.machine.state
        self.status_pub.publish(
            String(data=json.dumps({"state": state, "machine_state": machine_state, "detail": detail}, ensure_ascii=False))
        )

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LightweightAgentNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
