from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    frame_id: str = "camera_optical_frame"


@dataclass(frozen=True)
class Position3D:
    x: float
    y: float
    z: float
    frame: str = "camera_optical_frame"
    source: str = "rgbd"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Position3D":
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data["z"]),
            frame=str(data.get("frame", "camera_optical_frame")),
            source=str(data.get("source", "rgbd")),
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "frame": self.frame,
            "source": self.source,
        }


@dataclass(frozen=True)
class TargetRank:
    score: float
    factors: dict[str, float]


@dataclass(frozen=True)
class TargetRankConfig:
    confidence_weight: float = 0.35
    center_weight: float = 0.30
    depth_weight: float = 0.20
    area_weight: float = 0.10
    class_match_weight: float = 0.05
    depth_max_m: float = 6.0
    max_area_ratio: float = 0.30


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: list[float]
    image_width: int
    image_height: int
    track_id: int | None = None
    estimated_distance_m: float | None = None
    distance_source: str = "monocular"
    position_3d_m: Position3D | None = None
    rank_score: float = 0.0
    rank_reason: dict[str, float] | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def area_ratio(self) -> float:
        x1, y1, x2, y2 = self.bbox
        image_area = max(float(self.image_width * self.image_height), 1.0)
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return box_area / image_area

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        position_data = data.get("position_3d_m")
        return cls(
            class_name=str(data["class_name"]),
            confidence=float(data["confidence"]),
            bbox=[float(value) for value in data["bbox"]],
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            track_id=data.get("track_id"),
            estimated_distance_m=data.get("estimated_distance_m"),
            distance_source=str(data.get("distance_source", "monocular")),
            position_3d_m=Position3D.from_dict(position_data) if position_data else None,
            rank_score=float(data.get("rank_score", 0.0)),
            rank_reason={str(key): float(value) for key, value in data.get("rank_reason", {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        cx, cy = self.center
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "center": [cx, cy],
            "area_ratio": self.area_ratio,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "track_id": self.track_id,
            "estimated_distance_m": self.estimated_distance_m,
            "distance_source": self.distance_source,
            "position_3d_m": self.position_3d_m.to_dict() if self.position_3d_m else None,
            "rank_score": self.rank_score,
            "rank_reason": self.rank_reason or {},
        }


@dataclass(frozen=True)
class PlannerConfig:
    center_tolerance: float = 0.25
    near_area_ratio: float = 0.18
    medium_area_ratio: float = 0.08
    approach_distance_m: float = 1.0
    target_reached_distance_m: float = 0.8
    target_far_distance_m: float = 1.5
    scan_angular_z: float = 0.35
    turn_angular_z: float = 0.25


@dataclass(frozen=True)
class SceneState:
    target_class: str
    visible: bool
    class_name: str | None = None
    track_id: int | None = None
    confidence: float = 0.0
    bbox: list[float] | None = None
    center: list[float] | None = None
    horizontal_offset: float = 0.0
    vertical_offset: float = 0.0
    area_ratio: float = 0.0
    estimated_distance_m: float | None = None
    distance_source: str = "unknown"
    position_3d_m: dict[str, float | str] | None = None
    rank_score: float = 0.0
    rank_reason: dict[str, float] | None = None
    distance_level: str = "unknown"
    suggested_action: str = "scan"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "SceneState":
        return cls(**json.loads(payload))


@dataclass(frozen=True)
class PlannerIntent:
    action: str
    reason: str
    target_class: str
    angular_z: float = 0.0
    forward_m: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "PlannerIntent":
        return cls(**json.loads(payload))


@dataclass(frozen=True)
class AgentCommand:
    tool: str
    target_class: str | None = None


@dataclass(frozen=True)
class AgentState:
    state: str
    target_class: str
    reason: str


@dataclass
class _Track:
    track_id: int
    detection: Detection
    missed: int = 0


class SimpleTracker:
    """Small IoU tracker for giving detections stable IDs without external deps."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 3) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: list[_Track] = []

    def update(self, detections: list[Detection]) -> list[Detection]:
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        output: list[Detection] = []

        pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            for det_index, detection in enumerate(detections):
                if track.detection.class_name == detection.class_name:
                    pairs.append((bbox_iou(track.detection.bbox, detection.bbox), track_index, det_index))

        for score, track_index, det_index in sorted(pairs, reverse=True):
            if score < self.iou_threshold or track_index in assigned_tracks or det_index in assigned_detections:
                continue
            track = self.tracks[track_index]
            updated = replace_detection_id(detections[det_index], track.track_id)
            self.tracks[track_index] = _Track(track.track_id, updated, missed=0)
            assigned_tracks.add(track_index)
            assigned_detections.add(det_index)
            output.append(updated)

        retained: list[_Track] = []
        for track_index, track in enumerate(self.tracks):
            if track_index in assigned_tracks:
                retained.append(self.tracks[track_index])
            else:
                track.missed += 1
                if track.missed <= self.max_missed:
                    retained.append(track)
        self.tracks = retained

        for det_index, detection in enumerate(detections):
            if det_index in assigned_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            updated = replace_detection_id(detection, track_id)
            self.tracks.append(_Track(track_id, updated, missed=0))
            output.append(updated)

        return sorted(output, key=lambda item: item.track_id or 0)


class DistanceEstimator:
    """Monocular pinhole distance estimator using object height in pixels."""

    def __init__(
        self,
        focal_length_px: float,
        default_object_height_m: float = 1.7,
        class_heights_m: dict[str, float] | None = None,
    ) -> None:
        self.focal_length_px = focal_length_px
        self.default_object_height_m = default_object_height_m
        self.class_heights_m = class_heights_m or {}

    def estimate(self, bbox: list[float], class_name: str | None = None) -> float | None:
        x1, y1, x2, y2 = bbox
        pixel_height = max(0.0, y2 - y1)
        if pixel_height <= 0:
            return None
        real_height = self.class_heights_m.get(class_name or "", self.default_object_height_m)
        return round((real_height * self.focal_length_px) / pixel_height, 3)


class LightweightStateMachine:
    """Task-level Agent state machine for explainable target search behavior."""

    def __init__(self, target_class: str = "person", timeout_s: float = 30.0, max_subgoals: int = 3) -> None:
        self.target_class = target_class
        self.timeout_s = timeout_s
        self.max_subgoals = max(1, int(max_subgoals))
        self.state = "IDLE"
        self.started_at_s: float | None = None
        self.subgoal_count = 0
        self.failure_reason = ""

    def start(self, timestamp_s: float, target_class: str | None = None) -> AgentState:
        if target_class:
            self.target_class = target_class
        self.started_at_s = timestamp_s
        self.state = "SEARCHING"
        self.subgoal_count = 0
        self.failure_reason = ""
        return AgentState(self.state, self.target_class, "task started")

    def stop(self) -> AgentState:
        self.state = "IDLE"
        self.started_at_s = None
        self.subgoal_count = 0
        self.failure_reason = ""
        return AgentState(self.state, self.target_class, "task stopped")

    def on_nav_subgoal_succeeded(
        self,
        state: SceneState,
        timestamp_s: float,
        visual_reached_confirmed: bool = True,
    ) -> AgentState:
        if self.started_at_s is not None and timestamp_s - self.started_at_s > self.timeout_s:
            self.state = "FAILED"
            self.failure_reason = "task timed out after navigation subgoal"
            return AgentState(self.state, self.target_class, self.failure_reason)
        if state.target_class != self.target_class:
            self.state = "FAILED"
            self.failure_reason = "navigation result target class mismatch"
            return AgentState(self.state, self.target_class, self.failure_reason)
        if state.suggested_action == "reached" and visual_reached_confirmed:
            self.state = "REACHED"
            return AgentState(self.state, self.target_class, "visual target reached")
        self.subgoal_count += 1
        if self.subgoal_count >= self.max_subgoals:
            self.state = "FAILED"
            self.failure_reason = "maximum navigation subgoals reached without visual success"
            return AgentState(self.state, self.target_class, self.failure_reason)
        self.state = "SEARCHING"
        return AgentState(self.state, self.target_class, "replan after navigation subgoal")

    def update(self, state: SceneState, timestamp_s: float) -> AgentState:
        if self.started_at_s is not None and timestamp_s - self.started_at_s > self.timeout_s:
            self.state = "FAILED"
            self.failure_reason = "task timed out"
            return AgentState(self.state, self.target_class, "task timed out")
        if self.state == "IDLE":
            return AgentState(self.state, self.target_class, "waiting for command")
        if not state.visible:
            self.state = "SEARCHING"
            return AgentState(self.state, self.target_class, "target not visible")
        if state.suggested_action in {"turn_left", "turn_right"}:
            self.state = "CENTERING"
            return AgentState(self.state, self.target_class, "centering visible target")
        if state.suggested_action == "navigate_forward":
            self.state = "APPROACHING"
            return AgentState(self.state, self.target_class, "approaching target")
        if state.suggested_action == "reached":
            self.state = "REACHED"
            return AgentState(self.state, self.target_class, "target reached")
        return AgentState(self.state, self.target_class, "state unchanged")


class MetricsTracker:
    """Collects portfolio-friendly runtime metrics from ROS2 topic events."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.detection_timestamps: deque[float] = deque(maxlen=window_size)
        self.inference_latencies_ms: deque[float] = deque(maxlen=window_size)
        self.detection_counts: deque[int] = deque(maxlen=window_size)
        self.depth_valid_samples: deque[bool] = deque(maxlen=window_size)
        self.visibility_samples: deque[bool] = deque(maxlen=window_size)
        self.task_start_time_s: float | None = None
        self.first_detection_time_s: float | None = None
        self.completed_task_durations_s: list[float] = []
        self.navigation_successes = 0
        self.navigation_failures = 0
        self.agent_tool_call_count = 0

    def record_detection_frame(
        self,
        timestamp_s: float,
        detection_count: int,
        inference_latency_ms: float | None = None,
        depth_valid: bool | None = None,
    ) -> None:
        self.detection_timestamps.append(timestamp_s)
        self.detection_counts.append(detection_count)
        if inference_latency_ms is not None:
            self.inference_latencies_ms.append(inference_latency_ms)
        if depth_valid is not None:
            self.depth_valid_samples.append(bool(depth_valid))

    def record_scene_state(self, visible: bool, timestamp_s: float) -> None:
        self.visibility_samples.append(visible)
        if visible and self.task_start_time_s is not None and self.first_detection_time_s is None:
            self.first_detection_time_s = timestamp_s - self.task_start_time_s

    def record_task_started(self, timestamp_s: float) -> None:
        self.detection_timestamps.clear()
        self.inference_latencies_ms.clear()
        self.detection_counts.clear()
        self.depth_valid_samples.clear()
        self.visibility_samples.clear()
        self.completed_task_durations_s.clear()
        self.navigation_successes = 0
        self.navigation_failures = 0
        self.task_start_time_s = timestamp_s
        self.first_detection_time_s = None

    def record_task_completed(self, timestamp_s: float) -> None:
        if self.task_start_time_s is not None:
            self.completed_task_durations_s.append(timestamp_s - self.task_start_time_s)
        self.task_start_time_s = None

    def record_navigation_result(self, success: bool, timestamp_s: float) -> None:
        if success:
            self.navigation_successes += 1
            self.record_task_completed(timestamp_s)
        else:
            self.navigation_failures += 1

    def record_agent_tool_call(self) -> None:
        self.agent_tool_call_count += 1

    def snapshot(self, now_s: float | None = None) -> dict[str, float | int | None]:
        return {
            "perception_fps": self._perception_fps(),
            "inference_latency_ms": self._average(self.inference_latencies_ms),
            "detection_count": sum(self.detection_counts),
            "detection_frame_count": len(self.detection_counts),
            "depth_valid_rate": self._depth_valid_rate(),
            "target_visible_rate": self._visible_rate(),
            "first_detection_time_s": self.first_detection_time_s,
            "navigation_success_rate": self._navigation_success_rate(),
            "average_task_time_s": self._average(self.completed_task_durations_s),
            "agent_tool_call_count": self.agent_tool_call_count,
        }

    def to_json(self, now_s: float | None = None) -> str:
        return json.dumps(self.snapshot(now_s), ensure_ascii=False)

    def write_jsonl(self, path: str | Path, now_s: float | None = None) -> None:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.snapshot(now_s)
        payload["logged_at_s"] = now_s
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _perception_fps(self) -> float | None:
        if len(self.detection_timestamps) < 2:
            return None
        duration = self.detection_timestamps[-1] - self.detection_timestamps[0]
        if duration <= 0:
            return None
        return round((len(self.detection_timestamps) - 1) / duration, 3)

    def _depth_valid_rate(self) -> float | None:
        if not self.depth_valid_samples:
            return None
        return round(sum(self.depth_valid_samples) / len(self.depth_valid_samples), 3)

    def _visible_rate(self) -> float:
        if not self.visibility_samples:
            return 0.0
        visible_count = sum(1 for item in self.visibility_samples if item)
        return round(visible_count / len(self.visibility_samples), 3)

    def _navigation_success_rate(self) -> float:
        total = self.navigation_successes + self.navigation_failures
        if total == 0:
            return 0.0
        return round(self.navigation_successes / total, 3)

    @staticmethod
    def _average(values) -> float | None:
        if not values:
            return None
        return round(sum(values) / len(values), 3)


def distance_level(area_ratio: float, config: PlannerConfig | None = None) -> str:
    cfg = config or PlannerConfig()
    if area_ratio >= cfg.near_area_ratio:
        return "near"
    if area_ratio >= cfg.medium_area_ratio:
        return "medium"
    return "far"


def deproject_pixel_to_3d(
    pixel_x: float,
    pixel_y: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> Position3D:
    if depth_m <= 0:
        raise ValueError("depth_m must be positive")
    if intrinsics.fx <= 0 or intrinsics.fy <= 0:
        raise ValueError("camera focal lengths must be positive")
    x = (pixel_x - intrinsics.cx) * depth_m / intrinsics.fx
    y = (pixel_y - intrinsics.cy) * depth_m / intrinsics.fy
    return Position3D(
        x=round(x, 4),
        y=round(y, 4),
        z=round(depth_m, 4),
        frame=intrinsics.frame_id,
        source="rgbd",
    )


def nearest_timestamp_index(target_s: float, timestamps: list[float], max_age_s: float) -> int | None:
    """Return the closest timestamp index when it is within the allowed age."""
    if not timestamps or max_age_s < 0:
        return None
    index = min(range(len(timestamps)), key=lambda item: abs(timestamps[item] - target_s))
    return index if abs(timestamps[index] - target_s) <= max_age_s else None


def inference_is_due(now_s: float, last_inference_s: float | None, min_period_s: float) -> bool:
    """Return whether a new inference may start without processing stale frames."""
    if min_period_s < 0:
        raise ValueError("min_period_s must be non-negative")
    return last_inference_s is None or now_s - last_inference_s >= min_period_s - 1e-9


def navigation_goal_available(active: bool, cancel_pending: bool, request_pending: bool) -> bool:
    return not active and not cancel_pending and not request_pending


def detection_offsets(detection: Detection) -> tuple[float, float]:
    cx, cy = detection.center
    half_width = max(detection.image_width / 2.0, 1.0)
    half_height = max(detection.image_height / 2.0, 1.0)
    return (cx - half_width) / half_width, (cy - half_height) / half_height


def rank_detection(
    detection: Detection,
    target_class: str,
    config: TargetRankConfig | None = None,
) -> TargetRank:
    cfg = config or TargetRankConfig()
    class_match = 1.0 if detection.class_name == target_class else 0.0
    confidence = max(0.0, min(float(detection.confidence), 1.0))
    horizontal_offset, vertical_offset = detection_offsets(detection)
    normalized_offset = min(
        ((horizontal_offset * horizontal_offset + vertical_offset * vertical_offset) ** 0.5) / (2.0 ** 0.5),
        1.0,
    )
    center = 1.0 - normalized_offset
    if detection.estimated_distance_m is not None and detection.estimated_distance_m > 0:
        depth = 1.0 - min(detection.estimated_distance_m / max(cfg.depth_max_m, 0.001), 1.0)
        if detection.distance_source == "depth" or detection.position_3d_m is not None:
            depth = min(depth + 0.15, 1.0)
    else:
        depth = 0.0
    area = min(detection.area_ratio / max(cfg.max_area_ratio, 0.001), 1.0)
    factors = {
        "class_match": round(class_match, 4),
        "confidence": round(confidence, 4),
        "center": round(center, 4),
        "depth": round(depth, 4),
        "area": round(area, 4),
    }
    score = (
        cfg.class_match_weight * class_match
        + cfg.confidence_weight * confidence
        + cfg.center_weight * center
        + cfg.depth_weight * depth
        + cfg.area_weight * area
    )
    return TargetRank(score=round(score, 4), factors=factors)


def with_rank(
    detection: Detection,
    target_class: str,
    config: TargetRankConfig | None = None,
) -> Detection:
    rank = rank_detection(detection, target_class, config)
    return Detection(
        class_name=detection.class_name,
        confidence=detection.confidence,
        bbox=detection.bbox,
        image_width=detection.image_width,
        image_height=detection.image_height,
        track_id=detection.track_id,
        estimated_distance_m=detection.estimated_distance_m,
        distance_source=detection.distance_source,
        position_3d_m=detection.position_3d_m,
        rank_score=rank.score,
        rank_reason=rank.factors,
    )


def choose_target(
    detections: list[Detection],
    target_class: str,
    rank_config: TargetRankConfig | None = None,
) -> Detection | None:
    candidates = [item for item in detections if item.class_name == target_class]
    if not candidates:
        return None
    ranked = [with_rank(item, target_class, rank_config) for item in candidates]
    return max(ranked, key=lambda item: item.rank_score)


def bbox_iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def replace_detection_id(detection: Detection, track_id: int) -> Detection:
    return Detection(
        class_name=detection.class_name,
        confidence=detection.confidence,
        bbox=detection.bbox,
        image_width=detection.image_width,
        image_height=detection.image_height,
        track_id=track_id,
        estimated_distance_m=detection.estimated_distance_m,
        distance_source=detection.distance_source,
        position_3d_m=detection.position_3d_m,
        rank_score=detection.rank_score,
        rank_reason=detection.rank_reason,
    )


def build_scene_state(
    detections: list[Detection],
    target_class: str,
    config: PlannerConfig | None = None,
    distance_estimator: DistanceEstimator | None = None,
    rank_config: TargetRankConfig | None = None,
) -> SceneState:
    cfg = config or PlannerConfig()
    target = choose_target(detections, target_class, rank_config)
    if target is None:
        return SceneState(target_class=target_class, visible=False, suggested_action="scan")

    cx, cy = target.center
    horizontal_offset, vertical_offset = detection_offsets(target)
    estimated_distance = target.estimated_distance_m
    distance_source = target.distance_source if estimated_distance is not None else "unknown"
    if estimated_distance is None and distance_estimator is not None:
        estimated_distance = distance_estimator.estimate(target.bbox, target.class_name)
        distance_source = "monocular" if estimated_distance is not None else "unknown"
    level = distance_level(target.area_ratio, cfg)
    suggested = plan_next_action(
        SceneState(
            target_class=target_class,
            visible=True,
            class_name=target.class_name,
            track_id=target.track_id,
            confidence=target.confidence,
            bbox=target.bbox,
            center=[cx, cy],
            horizontal_offset=horizontal_offset,
            vertical_offset=vertical_offset,
            area_ratio=target.area_ratio,
            estimated_distance_m=estimated_distance,
            distance_source=distance_source,
            position_3d_m=target.position_3d_m.to_dict() if target.position_3d_m else None,
            rank_score=target.rank_score,
            rank_reason=target.rank_reason or {},
            distance_level=level,
            suggested_action="scan",
        ),
        cfg,
    ).action

    return SceneState(
        target_class=target_class,
        visible=True,
        class_name=target.class_name,
        track_id=target.track_id,
        confidence=target.confidence,
        bbox=target.bbox,
        center=[cx, cy],
        horizontal_offset=horizontal_offset,
        vertical_offset=vertical_offset,
        area_ratio=target.area_ratio,
        estimated_distance_m=estimated_distance,
        distance_source=distance_source,
        position_3d_m=target.position_3d_m.to_dict() if target.position_3d_m else None,
        rank_score=target.rank_score,
        rank_reason=target.rank_reason or {},
        distance_level=level,
        suggested_action=suggested,
    )


def plan_next_action(state: SceneState, config: PlannerConfig | None = None) -> PlannerIntent:
    cfg = config or PlannerConfig()
    if not state.visible:
        return PlannerIntent("scan", "target is not visible", state.target_class, angular_z=cfg.scan_angular_z)

    if state.horizontal_offset < -cfg.center_tolerance:
        return PlannerIntent("turn_left", "target is left of image center", state.target_class, angular_z=cfg.turn_angular_z)

    if state.horizontal_offset > cfg.center_tolerance:
        return PlannerIntent("turn_right", "target is right of image center", state.target_class, angular_z=-cfg.turn_angular_z)

    if state.estimated_distance_m is not None:
        if state.estimated_distance_m <= cfg.target_reached_distance_m:
            return PlannerIntent("reached", "target is centered and close by depth", state.target_class)
        # Depth pixels can occasionally sample the background or a hole in the
        # simulated depth image.  A large, centered target is a safer close-range
        # signal than a single implausible far-depth sample.
        if state.area_ratio >= cfg.near_area_ratio:
            return PlannerIntent("reached", "target is centered and close by image area", state.target_class)
        if state.estimated_distance_m > cfg.target_far_distance_m:
            return PlannerIntent(
                "navigate_forward",
                "target is centered but still far by depth",
                state.target_class,
                forward_m=cfg.approach_distance_m,
            )

    if state.area_ratio >= cfg.near_area_ratio:
        return PlannerIntent("reached", "target is centered and close", state.target_class)

    return PlannerIntent(
        "navigate_forward",
        "target is centered but still far",
        state.target_class,
        forward_m=cfg.approach_distance_m,
    )


def preferred_scan_angular_z(
    last_horizontal_offset: float | None,
    default_angular_z: float = 0.35,
    side_threshold: float = 0.25,
) -> float:
    """Choose the next scan direction from the target's last visible side.

    Positive angular velocity rotates left in the robot base frame.  If the
    target was last seen on the left, scan left first; if it was last seen on
    the right, scan right first.  This follows the same sign convention as
    the visible-target turn actions.  A centered or unknown target keeps the
    configured default direction.
    """
    magnitude = abs(float(default_angular_z))
    if last_horizontal_offset is not None:
        if last_horizontal_offset < -abs(side_threshold):
            return magnitude
        if last_horizontal_offset > abs(side_threshold):
            return -magnitude
    return float(default_angular_z)


def is_strong_visual_reach(
    state: SceneState,
    max_depth_m: float = 1.5,
    min_area_ratio: float = 0.12,
) -> bool:
    """Return whether one RGB-D observation is strong enough to finish a task.

    A depth-backed, centered target with a large image footprint is a much
    stronger signal than a monocular estimate.  Accepting this signal avoids
    losing a genuinely reached target when a small object flickers out of the
    next CPU-limited detector frame.
    """
    return (
        state.visible
        and state.suggested_action == "reached"
        and state.distance_source in {"depth", "rgbd"}
        and state.estimated_distance_m is not None
        and state.estimated_distance_m <= max_depth_m
        and state.area_ratio >= min_area_ratio
        and abs(state.horizontal_offset) <= 0.25
    )


def parse_agent_command(text: str) -> AgentCommand:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if normalized in {"stop", "halt", "cancel"}:
        return AgentCommand(tool="stop_robot")
    if normalized in {"status", "state"}:
        return AgentCommand(tool="get_status")

    match = re.match(r"^(find|search|track|go to|goto|navigate to)\s+(.+)$", normalized)
    if match:
        target = match.group(2).strip()
        return AgentCommand(tool="search_target", target_class=target)

    return AgentCommand(tool="unknown")
