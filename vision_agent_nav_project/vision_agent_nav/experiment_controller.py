from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class Scenario:
    test_id: str
    target: str
    name: str
    target_model: str = "demo_person_standing"
    target_exists: bool = True
    target_x: float = -0.3
    target_y: float = -0.5
    target_z: float = 0.0
    target_yaw: float = 3.14
    supported: bool = True
    note: str = ""
    reveal_after_s: float | None = None
    expected_outcome: str = "success"
    robot_x: float = -2.0
    robot_y: float = -0.5
    robot_yaw: float = 0.0


def default_scenarios() -> list[Scenario]:
    return [
        Scenario("01", "person", "front_1_5m", target_x=-0.3),
        Scenario("02", "person", "front_2_5m", target_x=0.5),
        Scenario("03", "person", "left_30deg", target_x=-1.2, target_y=-0.2, robot_yaw=-0.2),
        Scenario("04", "person", "right_30deg", target_x=-1.0, target_y=-0.8, robot_yaw=0.2),
        Scenario("05", "person", "near_0_8m", target_x=-1.2),
        Scenario("06", "person", "temporary_occlusion", reveal_after_s=5.0, note="target is hidden for the first 5 seconds"),
        Scenario("07", "person", "target_missing", target_exists=False, expected_outcome="no_target"),
        Scenario(
            "08",
            "cup",
            "tabletop_cup",
            target_model="demo_plastic_cup",
            target_x=0.9,
            target_y=0.15,
            target_z=0.78,
            target_yaw=0.0,
            robot_x=-0.3,
            robot_y=-0.5,
            robot_yaw=0.45,
            note="robot starts near and oriented toward the table so the cup is in the RGB-D camera field of view",
        ),
        Scenario("09", "person", "multi_target", target_x=-0.3),
        Scenario("10", "person", "baseline_repeat", target_x=-0.3),
    ]


def validate_scenario_observation(scenario: Scenario, visible: bool) -> tuple[bool, str]:
    """Validate observations that have an explicit negative expectation."""
    if scenario.expected_outcome == "no_target" and visible:
        return False, "target unexpectedly visible"
    return True, ""


def readiness_failure_reason(checks: dict[str, bool]) -> str:
    missing = sorted(name for name, ready in checks.items() if not ready)
    return "" if not missing else f"missing readiness: {', '.join(missing)}"


def clock_is_advancing(samples: list[float]) -> bool:
    return len(samples) >= 2 and any(current != previous for previous, current in zip(samples, samples[1:]))


@dataclass
class TrialResult:
    test_id: str
    scenario: str
    target: str
    success: bool
    final_state: str
    duration_s: float
    first_detection_time_s: float | None
    perception_fps: float | None
    inference_latency_ms: float | None
    detection_count: int
    target_visible_rate: float
    nav_subgoal_successes: int
    failure_reason: str
    scenario_supported: bool = True
    failure_stage: str = "unknown"
    expected_outcome: str = "success"
    valid_for_performance: bool = True
    sample_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    depth_valid_rate: float | None = None
    nav_goal_attempts: int = 0
    failure_category: str = "none"


FAILURE_CATEGORIES = {"none", "setup", "perception", "search", "navigation", "timeout"}


def classify_failure_category(
    state: str,
    reason: str = "",
    result_context: dict[str, Any] | None = None,
) -> str:
    """Map a trial outcome to one stable, report-friendly failure category."""
    context = result_context or {}
    if bool(context.get("success")) or state in {"reached", "expected_no_target"}:
        return "none"

    failure_stage = str(context.get("failure_stage", ""))
    if failure_stage == "setup" or state in {"setup_failed", "not_ready", "unsupported"}:
        return "setup"
    if failure_stage == "navigation" or state in {"nav_aborted", "nav_failed", "nav_rejected", "no_progress"}:
        return "navigation"
    if failure_stage == "perception" or state == "perception_failed":
        return "perception"

    if state == "timeout":
        scene_samples = context.get("scene_samples", [])
        had_visible_target = any(bool(item.get("visible")) for item in scene_samples if isinstance(item, dict))
        if not had_visible_target:
            had_visible_target = float(context.get("target_visible_rate", 0.0) or 0.0) > 0.0
        return "search" if had_visible_target else "perception"

    lowered_reason = reason.lower()
    if "nav2" in lowered_reason or "progress" in lowered_reason:
        return "navigation"
    return "timeout"


def trial_result_from_dict(data: dict[str, Any]) -> TrialResult:
    """Load current and legacy result dictionaries without requiring new fields."""
    payload = dict(data)
    if "failure_category" not in payload:
        payload["failure_category"] = classify_failure_category(
            str(payload.get("final_state", "")),
            str(payload.get("failure_reason", "")),
            {
                "success": bool(payload.get("success", False)),
                "failure_stage": payload.get("failure_stage", ""),
                "target_visible_rate": payload.get("target_visible_rate", 0.0),
            },
        )
    known_fields = {item.name for item in fields(TrialResult)}
    return TrialResult(**{key: value for key, value in payload.items() if key in known_fields})


class TrialRecorder:
    def __init__(self, test_id: str, scenario: Scenario, output_dir: Path | None = None) -> None:
        self.test_id = test_id
        self.scenario = scenario
        self.output_dir = output_dir
        self.started_at: float | None = None
        self.finished = False
        self.events: list[dict[str, Any]] = []
        self.metric_samples: list[dict[str, Any]] = []
        self.scene_samples: list[dict[str, Any]] = []
        self.nav_subgoal_successes = 0
        self.nav_goal_attempts = 0
        self.warnings: list[str] = []
        self._last_status: tuple[str, str, float] | None = None

    def record_event(self, state: str, timestamp_s: float | None = None, detail: str = "") -> None:
        timestamp = time.monotonic() if timestamp_s is None else timestamp_s
        if self.started_at is None and state == "started":
            self.started_at = timestamp
        event = {"timestamp_s": timestamp, "state": state, "detail": detail}
        self.events.append(event)
        self._append_jsonl("events.jsonl", event)

    def record_status(self, state: str, detail: str = "", timestamp_s: float | None = None) -> None:
        timestamp = time.monotonic() if timestamp_s is None else timestamp_s
        if self._last_status is not None:
            previous_state, previous_detail, previous_timestamp = self._last_status
            if state == previous_state and detail == previous_detail and timestamp - previous_timestamp <= 0.1:
                return
        if state == "nav_goal_sent":
            self.nav_goal_attempts += 1
        if state in {"nav_succeeded", "nav_subgoal_succeeded"}:
            self.nav_subgoal_successes += 1
        self._last_status = (state, detail, timestamp)
        self.record_event(state, timestamp, detail)

    def record_metrics(self, metrics: dict[str, Any], timestamp_s: float | None = None) -> None:
        sample = {"timestamp_s": time.monotonic() if timestamp_s is None else timestamp_s, **metrics}
        self.metric_samples.append(sample)
        self._append_jsonl("metrics.jsonl", sample)

    def record_scene(self, scene: dict[str, Any], timestamp_s: float | None = None) -> None:
        sample = {"timestamp_s": time.monotonic() if timestamp_s is None else timestamp_s, **scene}
        self.scene_samples.append(sample)
        self._append_jsonl("scene.jsonl", sample)

    def finish_success(self, timestamp_s: float | None = None) -> TrialResult:
        return self._finish("reached", True, "", timestamp_s, "task", self.scenario.expected_outcome)

    def finish_timeout(self, timestamp_s: float | None = None, reason: str = "timeout") -> TrialResult:
        return self._finish("timeout", False, reason, timestamp_s, "execution", self.scenario.expected_outcome)

    def finish_expected_no_target(self, timestamp_s: float | None = None) -> TrialResult:
        """Record a successful negative test without mixing it into nav performance."""
        return self._finish(
            "expected_no_target",
            True,
            "target absent as expected",
            timestamp_s,
            "perception",
            "no_target",
        )

    def finish_failure(
        self,
        state: str,
        reason: str,
        timestamp_s: float | None = None,
        *,
        failure_stage: str | None = None,
        expected_outcome: str | None = None,
        warning: str | None = None,
    ) -> TrialResult:
        if warning:
            self.warnings.append(warning)
        stage = failure_stage or self._failure_stage_for(state)
        expected = expected_outcome or self.scenario.expected_outcome
        return self._finish(state, False, reason, timestamp_s, stage, expected)

    def _finish(
        self,
        state: str,
        success: bool,
        reason: str,
        timestamp_s: float | None,
        failure_stage: str,
        expected_outcome: str,
    ) -> TrialResult:
        if self.finished:
            raise RuntimeError("trial already finished")
        finish_time = time.monotonic() if timestamp_s is None else timestamp_s
        if self.started_at is None:
            self.started_at = finish_time
        self.finished = True
        self.record_event(state, finish_time, reason)
        valid_for_performance = (
            failure_stage != "setup"
            and self.scenario.supported
            and expected_outcome == "success"
        )
        failure_category = classify_failure_category(
            state,
            reason,
            {
                "success": success,
                "failure_stage": failure_stage,
                "scene_samples": self.scene_samples,
                "target_visible_rate": self._visible_rate(),
            },
        )
        result = TrialResult(
            test_id=self.test_id,
            scenario=self.scenario.name,
            target=self.scenario.target,
            success=success,
            final_state=state,
            duration_s=round(finish_time - self.started_at, 3),
            first_detection_time_s=self._first_detection_time(),
            perception_fps=self._metric_average("perception_fps"),
            inference_latency_ms=self._metric_average("inference_latency_ms"),
            detection_count=int(self._metric_max("detection_count", 0)),
            target_visible_rate=self._visible_rate(),
            nav_subgoal_successes=self.nav_subgoal_successes,
            failure_reason=reason,
            scenario_supported=self.scenario.supported,
            failure_stage=failure_stage,
            expected_outcome=expected_outcome,
            valid_for_performance=valid_for_performance,
            sample_counts={
                "events": len(self.events),
                "metrics": len(self.metric_samples),
                "scene": len(self.scene_samples),
            },
            warnings=list(self.warnings),
            depth_valid_rate=self._metric_average("depth_valid_rate"),
            nav_goal_attempts=self.nav_goal_attempts,
            failure_category=failure_category,
        )
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "result.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    @staticmethod
    def _failure_stage_for(state: str) -> str:
        if state in {"setup_failed", "not_ready", "unsupported"}:
            return "setup"
        if state in {"nav_aborted", "nav_failed", "nav_rejected"}:
            return "navigation"
        if state in {"expected_no_target", "perception_failed"}:
            return "perception"
        if state == "no_progress":
            return "navigation"
        return "execution"

    def _first_detection_time(self) -> float | None:
        for sample in self.scene_samples:
            if sample.get("visible") and self.started_at is not None:
                return round(float(sample["timestamp_s"]) - self.started_at, 3)
        return None

    def _visible_rate(self) -> float:
        if not self.scene_samples:
            return 0.0
        return round(sum(bool(item.get("visible")) for item in self.scene_samples) / len(self.scene_samples), 3)

    def _latest_metric(self, key: str, default: Any) -> Any:
        for sample in reversed(self.metric_samples):
            if key in sample and sample[key] is not None:
                return sample[key]
        return default

    def _metric_average(self, key: str) -> float | None:
        values = [float(item[key]) for item in self.metric_samples if isinstance(item.get(key), (int, float))]
        return round(mean(values), 3) if values else None

    def _metric_max(self, key: str, default: Any) -> Any:
        values = [item[key] for item in self.metric_samples if isinstance(item.get(key), (int, float))]
        return max(values) if values else default

    def _append_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def aggregate_trial_results(results: list[TrialResult]) -> dict[str, Any]:
    successful = [item for item in results if item.success]
    performance = [item for item in results if item.valid_for_performance]
    stage_counts: dict[str, int] = {}
    for item in results:
        stage_counts[item.failure_stage] = stage_counts.get(item.failure_stage, 0) + 1
    category_counts: dict[str, int] = {}
    for item in results:
        category_counts[item.failure_category] = category_counts.get(item.failure_category, 0) + 1
    return {
        "total_tests": len(results),
        "successful_tests": len(successful),
        "failed_tests": len(results) - len(successful),
        "task_success_rate": len(successful) / len(results) if results else 0.0,
        "performance_trial_count": len(performance),
        "setup_failed_tests": sum(item.failure_stage == "setup" for item in results),
        "failure_stage_counts": stage_counts,
        "failure_category_counts": category_counts,
        "positive_target_success_rate": len([item for item in successful if item.valid_for_performance]) / len(performance)
        if performance
        else 0.0,
        "average_task_time_s": mean(item.duration_s for item in performance) if performance else 0.0,
        "successful_task_time_s": mean(item.duration_s for item in successful) if successful else 0.0,
        "average_first_detection_time_s": _mean_available(item.first_detection_time_s for item in performance),
        "average_perception_fps": _mean_available(item.perception_fps for item in performance),
        "average_inference_latency_ms": _mean_available(item.inference_latency_ms for item in performance),
        "average_target_visible_rate": mean(item.target_visible_rate for item in performance) if performance else 0.0,
        "average_depth_valid_rate": _mean_available(item.depth_valid_rate for item in performance),
        "total_nav_subgoals": sum(item.nav_subgoal_successes for item in results),
        "total_nav_goal_attempts": sum(item.nav_goal_attempts for item in results),
        "nav_goal_abort_rate": (
            sum(item.failure_stage == "navigation" and not item.success for item in results)
            / sum(item.nav_goal_attempts for item in results)
            if sum(item.nav_goal_attempts for item in results)
            else 0.0
        ),
    }


def _mean_available(values) -> float | None:
    available = [float(value) for value in values if isinstance(value, (int, float))]
    return round(mean(available), 3) if available else None


def write_trial_reports(output_dir: Path, results: list[TrialResult]) -> None:
    summary = aggregate_trial_results(results)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(asdict(results[0]).keys()) if results else list(TrialResult.__annotations__.keys())
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(item) for item in results)
    lines = ["# Navigation Experiment Report", "", "## Summary", ""]
    lines.extend(f"- `{key}`: {value:.3f}" if isinstance(value, float) else f"- `{key}`: {value}" for key, value in summary.items())
    lines.extend(["", "## Trials", "", "| ID | Scenario | Target | Success | State | Stage | Category | Valid | Time (s) | FPS | Latency (ms) | Detections | Depth valid | Visible rate | Nav goals | Nav subgoals | Failure |", "|---:|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
    for item in results:
        fps = "n/a" if item.perception_fps is None else f"{item.perception_fps:.3f}"
        latency = "n/a" if item.inference_latency_ms is None else f"{item.inference_latency_ms:.3f}"
        depth = "n/a" if item.depth_valid_rate is None else f"{item.depth_valid_rate:.3f}"
        lines.append(f"| {item.test_id} | {item.scenario} | {item.target} | {item.success} | {item.final_state} | {item.failure_stage} | {item.failure_category} | {item.valid_for_performance} | {item.duration_s:.3f} | {fps} | {latency} | {item.detection_count} | {depth} | {item.target_visible_rate:.3f} | {item.nav_goal_attempts} | {item.nav_subgoal_successes} | {item.failure_reason} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
