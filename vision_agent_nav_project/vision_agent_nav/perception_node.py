from __future__ import annotations

import json
import math
import time
from collections import deque

import numpy as np
import rclpy
import cv2
import torch
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from vision_agent_nav.core import (
    CameraIntrinsics,
    Detection,
    DistanceEstimator,
    SimpleTracker,
    deproject_pixel_to_3d,
    inference_is_due,
    nearest_timestamp_index,
)


class YoloPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_perception_node")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("depth_image_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("use_depth_distance", True)
        self.declare_parameter("depth_scale", 1.0)
        self.declare_parameter("depth_max_age_s", 0.25)
        self.declare_parameter("depth_max_m", 10.0)
        self.declare_parameter("camera_fx", 554.0)
        self.declare_parameter("camera_fy", 554.0)
        self.declare_parameter("camera_cx", 320.0)
        self.declare_parameter("camera_cy", 240.0)
        self.declare_parameter("camera_frame", "camera_optical_frame")
        self.declare_parameter("model", "yolov8n.pt")
        self.declare_parameter("confidence", 0.45)
        self.declare_parameter("target_classes", ["person", "chair", "bottle"])
        self.declare_parameter("focal_length_px", 554.0)
        self.declare_parameter("default_object_height_m", 1.7)
        self.declare_parameter("inference_fps", 3.0)
        self.declare_parameter("inference_image_size", 640)

        # Keep CPU inference from starving Gazebo and Nav2 on CPU-only hosts.
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)

        self.bridge = CvBridge()
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        depth_image_topic = self.get_parameter("depth_image_topic").get_parameter_value().string_value
        camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.use_depth_distance = self.get_parameter("use_depth_distance").get_parameter_value().bool_value
        self.depth_scale = self.get_parameter("depth_scale").get_parameter_value().double_value
        self.depth_max_age_s = self.get_parameter("depth_max_age_s").get_parameter_value().double_value
        self.depth_max_m = self.get_parameter("depth_max_m").get_parameter_value().double_value
        inference_fps = self.get_parameter("inference_fps").get_parameter_value().double_value
        if inference_fps <= 0:
            raise ValueError("inference_fps must be positive")
        self.inference_period_s = 1.0 / inference_fps
        self.inference_image_size = self.get_parameter("inference_image_size").get_parameter_value().integer_value
        if self.inference_image_size <= 0:
            raise ValueError("inference_image_size must be positive")
        self.camera_intrinsics = CameraIntrinsics(
            fx=self.get_parameter("camera_fx").get_parameter_value().double_value,
            fy=self.get_parameter("camera_fy").get_parameter_value().double_value,
            cx=self.get_parameter("camera_cx").get_parameter_value().double_value,
            cy=self.get_parameter("camera_cy").get_parameter_value().double_value,
            frame_id=self.get_parameter("camera_frame").get_parameter_value().string_value,
        )
        model_name = self.get_parameter("model").get_parameter_value().string_value
        self.confidence = self.get_parameter("confidence").get_parameter_value().double_value
        self.target_classes = set(
            self.get_parameter("target_classes").get_parameter_value().string_array_value
        )
        self.distance_estimator = DistanceEstimator(
            focal_length_px=self.get_parameter("focal_length_px").get_parameter_value().double_value,
            default_object_height_m=self.get_parameter("default_object_height_m").get_parameter_value().double_value,
        )
        self.tracker = SimpleTracker()
        self.latest_depth: np.ndarray | None = None
        self.latest_depth_stamp_s: float | None = None
        self.depth_history: deque[tuple[float, np.ndarray]] = deque(maxlen=12)
        self.last_inference_s: float | None = None

        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.publisher = self.create_publisher(String, "/perception/detections", 10)
        self.subscription = self.create_subscription(
            Image, image_topic, self.on_image, qos_profile_sensor_data
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo, camera_info_topic, self.on_camera_info, qos_profile_sensor_data
        )
        self.depth_subscription = None
        if self.use_depth_distance and depth_image_topic:
            self.depth_subscription = self.create_subscription(
                Image, depth_image_topic, self.on_depth_image, qos_profile_sensor_data
            )
        self.get_logger().info(
            f"YOLO perception node listening on {image_topic}, "
            f"depth={depth_image_topic if self.depth_subscription else 'disabled'}, "
            f"camera_info={camera_info_topic}, model={model_name}"
        )

    def on_camera_info(self, msg: CameraInfo) -> None:
        if len(msg.k) >= 6 and msg.k[0] > 0 and msg.k[4] > 0:
            frame_id = msg.header.frame_id or self.camera_intrinsics.frame_id
            self.camera_intrinsics = CameraIntrinsics(
                fx=float(msg.k[0]),
                fy=float(msg.k[4]),
                cx=float(msg.k[2]),
                cy=float(msg.k[5]),
                frame_id=frame_id,
            )

    def on_depth_image(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().warning(f"Cannot convert depth image: {exc}")
            return
        if depth is None or len(depth.shape) < 2:
            self.get_logger().warning("Received a depth image with an invalid shape")
            return
        self.latest_depth = np.asarray(depth)
        stamp = msg.header.stamp
        self.latest_depth_stamp_s = float(stamp.sec) + float(stamp.nanosec) / 1e9
        self.depth_history.append((self.latest_depth_stamp_s, self.latest_depth))

    def depth_distance_m(
        self,
        bbox: list[float],
        image_width: int,
        image_height: int,
        image_stamp_s: float | None = None,
    ) -> float | None:
        if not self.depth_history:
            return None

        timestamps = [stamp for stamp, _depth in self.depth_history]
        target_stamp = image_stamp_s if image_stamp_s and image_stamp_s > 0 else timestamps[-1]
        depth_index = nearest_timestamp_index(target_stamp, timestamps, self.depth_max_age_s)
        if depth_index is None:
            return None
        _depth_stamp_s, depth = self.depth_history[depth_index]
        depth_height, depth_width = depth.shape[:2]
        if depth_width <= 0 or depth_height <= 0:
            return None

        x1, y1, x2, y2 = bbox
        center_x = ((x1 + x2) * 0.5 / max(float(image_width), 1.0)) * depth_width
        center_y = ((y1 + y2) * 0.5 / max(float(image_height), 1.0)) * depth_height
        box_width = abs(x2 - x1) / max(float(image_width), 1.0) * depth_width
        box_height = abs(y2 - y1) / max(float(image_height), 1.0) * depth_height
        half_window = max(2, int(min(box_width, box_height) * 0.08))
        cx = int(round(center_x))
        cy = int(round(center_y))
        left = max(0, cx - half_window)
        right = min(depth_width, cx + half_window + 1)
        top = max(0, cy - half_window)
        bottom = min(depth_height, cy + half_window + 1)
        if left >= right or top >= bottom:
            return None

        values = np.asarray(depth[top:bottom, left:right], dtype=np.float32).reshape(-1)
        values *= float(self.depth_scale)
        values = values[np.isfinite(values)]
        values = values[(values > 0.05) & (values <= self.depth_max_m)]
        if values.size == 0:
            return None
        return round(float(np.median(values)), 3)

    def on_image(self, msg: Image) -> None:
        now_s = time.perf_counter()
        if not inference_is_due(now_s, self.last_inference_s, self.inference_period_s):
            return
        self.last_inference_s = now_s
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        height, width = frame.shape[:2]
        image_stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
        started = time.perf_counter()
        result = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.inference_image_size,
            verbose=False,
        )[0]
        inference_latency_ms = (time.perf_counter() - started) * 1000.0
        raw_detections: list[Detection] = []
        depth_valid_detection_count = 0

        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = self.model.names[class_id]
            if self.target_classes and class_name not in self.target_classes:
                continue
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            estimated_distance_m = self.distance_estimator.estimate([x1, y1, x2, y2], class_name)
            distance_source = "monocular"
            depth_distance_m = self.depth_distance_m([x1, y1, x2, y2], width, height, image_stamp_s)
            position_3d_m = None
            if depth_distance_m is not None and math.isfinite(depth_distance_m):
                estimated_distance_m = depth_distance_m
                distance_source = "depth"
                depth_valid_detection_count += 1
                center_x = (x1 + x2) * 0.5
                center_y = (y1 + y2) * 0.5
                try:
                    position_3d_m = deproject_pixel_to_3d(
                        center_x,
                        center_y,
                        depth_distance_m,
                        self.camera_intrinsics,
                    )
                except ValueError:
                    position_3d_m = None
            detection = Detection(
                class_name=class_name,
                confidence=float(box.conf.item()),
                bbox=[x1, y1, x2, y2],
                image_width=width,
                image_height=height,
                estimated_distance_m=estimated_distance_m,
                distance_source=distance_source,
                position_3d_m=position_3d_m,
            )
            raw_detections.append(detection)

        tracked_detections = self.tracker.update(raw_detections)

        payload = {
            "stamp": image_stamp_s,
            "inference_latency_ms": inference_latency_ms,
            "depth_valid": depth_valid_detection_count > 0,
            "depth_valid_detection_count": depth_valid_detection_count,
            "depth_age_s": self.depth_age_s(),
            "detections": [item.to_dict() for item in tracked_detections],
        }
        self.publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def depth_age_s(self) -> float | None:
        if self.latest_depth_stamp_s is None:
            return None
        now_s = self.get_clock().now().nanoseconds / 1e9
        return round(abs(now_s - self.latest_depth_stamp_s), 4)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = YoloPerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
