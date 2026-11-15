import json
import time
from collections import OrderedDict

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from sensor_msgs.msg import Image
from std_msgs.msg import String

class LatencyProbe(Node):
    def __init__(self, target=20):
        super().__init__('ros_pipeline_latency_probe')
        self.target = target
        self.images = OrderedDict()
        self.samples_ms = []
        self.done = Future()
        image_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                               reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, image_qos)
        self.create_subscription(String, '/perception/detections', self.on_detection, 10)

    def on_image(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.images[stamp] = time.perf_counter()
        while len(self.images) > 200:
            self.images.popitem(last=False)

    def on_detection(self, msg):
        payload = json.loads(msg.data)
        stamp = float(payload.get('stamp', 0.0))
        if not self.images or stamp <= 0:
            return
        matched = min(self.images, key=lambda value: abs(value - stamp))
        if abs(matched - stamp) > 0.02:
            return
        self.samples_ms.append((time.perf_counter() - self.images[matched]) * 1000)
        if len(self.samples_ms) >= self.target and not self.done.done():
            self.done.set_result(True)

rclpy.init()
probe = LatencyProbe()
rclpy.spin_until_future_complete(probe, probe.done, timeout_sec=45)
if len(probe.samples_ms) < probe.target:
    raise SystemExit(f'only collected {len(probe.samples_ms)} matching samples')
print(json.dumps({'n': len(probe.samples_ms),
                  'p50_ms': round(float(np.percentile(probe.samples_ms, 50)), 2),
                  'p95_ms': round(float(np.percentile(probe.samples_ms, 95)), 2),
                  'min_ms': round(min(probe.samples_ms), 2),
                  'max_ms': round(max(probe.samples_ms), 2)}, sort_keys=True))
probe.destroy_node()
rclpy.shutdown()
