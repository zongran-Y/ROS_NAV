# GPU 部署与 ROS 2 闭环测试记录

## 测试架构

YOLOv8n 感知节点运行在 GPU 容器中，Gazebo、Nav2 和任务状态机运行在仿真环境中。两侧通过 ROS 2 DDS 交换相机数据和检测结果；测试使用 `ROS_DOMAIN_ID=0` 和 Fast DDS UDPv4 传输。

仿真包含 TurtleBot3 RGB-D 相机、Nav2 地图和 Gazebo Fuel Standing Person 模型。Fuel 模型文件没有随公开代码分发，获取方式见仓库中的模型说明。

## 性能测量

固定同一张实际相机帧（320×240）、YOLOv8n、`imgsz=320`、`batch=1` 和 `conf=0.25`。CPU 使用 1 个 PyTorch 线程；每种设备预热 5 次，再计时 20 次。CUDA 计时包含设备同步。

| 设备 | 单帧推理 P50 | 单帧推理 P95 |
|---|---:|---:|
| CPU | 8.38 ms | 11.42 ms |
| NVIDIA CUDA | 5.36 ms | 6.57 ms |

该组 20 次样本中，CUDA 推理 P50 降低约 36.0%，P95 降低约 42.5%。样本量较小，数据用于描述本次验证，不等同于大样本稳定基准。

使用源 RGB 帧时间戳关联 ROS 图像订阅与检测发布，共测 20 个样本：

| ROS 感知链路 | P50 | P95 |
|---|---:|---:|
| `/camera/image_raw` 到 `/perception/detections` | 7.37 ms | 8.13 ms |

现场检测样本中，YOLO 推理耗时 6.70 ms，`depth_valid=true`，深度龄期 62 ms；目标 person 置信度 0.861，深度估距 1.235 m。

## 导航闭环

输入 `find person` 后，任务状态依次为：

```text
searching → navigate_forward → nav_goal_sent → reached
```

最终状态为 `REACHED`，细节为 `visual target reached`。Nav2 返回成功，任务耗时约 3.9 秒，0 次 recovery。修复检测时间戳后再次运行时，Gazebo 也记录了导航目标成功。

## 已修复问题

- Gazebo factory 服务就绪较慢，机器人与目标的 spawn 等待上限从 30 秒调整为 600 秒。
- 修正 `enable_perception` launch 参数传递，使 GPU 部署可以关闭仿真侧 CPU 感知节点，同时保留任务状态机和 Nav2 客户端。
- 初始定位四元数按配置的 yaw 计算，与机器人生成方向保持一致。
- 检测消息的 `stamp` 改为对应的 RGB 源帧时间戳，便于关联图像、检测和延迟样本。
- 跨容器 Fast DDS 使用 UDPv4，避免依赖共享内存传输相机消息。

## 回归验证

- `test_simulation_launch_regressions.py`：3 项通过，覆盖 spawn 等待、关闭 CPU 感知的 launch 参数传递和源帧时间戳。
- `test_initial_pose.py`：ROS 环境下 2 项通过，覆盖默认初始位姿和自定义 yaw。
- ROS 链路延迟可使用 `scripts/measure_ros_pipeline_latency.py` 复测。
