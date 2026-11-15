# ROS 2 视觉目标搜索与自主导航

一个基于 ROS 2 Humble、Gazebo Classic、TurtleBot3 和 Nav2 的视觉导航项目。机器人接收 `find person` 等目标指令，通过 YOLOv8n 感知、RGB-D 距离估计和任务状态机搜索目标，再调用 Nav2 完成接近与到达确认。

## 功能与结构

```text
Gazebo RGB-D 相机
       ↓
perception_node（YOLOv8n 检测框 + 深度配对）
       ↓
scene_state_node → target_planner_node → agent_node
                                            ↓
                                  nav_client_node → Nav2
                                            ↓
                                      Gazebo 机器人
```

- 感知节点发布目标类别、置信度、图像位置、深度距离和源图像时间戳。
- 任务状态机根据目标可见度安排扫描、对准、分段接近和到达确认。
- GPU 与仿真可以放在不同容器，通过 ROS 2 DDS 通信；使用 Fast DDS 时可设置 `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`。
- `metrics_node` 记录感知和任务相关指标。

## 目录

| 路径 | 内容 |
|---|---|
| `vision_agent_nav/` | ROS 2 节点和共享算法 |
| `launch/` | 仿真、导航和感知 launch 文件 |
| `config/params.yaml` | ROS 话题与算法默认参数，不含主机凭据 |
| `models/gazebo_models/` | 项目仿真道具模型 |
| `models/person_standing_fuel/` | 第三方人物模型的获取说明；模型文件未随仓库分发 |
| `Dockerfile` | YOLOv8n GPU 感知镜像 |
| `Dockerfile.sim-base` | Gazebo、Nav2 和 TurtleBot3 仿真基础镜像 |
| `Dockerfile.simfinal` | 带 Xvfb 启动检查的仿真镜像 |
| `test/` | ROS 运行时单元测试及 launch 回归测试 |
| `scripts/measure_ros_pipeline_latency.py` | ROS 图像到检测发布延迟测量工具 |
| `docs/GPU闭环测试记录.md` | GPU 与导航闭环实测记录 |

## 本地运行

需要 Ubuntu 22.04、ROS 2 Humble、Gazebo Classic、Nav2、TurtleBot3 和 `colcon`。先安装 ROS 依赖，再在已加载 ROS 环境的终端执行：

```bash
source /opt/ros/humble/setup.bash
python3 -m pip install -r requirements.txt
colcon build --symlink-install
source install/setup.bash
ros2 launch vision_agent_nav vision_agent_bringup.launch.py gui:=false
```

默认参数位于 `config/params.yaml`。仿真和感知节点都应使用同一个 `ROS_DOMAIN_ID`；仿真默认使用 `/clock`。Ultralytics 首次运行时会按 `model` 参数获取 YOLOv8n 权重。
另开一个终端发送搜索任务：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic pub --once --wait-matching-subscriptions 1 \
  /agent/command std_msgs/msg/String "{data: 'find person'}"
```

## Docker 构建

从仓库根目录按顺序构建。构建镜像前，先按 `models/person_standing_fuel/README.md` 获取上游人物模型文件，并确认模型授权：

```bash
docker build -f Dockerfile -t vision-agent-nav:gpu .
docker build -f Dockerfile.sim-base -t vision-agent-nav-sim:xvfb .
docker build -f Dockerfile.simfinal -t vision-agent-nav-sim:final .
```

GPU 镜像需要 NVIDIA Container Toolkit。示例启动命令：

```bash
docker run --rm --gpus all --network host \
  -e ROS_DOMAIN_ID=0 \
  -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  vision-agent-nav:gpu
```

启动仿真镜像：

```bash
docker run --rm --network host --shm-size=1g \
  -e ROS_DOMAIN_ID=0 \
  vision-agent-nav-sim:final
```


## 检查与测量

不依赖 ROS 运行时的 launch 回归检查：

```bash
python3 -m unittest discover -s test -p 'test_simulation_launch_regressions.py' -v
```

初始位姿测试需要已安装 ROS 2 消息包并加载 ROS 环境：

```bash
source /opt/ros/humble/setup.bash
python3 test/test_initial_pose.py -v
```

仿真和感知节点运行后，可测量图像进入 ROS 图到检测结果发布的延迟：

```bash
source /opt/ros/humble/setup.bash
python3 scripts/measure_ros_pipeline_latency.py
```

测量端与项目节点需要使用相同 `ROS_DOMAIN_ID`。脚本按源图像时间戳匹配输入帧和检测结果，默认收集 20 个样本。

## GPU 验证摘要

同一张 320×240 相机图像、`imgsz=320`、`batch=1`，每种设备预热 5 次并测量 20 次：

| 设备 | 推理 P50 | 推理 P95 |
|---|---:|---:|
| CPU | 8.38 ms | 11.42 ms |
| NVIDIA CUDA | 5.36 ms | 6.57 ms |

GPU 感知链路的图像到检测发布延迟为 P50 7.37 ms、P95 8.13 ms。`find person` 仿真导航闭环成功，记录用时约 3.9 秒、0 次 Nav2 recovery。每组性能数据为 20 个样本，详见 GPU 闭环测试记录(docs/GPU闭环测试记录.md)。



