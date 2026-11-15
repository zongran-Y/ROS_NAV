FROM public.ecr.aws/docker/library/ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-opencv ros-humble-cv-bridge \
    ros-humble-sensor-msgs ros-humble-std-msgs \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip 'setuptools>=77,<80' wheel
RUN python3 -m pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128'
RUN python3 -m pip install --no-cache-dir \
      'ultralytics>=8.0.0' 'numpy<2' 'opencv-python>=4.8,<5' packaging

WORKDIR /opt/ros_ws/src/vision_agent_nav
COPY . /opt/ros_ws/src/vision_agent_nav
WORKDIR /opt/ros_ws
RUN bash -lc 'source /opt/ros/humble/setup.bash && colcon build --symlink-install --packages-select vision_agent_nav'

ENTRYPOINT ["bash", "-lc", "source /opt/ros/humble/setup.bash && source /opt/ros_ws/install/setup.bash && exec ros2 run vision_agent_nav perception_node --ros-args --params-file /opt/ros_ws/src/vision_agent_nav/config/params.yaml"]
