"""Launch TurtleBot3 Waffle Pi with the project's RGB-D Gazebo camera.

The stock Humble Waffle Pi model already contains an RGB camera.  This launch
file creates a temporary copy of that model and adds a Gazebo depth sensor to
the same optical frame, so the system does not need to modify /opt/ros.
"""

from pathlib import Path
import re
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


RGBD_IMAGE_WIDTH = 320
RGBD_IMAGE_HEIGHT = 240
RGBD_UPDATE_RATE = 10


def _replace_sensor_tag(block: str, tag: str, value: int) -> str:
    pattern = rf"(<{tag}>)[^<]+(</{tag}>)"
    updated, count = re.subn(pattern, rf"\g<1>{value}\g<2>", block, count=1)
    if count != 1:
        raise ValueError(f"could not tune <{tag}> in sensor block")
    return updated


def _tune_rgb_camera(model_text: str) -> str:
    marker = '<sensor name="camera" type="camera">'
    start = model_text.index(marker)
    end = model_text.index("</sensor>", start)
    block = model_text[start:end]
    for tag, value in (
        ("update_rate", RGBD_UPDATE_RATE),
        ("width", RGBD_IMAGE_WIDTH),
        ("height", RGBD_IMAGE_HEIGHT),
    ):
        block = _replace_sensor_tag(block, tag, value)
    return model_text[:start] + block + model_text[end:]


def _make_static_model_dir(model_dir: Path) -> Path:
    """Create a repeatable, non-interacting copy of a demo model."""
    source = model_dir / "model.sdf"
    model_text = source.read_text(encoding="utf-8")
    if "<static>true</static>" not in model_text:
        model_start = model_text.index("<model")
        model_end = model_text.index(">", model_start) + 1
        model_text = model_text[:model_end] + "\n    <static>true</static>" + model_text[model_end:]

    cache_dir = Path(tempfile.gettempdir()) / "vision_agent_nav" / f"{model_dir.name}_static"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "model.sdf").write_text(model_text, encoding="utf-8")
    return cache_dir


def _spawn_model(name: str, model_dir: Path, x: str, y: str, z: str, yaw: str = "0.0") -> Node:
    return Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", name,
            "-file", str(model_dir / "model.sdf"),
            "-x", x,
            "-y", y,
            "-z", z,
            "-Y", yaw,
            "-timeout", "600",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("spawn_demo_objects")),
    )


def _make_rgbd_model() -> Path:
    gazebo_share = Path(get_package_share_directory("turtlebot3_gazebo"))
    source = gazebo_share / "models" / "turtlebot3_waffle_pi" / "model.sdf"
    model_text = _tune_rgb_camera(source.read_text(encoding="utf-8"))

    if "vision_agent_depth_camera" not in model_text:
        camera_link = '<link name="camera_rgb_optical_frame">'
        link_start = model_text.index(camera_link)
        link_end = model_text.index("</link>", link_start)
        depth_sensor = f'''

      <!-- RGB-D extension added by vision_agent_nav. -->
      <sensor name="vision_agent_depth_camera" type="depth">
        <always_on>true</always_on>
        <visualize>false</visualize>
        <update_rate>{RGBD_UPDATE_RATE}</update_rate>
        <camera name="depth_camera">
          <horizontal_fov>1.085595</horizontal_fov>
          <image>
            <width>{RGBD_IMAGE_WIDTH}</width>
            <height>{RGBD_IMAGE_HEIGHT}</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.05</near>
            <far>10.0</far>
          </clip>
          <depth_camera>
            <output>depths</output>
          </depth_camera>
        </camera>
        <!-- Humble's gazebo_plugins exposes depth support through this plugin. -->
        <plugin name="vision_agent_depth_camera_driver" filename="libgazebo_ros_camera.so">
          <ros>
            <remapping>depth_camera/image_raw:=/camera/depth/rgb/image_raw</remapping>
            <remapping>depth_camera/camera_info:=/camera/depth/rgb/camera_info</remapping>
            <remapping>depth_camera/depth/image_raw:=/camera/depth/image_raw</remapping>
            <remapping>depth_camera/depth/camera_info:=/camera/depth/camera_info</remapping>
            <remapping>depth_camera/points:=/camera/depth/points</remapping>
          </ros>
          <camera_name>depth_camera</camera_name>
          <frame_name>camera_depth_optical_frame</frame_name>
        </plugin>
      </sensor>
'''
        model_text = model_text[:link_end] + depth_sensor + model_text[link_end:]

    cache_dir = Path(tempfile.gettempdir()) / "vision_agent_nav"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "turtlebot3_waffle_pi_rgbd.sdf"
    target.write_text(model_text, encoding="utf-8")
    return target


def generate_launch_description():
    gazebo_share = Path(get_package_share_directory("gazebo_ros"))
    turtlebot_share = Path(get_package_share_directory("turtlebot3_gazebo"))
    model_path = _make_rgbd_model()
    world_path = turtlebot_share / "worlds" / "turtlebot3_world.world"
    urdf_path = turtlebot_share / "urdf" / "turtlebot3_waffle_pi.urdf"
    robot_description = urdf_path.read_text(encoding="utf-8")
    model_search_path = turtlebot_share / "models"
    demo_model_path = Path.home() / ".gazebo" / "models"
    person_model_dir = _make_static_model_dir(demo_model_path / "person_standing")
    cup_model_dir = _make_static_model_dir(demo_model_path / "plastic_cup")

    use_sim_time = LaunchConfiguration("use_sim_time")
    gui = LaunchConfiguration("gui")
    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    yaw_pose = LaunchConfiguration("yaw_pose")

    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(gazebo_share / "launch" / "gzserver.launch.py")),
        launch_arguments={"world": str(world_path), "verbose": "false"}.items(),
    )
    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(gazebo_share / "launch" / "gzclient.launch.py")),
        condition=IfCondition(gui),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "robot_description": robot_description}],
    )

    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "vision_agent_waffle_pi",
            "-file", str(model_path),
            "-x", x_pose,
            "-y", y_pose,
            "-z", "0.01",
            "-Y", yaw_pose,
            "-timeout", "600",
        ],
        output="screen",
    )

    depth_frame_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_depth_frame_tf",
        arguments=[
            "0", "0", "0", "0", "0", "0",
            "camera_rgb_optical_frame", "camera_depth_optical_frame",
        ],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("x_pose", default_value="-2.0"),
        DeclareLaunchArgument("y_pose", default_value="-0.5"),
        DeclareLaunchArgument("yaw_pose", default_value="1.57079632679"),
        DeclareLaunchArgument("spawn_demo_objects", default_value="true"),
        SetEnvironmentVariable(
            "GAZEBO_MODEL_PATH",
            [
                str(model_search_path),
                ":",
                str(demo_model_path),
                ":",
                EnvironmentVariable("GAZEBO_MODEL_PATH", default_value=""),
            ],
        ),
        gazebo_server,
        gazebo_client,
        robot_state_publisher,
        depth_frame_tf,
        TimerAction(period=5.0, actions=[spawn_robot]),
        TimerAction(period=7.0, actions=[
            # Keep the demo target in the mapped free corridor. The previous
            # pose at (-0.3, -0.5) was inside the dense TurtleBot3 pillar field,
            # leaving Nav2 no safe local trajectory to approach it.
            _spawn_model("demo_person_standing", person_model_dir, "-2.0", "1.2", "0.0", "3.14"),
            _spawn_model("demo_cafe_table", demo_model_path / "cafe_table", "0.9", "0.15", "0.0", "0.0"),
            _spawn_model("demo_plastic_cup", cup_model_dir, "0.9", "0.15", "0.78", "0.0"),
            _spawn_model("demo_coke_can", demo_model_path / "coke_can", "1.15", "-0.1", "0.0", "0.0"),
        ]),
    ])
