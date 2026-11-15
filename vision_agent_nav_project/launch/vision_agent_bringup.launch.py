"""Start the RGB-D simulation, Nav2, and the vision-agent pipeline together."""

import os
from pathlib import Path
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _tune_nav2_params(source: Path) -> Path:
    """Create a CPU-friendly Nav2 parameter copy for the simulation profile."""
    params_text = source.read_text(encoding="utf-8")
    start = params_text.index("amcl:\n")
    end = params_text.index("\namcl_map_client:", start)
    amcl_block = params_text[start:end]
    for old, new in (
        ("    update_min_a: 0.2", "    update_min_a: 0.0"),
        ("    update_min_d: 0.25", "    update_min_d: 0.0"),
    ):
        if old not in amcl_block:
            raise ValueError(f"could not tune AMCL parameter: {old}")
        amcl_block = amcl_block.replace(old, new, 1)

    for old, new in (
        ("    expected_planner_frequency: 20.0", "    expected_planner_frequency: 3.0"),
        ("      debug_trajectory_details: True", "      debug_trajectory_details: False"),
        ("        publish_voxel_map: True", "        publish_voxel_map: False"),
        ("      always_send_full_costmap: True", "      always_send_full_costmap: False"),
        ("    enable_groot_monitoring: True", "    enable_groot_monitoring: False"),
    ):
        if old not in params_text:
            raise ValueError(f"could not tune Nav2 parameter: {old}")
        params_text = params_text.replace(old, new)

    target_dir = Path(tempfile.gettempdir()) / "vision_agent_nav"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "waffle_pi_nav_params.yaml"
    target.write_text(params_text[:start] + amcl_block + params_text[end:], encoding="utf-8")
    return target


def generate_launch_description():
    package_share = Path(get_package_share_directory("vision_agent_nav"))
    turtlebot_nav_share = Path(get_package_share_directory("turtlebot3_navigation2"))
    nav2_bringup_share = Path(get_package_share_directory("nav2_bringup"))
    gazebo_launch = package_share / "launch" / "vision_agent_gazebo.launch.py"
    nav_launch = nav2_bringup_share / "launch" / "bringup_launch.py"
    rviz_config = nav2_bringup_share / "rviz" / "nav2_default_view.rviz"
    nav_map = turtlebot_nav_share / "map" / "map.yaml"
    nav_params = turtlebot_nav_share / "param" / "humble" / "waffle_pi.yaml"
    tuned_nav_params = _tune_nav2_params(nav_params)
    pipeline_launch = package_share / "launch" / "vision_agent_nav.launch.py"

    # turtlebot3_navigation2 selects its parameter file from this variable.
    # The bundled Gazebo model is Waffle Pi, so keep Nav2's parameter set in sync
    # even when the calling shell has another TURTLEBOT3_MODEL value.
    os.environ["TURTLEBOT3_MODEL"] = "waffle_pi"
    use_sim_time = LaunchConfiguration("use_sim_time")
    auto_initial_pose = LaunchConfiguration("auto_initial_pose")
    enable_perception = LaunchConfiguration("enable_perception")
    gui = LaunchConfiguration("gui")
    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    yaw_pose = LaunchConfiguration("yaw_pose")

    publish_initial_pose = TimerAction(
        period=12.0,
        actions=[
            Node(
                package="vision_agent_nav",
                executable="initial_pose_node",
                name="initial_pose_node",
                parameters=[{
                    "use_sim_time": use_sim_time,
                    "initial_x": x_pose,
                    "initial_y": y_pose,
                    "initial_yaw": yaw_pose,
                }],
                output="screen",
                condition=IfCondition(auto_initial_pose),
            )
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", str(rviz_config)],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
        condition=IfCondition(gui),
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("auto_initial_pose", default_value="true"),
        DeclareLaunchArgument("enable_perception", default_value="true"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("x_pose", default_value="-2.0"),
        DeclareLaunchArgument("y_pose", default_value="-0.5"),
        DeclareLaunchArgument("yaw_pose", default_value="1.57079632679"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(gazebo_launch)),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "gui": gui,
                "x_pose": x_pose,
                "y_pose": y_pose,
                "yaw_pose": yaw_pose,
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(nav_launch)),
            launch_arguments={
                "map": str(nav_map),
                "params_file": str(tuned_nav_params),
                "use_sim_time": use_sim_time,
            }.items(),
        ),
        rviz,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(pipeline_launch)),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "enable_perception": enable_perception,
            }.items(),
        ),
        publish_initial_pose,
    ])
