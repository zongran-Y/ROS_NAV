from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from pathlib import Path
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    params = str(Path(get_package_share_directory("vision_agent_nav")) / "config" / "params.yaml")
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_perception = LaunchConfiguration("enable_perception")
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("enable_perception", default_value="true"),
            Node(
                package="vision_agent_nav",
                executable="perception_node",
                name="perception_node",
                parameters=[params, {"use_sim_time": use_sim_time}],
                output="screen",
                condition=IfCondition(enable_perception),
            ),
            Node(
                package="vision_agent_nav",
                executable="scene_state_node",
                name="scene_state_node",
                parameters=[params, {"use_sim_time": use_sim_time}],
                output="screen",
            ),
            Node(
                package="vision_agent_nav",
                executable="target_planner_node",
                name="target_planner_node",
                parameters=[params, {"use_sim_time": use_sim_time}],
                output="screen",
            ),
            Node(
                package="vision_agent_nav",
                executable="agent_node",
                name="agent_node",
                parameters=[params, {"use_sim_time": use_sim_time}],
                output="screen",
            ),
            Node(
                package="vision_agent_nav",
                executable="nav_client_node",
                name="nav_client_node",
                parameters=[{"use_sim_time": use_sim_time}],
                output="screen",
            ),
            Node(
                package="vision_agent_nav",
                executable="metrics_node",
                name="metrics_node",
                parameters=[params, {"use_sim_time": use_sim_time}],
                output="screen",
            ),
        ]
    )
