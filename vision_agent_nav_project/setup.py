from setuptools import setup

package_name = "vision_agent_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/vision_agent_nav.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/vision_agent_gazebo.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/vision_agent_bringup.launch.py"]),
        ("share/" + package_name + "/config", ["config/params.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Vision Agent Navigation contributors",
    maintainer_email="maintainers@example.org",
    description="ROS2 visual target search and lightweight Agent navigation for TurtleBot3.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "perception_node = vision_agent_nav.perception_node:main",
            "scene_state_node = vision_agent_nav.scene_state_node:main",
            "target_planner_node = vision_agent_nav.target_planner_node:main",
            "agent_node = vision_agent_nav.agent_node:main",
            "nav_client_node = vision_agent_nav.nav_client_node:main",
            "metrics_node = vision_agent_nav.metrics_node:main",
            "initial_pose_node = vision_agent_nav.initial_pose_node:main",
        ],
    },
)
