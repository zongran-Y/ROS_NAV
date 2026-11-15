from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SimulationLaunchRegressionTest(unittest.TestCase):
    def test_spawn_nodes_wait_for_slow_gazebo_factory_startup(self):
        source = (PROJECT_ROOT / "launch" / "vision_agent_gazebo.launch.py").read_text()
        self.assertGreaterEqual(source.count('"-timeout", "600"'), 2)

    def test_bringup_can_disable_cpu_perception_for_gpu_deployment(self):
        bringup = (PROJECT_ROOT / "launch" / "vision_agent_bringup.launch.py").read_text()
        pipeline = (PROJECT_ROOT / "launch" / "vision_agent_nav.launch.py").read_text()

        self.assertIn('DeclareLaunchArgument("enable_perception", default_value="true")', bringup)
        self.assertIn('"enable_perception": enable_perception', bringup)
        self.assertNotIn("condition=IfCondition(enable_perception)", bringup)
        self.assertIn('DeclareLaunchArgument("enable_perception", default_value="true")', pipeline)
        self.assertEqual(pipeline.count("condition=IfCondition(enable_perception)"), 1)


if __name__ == "__main__":
    unittest.main()
