import math
import unittest

from builtin_interfaces.msg import Time

from vision_agent_nav.initial_pose_node import make_initial_pose_message


class InitialPoseMessageTest(unittest.TestCase):
    def test_default_pose_remains_at_demo_spawn(self):
        message = make_initial_pose_message(Time())

        self.assertAlmostEqual(message.pose.pose.position.x, -2.0)
        self.assertAlmostEqual(message.pose.pose.position.y, -0.5)
        self.assertAlmostEqual(message.pose.pose.orientation.z, 0.0)
        self.assertAlmostEqual(message.pose.pose.orientation.w, 1.0)

    def test_custom_yaw_matches_spawn_orientation(self):
        yaw = math.pi / 2
        message = make_initial_pose_message(Time(), -2.0, -0.5, yaw)

        self.assertAlmostEqual(message.pose.pose.position.x, -2.0)
        self.assertAlmostEqual(message.pose.pose.position.y, -0.5)
        self.assertAlmostEqual(message.pose.pose.orientation.z, math.sin(yaw / 2))
        self.assertAlmostEqual(message.pose.pose.orientation.w, math.cos(yaw / 2))


if __name__ == "__main__":
    unittest.main()
