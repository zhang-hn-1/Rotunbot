import unittest

import torch

from legged_gym.scripts.evaluate_depth_local import side_obstacle_observability


class DepthObservabilityTests(unittest.TestCase):
    def test_gate_one_point_five_has_a_measurable_side_signal(self):
        depth = torch.ones(2, 8, 32)
        depth[:, :, :8] = 0.4
        self.assertGreaterEqual(side_obstacle_observability(depth), 0.9)


if __name__ == "__main__":
    unittest.main()
