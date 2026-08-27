import unittest

import torch

from legged_gym.scripts.evaluate_depth_local import aggregate_records, side_obstacle_observability


class DepthLocalEvaluationTests(unittest.TestCase):
    def records(self):
        return [
            {"local_success": 1, "global_success": 1, "collision": 0, "timeout": 0,
             "waypoint_reach_count": 2, "final_distance": 0.1, "path_length": 3.0,
             "completion_time": 4.0, "depth_backend_requested": "fallback", "depth_backend_actual": "fallback"},
            {"local_success": 1, "global_success": 0, "collision": 1, "timeout": 0,
             "waypoint_reach_count": 1, "final_distance": 1.0, "path_length": 2.0,
             "completion_time": 5.0, "depth_backend_requested": "fallback", "depth_backend_actual": "fallback"},
        ]

    def test_aggregation(self):
        result = aggregate_records(self.records())
        self.assertEqual(result["local_success_rate"], 1.0)
        self.assertEqual(result["global_success_rate"], 0.5)
        self.assertEqual(result["collision_rate"], 0.5)
        self.assertEqual(result["waypoint_reach_count"], 3)
        self.assertEqual(result["depth_backend_actual"], ["fallback"])

    def test_formal_camera_rejects_fallback(self):
        with self.assertRaises(ValueError):
            aggregate_records(self.records(), formal_camera=True)

    def test_side_observability_metric(self):
        depth = torch.ones(1, 8, 32)
        depth[:, :, :8] = 0.5
        self.assertEqual(side_obstacle_observability(depth), 1.0)


if __name__ == "__main__":
    unittest.main()
