import unittest

import torch


class V1DepthPipelineTests(unittest.TestCase):
    def test_v1_camera_pose_uses_isaac_identity_forward_mount(self):
        from legged_gym.scripts.audit_v1_depth_pipeline import camera_forward_rotation

        self.assertEqual(camera_forward_rotation(), (0.0, 0.0, 0.0, 1.0))

    def test_pipeline_summary_reports_invalid_raw_values_and_finite_encoder_input(self):
        from legged_gym.scripts.audit_v1_depth_pipeline import summarize_depth_pipeline

        raw = torch.tensor([[-1.0, 0.0, 0.05, 4.0, 8.0, float("inf"), float("nan")]])
        normalized = torch.tensor([[1.0, 1.0, 0.0, 0.5, 1.0, 1.0, 1.0]])
        result = summarize_depth_pipeline(
            raw,
            normalized,
            near_plane=0.05,
            far_plane=8.0,
            metadata={"backend": "fallback"},
        )
        self.assertEqual(result["raw_invalid_count"], 4)
        self.assertEqual(result["raw_nonpositive_count"], 2)
        self.assertEqual(result["raw_nonfinite_count"], 2)
        self.assertTrue(result["encoder_input_finite"])
        self.assertEqual(result["encoder_input_range"], [0.0, 1.0])
        self.assertEqual(result["backend"], "fallback")

    def test_physical_sanity_requires_monotonic_center_depth_and_finite_inputs(self):
        from legged_gym.scripts.audit_v1_depth_physical_sanity import (
            validate_physical_sanity,
        )

        result = validate_physical_sanity([
            {"wall_distance_m": 0.5, "center_distance_m": 0.50, "finite_ratio": 1.0},
            {"wall_distance_m": 2.0, "center_distance_m": 2.00, "finite_ratio": 1.0},
            {"wall_distance_m": 5.0, "center_distance_m": 5.00, "finite_ratio": 1.0},
        ])
        self.assertTrue(result["pass"])
        self.assertEqual(result["ordered_wall_distances_m"], [0.5, 2.0, 5.0])

        failed = validate_physical_sanity([
            {"wall_distance_m": 0.5, "center_distance_m": 2.0, "finite_ratio": 1.0},
            {"wall_distance_m": 2.0, "center_distance_m": 1.0, "finite_ratio": 1.0},
            {"wall_distance_m": 5.0, "center_distance_m": 5.0, "finite_ratio": 0.5},
        ])
        self.assertFalse(failed["pass"])
        self.assertIn("center_depth_not_monotonic", failed["failures"])
        self.assertIn("encoder_input_not_finite", failed["failures"])

    def test_camera_capture_access_scope_ends_after_tensor_copy(self):
        from types import SimpleNamespace
        from legged_gym.envs.rotunbot.maze.rotunbot_maze_camera import (
            capture_isaac_depth_tensors,
        )

        calls = []
        gym = SimpleNamespace(
            fetch_results=lambda sim, wait: calls.append("fetch_results"),
            step_graphics=lambda sim: calls.append("step_graphics"),
            render_all_camera_sensors=lambda sim: calls.append("render"),
            start_access_image_tensors=lambda sim: calls.append("start"),
            end_access_image_tensors=lambda sim: calls.append("end"),
        )
        raw, normalized = capture_isaac_depth_tensors(
            gym,
            "sim",
            [torch.tensor([[0.5, 2.0]], dtype=torch.float32)],
            torch.device("cpu"),
            near=0.05,
            far=8.0,
        )
        self.assertEqual(
            calls,
            ["fetch_results", "step_graphics", "render", "start", "end"],
        )
        self.assertEqual(tuple(raw.shape), (1, 1, 2))
        self.assertTrue(torch.isfinite(normalized).all())


if __name__ == "__main__":
    unittest.main()
