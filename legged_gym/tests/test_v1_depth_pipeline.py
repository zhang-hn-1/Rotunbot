import unittest

import torch


class V1DepthPipelineTests(unittest.TestCase):
    def test_forward_camera_rotation_maps_optical_minus_z_to_robot_plus_x(self):
        from legged_gym.scripts.audit_v1_depth_pipeline import camera_forward_rotation

        self.assertEqual(camera_forward_rotation(), (0.0, -0.70710678, 0.0, 0.70710678))

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


if __name__ == "__main__":
    unittest.main()
