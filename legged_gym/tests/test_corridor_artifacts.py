import json
import tempfile
import unittest
from pathlib import Path

import isaacgym  # noqa: F401 - package imports existing Isaac Gym modules

from legged_gym.navigation.corridor_artifacts import (
    CheckpointMetadata,
    EpisodeLogger,
    GateResult,
    replay_episode,
)
from legged_gym.navigation.corridor_plotting import plot_corridor_artifacts


class CorridorArtifactTests(unittest.TestCase):
    def test_gate_result_requires_current_and_regression_pass(self):
        summary = {
            "success_rate": 1.0,
            "collision_rate": 0.0,
            "rate_violation_count": 0,
            "feasible_domain_violation_count": 0,
            "hidden_projection_jump_count": 0,
        }
        result = GateResult.evaluate(
            summary,
            current_rules={"success_rate": (">=", 1.0)},
            regression_rules={"success_rate": (">=", 0.95)},
        )
        self.assertTrue(result["current_pass"])
        self.assertTrue(result["regression_pass"])
        self.assertTrue(result["pass"])

        failed = GateResult.evaluate(
            dict(summary, success_rate=0.8),
            current_rules={"success_rate": (">=", 1.0)},
            regression_rules={"success_rate": (">=", 0.95)},
        )
        self.assertFalse(failed["pass"])
        self.assertTrue(failed["failures"])

    def test_gate_result_supports_strict_bounds(self):
        result = GateResult.evaluate(
            {"max_lateral_error_m": 0.19, "transition_activation_count": 1},
            current_rules={
                "max_lateral_error_m": ("<", 0.20),
                "transition_activation_count": (">", 0),
            },
            regression_rules={},
        )
        self.assertTrue(result["pass"])

    def test_checkpoint_metadata_records_sha_and_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(b"frozen-test-checkpoint")
            metadata = CheckpointMetadata.from_path(
                checkpoint, parent="model_0.pt", stage="A0", seed=17, iterations=0
            )
            self.assertEqual(metadata["stage"], "A0")
            self.assertEqual(metadata["parent_checkpoint"], "model_0.pt")
            self.assertEqual(metadata["sha256"], "ce104fb44ba64215bdfd19c9473a7a41775fee4dfb489c0eebddd5ff68608c7f")

    def test_episode_logger_writes_summary_csv_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logger = EpisodeLogger(root)
            logger.write_episode(
                {
                    "episode_id": 3,
                    "seed": 17,
                    "scenario_family": "straight",
                    "success": False,
                    "timeout": True,
                    "scenario_parameters": {"width_m": 2.0},
                }
            )
            logger.write_trajectory(
                [
                    {"episode_id": 3, "time_s": 0.0, "x": 0.0, "y": 0.0},
                    {"episode_id": 3, "time_s": 0.1, "x": 0.1, "y": 0.0},
                ]
            )
            logger.write_summary({"episodes": 1, "success_rate": 0.0})

            self.assertTrue((root / "episodes.csv").is_file())
            self.assertTrue((root / "summary.json").is_file())
            replay = replay_episode(root, 3)
            self.assertEqual(replay["episode_id"], 3)
            self.assertEqual(replay["seed"], 17)
            self.assertEqual(replay["scenario_parameters"]["width_m"], 2.0)
            self.assertEqual(len(replay["trajectory"]), 2)
            self.assertEqual(json.loads((root / "summary.json").read_text())["episodes"], 1)

    def test_plotter_writes_required_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logger = EpisodeLogger(root)
            logger.write_trajectory(
                [
                    {"episode_id": 1, "time_s": 0.0, "x": 0.0, "y": 0.0, "goal_distance": 1.0, "v_cmd": 0.1, "v_actual": 0.0, "w_cmd": 0.0, "w_actual": 0.0},
                    {"episode_id": 1, "time_s": 0.1, "x": 0.1, "y": 0.0, "goal_distance": 0.9, "v_cmd": 0.1, "v_actual": 0.1, "w_cmd": 0.0, "w_actual": 0.0},
                ]
            )
            outputs = plot_corridor_artifacts(root / "trajectory.csv", root / "plots")
            self.assertEqual(
                {Path(path).name for path in outputs},
                {"xy_trajectory.png", "velocity_tracking.png", "goal_distance.png"},
            )
            self.assertTrue(all(Path(path).is_file() for path in outputs))


if __name__ == "__main__":
    unittest.main()
