import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from legged_gym.navigation.baseline import (
    ACTION_DIM,
    CHECKPOINT_RELATIVE_PATH,
    FRAME_STACK,
    OBSERVATION_DIM,
    P2P_TASK_NAME,
    require_checkpoint,
)
from legged_gym.navigation.frozen_p2p import enforce_frozen_control_config


class NavigationBaselineTests(unittest.TestCase):
    def test_frozen_policy_contract_is_explicit(self):
        self.assertEqual(P2P_TASK_NAME, "rotunbot_target_repro")
        self.assertEqual(OBSERVATION_DIM, 19)
        self.assertEqual(FRAME_STACK, 20)
        self.assertEqual(ACTION_DIM, 2)
        self.assertEqual(
            CHECKPOINT_RELATIVE_PATH,
            "logs/rotunbot_target_repro/Aug16_02-57-06_uniform_t1_long500_from3809/model_4150.pt",
        )

    def test_checkpoint_validation_rejects_missing_path(self):
        with self.assertRaises(FileNotFoundError):
            require_checkpoint(Path("/tmp/definitely-missing-uniform-4150.pt"))

    def test_checkpoint_validation_accepts_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_4150.pt"
            path.write_bytes(b"checkpoint")
            self.assertEqual(require_checkpoint(path), path.resolve())

    def test_frozen_control_config_disables_gain_randomization_and_asserts_gains(self):
        cfg = SimpleNamespace(
            control=SimpleNamespace(
                direct_velocity_gain_randomize=True,
                direct_velocity_gain=100.0,
                direct_position_gain=600.0,
            )
        )
        enforce_frozen_control_config(cfg)
        self.assertFalse(cfg.control.direct_velocity_gain_randomize)

    def test_frozen_control_config_rejects_wrong_trained_gain(self):
        cfg = SimpleNamespace(
            control=SimpleNamespace(
                direct_velocity_gain_randomize=True,
                direct_velocity_gain=35.0,
                direct_position_gain=600.0,
            )
        )
        with self.assertRaises(AssertionError):
            enforce_frozen_control_config(cfg)


if __name__ == "__main__":
    unittest.main()
