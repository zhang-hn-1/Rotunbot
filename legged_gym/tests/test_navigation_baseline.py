import tempfile
import unittest
from pathlib import Path

from legged_gym.navigation.baseline import (
    ACTION_DIM,
    CHECKPOINT_RELATIVE_PATH,
    FRAME_STACK,
    OBSERVATION_DIM,
    P2P_TASK_NAME,
    require_checkpoint,
)


class NavigationBaselineTests(unittest.TestCase):
    def test_frozen_policy_contract_is_explicit(self):
        self.assertEqual(P2P_TASK_NAME, "rotunbot_target_repro")
        self.assertEqual(OBSERVATION_DIM, 19)
        self.assertEqual(FRAME_STACK, 20)
        self.assertEqual(ACTION_DIM, 2)
        self.assertEqual(
            CHECKPOINT_RELATIVE_PATH,
            "Aug16_02-57-06_uniform_t1_long500_from3809/model_4150.pt",
        )

    def test_checkpoint_validation_rejects_missing_path(self):
        with self.assertRaises(FileNotFoundError):
            require_checkpoint(Path("/tmp/definitely-missing-uniform-4150.pt"))

    def test_checkpoint_validation_accepts_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_4150.pt"
            path.write_bytes(b"checkpoint")
            self.assertEqual(require_checkpoint(path), path.resolve())


if __name__ == "__main__":
    unittest.main()
