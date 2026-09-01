import tempfile
import unittest
from pathlib import Path

import torch

from legged_gym.navigation.v1_teacher_dataset import (
    REQUIRED_STEP_FIELDS,
    TeacherSequenceWriter,
    load_teacher_dataset,
)


class V1TeacherDatasetTests(unittest.TestCase):
    def _step(self, episode_id, step_id, done=False):
        return {
            "episode_id": episode_id,
            "step_id": step_id,
            "depth": torch.full((8, 32), 0.5),
            "goal_xy_robot": torch.tensor([1.0, 0.0]),
            "proprioception": torch.zeros(12),
            "previous_command": torch.zeros(2),
            "previous_actual_velocity": torch.zeros(2),
            "teacher_command": torch.tensor([0.2, 0.0]),
            "actual_velocity": torch.zeros(2),
            "governor_command": torch.zeros(2),
            "projection_command": torch.tensor([0.2, 0.0]),
            "done": done,
            "success": done,
            "collision": False,
            "goal_distance": torch.tensor(0.3),
        }

    def test_schema_and_ordered_done_bounded_sequences(self):
        writer = TeacherSequenceWriter(sequence_length=2)
        writer.append(self._step(4, 0))
        writer.append(self._step(4, 1, done=True))
        writer.append(self._step(9, 0, done=True))
        dataset = writer.finalize()
        self.assertEqual(dataset["schema_version"], 1)
        self.assertEqual(set(dataset["step_fields"]), set(REQUIRED_STEP_FIELDS))
        self.assertEqual(len(dataset["episodes"]), 2)
        self.assertEqual(dataset["episodes"][0]["episode_id"], 4)
        self.assertEqual(dataset["episodes"][0]["step_id"].tolist(), [0, 1])
        self.assertEqual(dataset["episodes"][1]["episode_id"], 9)
        self.assertTrue(bool(dataset["episodes"][0]["done"][-1]))

    def test_done_boundary_rejects_nonterminal_episode_close(self):
        writer = TeacherSequenceWriter()
        writer.append(self._step(1, 0, done=False))
        with self.assertRaises(ValueError):
            writer.finalize()

    def test_save_load_preserves_sequence_and_metadata(self):
        writer = TeacherSequenceWriter(sequence_length=8)
        writer.append(self._step(3, 0, done=True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            writer.save(path, metadata={"depth_backend": "isaacgym", "seed": 2026})
            loaded = load_teacher_dataset(path)
        self.assertEqual(loaded["metadata"]["depth_backend"], "isaacgym")
        self.assertEqual(loaded["episodes"][0]["teacher_command"].shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
