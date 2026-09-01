import tempfile
import unittest
from pathlib import Path

import torch

from legged_gym.navigation.v1_teacher_dataset import (
    REQUIRED_STEP_FIELDS,
    TeacherSequenceWriter,
    load_teacher_dataset,
)
from legged_gym.navigation.v1_velocity_imitation import (
    build_imitation_observations,
    collate_imitation_sequences,
    iter_imitation_sequences,
    masked_huber_loss,
    imitation_loss,
    teacher_command_to_action,
    train_imitation_epoch,
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

    def test_imitation_observation_uses_current_v1_abi_layout(self):
        episode = TeacherSequenceWriter._materialize(
            {"episode_id": 2, "rows": [self._step(2, 0, done=True)]}, 16
        )
        observation = build_imitation_observations(episode)
        self.assertEqual(tuple(observation.shape), (1, 275))
        self.assertTrue(torch.equal(observation[:, :12], episode["proprioception"]))
        self.assertTrue(torch.equal(observation[:, 12:14], episode["goal_xy_robot"] / 8.0))
        self.assertTrue(torch.equal(observation[:, 14:16], episode["previous_command"]))
        self.assertTrue(torch.equal(observation[:, 16:18], episode["previous_actual_velocity"]))
        self.assertTrue(torch.equal(observation[:, 18:274].reshape(1, 8, 32), episode["depth"]))
        self.assertTrue(torch.equal(observation[:, 274], torch.zeros(1)))

    def test_imitation_sequences_are_padded_without_crossing_episode_done(self):
        writer = TeacherSequenceWriter(sequence_length=2)
        for episode_id in (4, 9):
            writer.append(self._step(episode_id, 0))
            writer.append(self._step(episode_id, 1, done=True))
        dataset = writer.finalize()
        sequences = list(iter_imitation_sequences(dataset, sequence_length=2))
        batch = collate_imitation_sequences(sequences, device="cpu")
        self.assertEqual(tuple(batch["observations"].shape), (2, 2, 275))
        self.assertEqual(batch["episode_ids"].tolist(), [4, 9])
        self.assertTrue(torch.equal(batch["valid_mask"], torch.ones(2, 2, dtype=torch.bool)))
        self.assertTrue(torch.equal(batch["done"][-1], torch.ones(2, dtype=torch.bool)))
        self.assertTrue(torch.equal(batch["recurrent_masks"][0], torch.zeros(2)))

    def test_teacher_commands_map_to_bounded_actor_domain_and_mask_loss(self):
        physical = torch.tensor([[0.25, 0.10], [0.0, 0.0]])
        action = teacher_command_to_action(physical, 0.25, 0.10)
        self.assertTrue(torch.allclose(action, torch.tensor([[1.0, 1.0], [0.0, 0.0]])))
        prediction = torch.zeros(2, 2, 2)
        target = torch.ones(2, 2, 2)
        mask = torch.tensor([[True, False], [True, False]])
        self.assertAlmostEqual(float(masked_huber_loss(prediction, target, mask)), 0.5)

    def test_synthetic_recurrent_imitation_converges(self):
        class TinySRUPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                from legged_gym.dwl.actor_critic_depth import SpatialRecurrentUnit

                self.encoder = torch.nn.Linear(275, 8)
                self.memory = SpatialRecurrentUnit(8, 8)
                self.head = torch.nn.Linear(8, 2)

            def _mean(self, observations, hidden_states=None, masks=None, update_state=False):
                steps, batch, _ = observations.shape
                encoded = self.encoder(observations.reshape(steps * batch, -1))
                encoded = encoded.reshape(steps, batch, -1).transpose(0, 1)
                recurrent, _ = self.memory(
                    encoded,
                    hidden=hidden_states,
                    masks=masks.transpose(0, 1),
                    return_sequence=True,
                )
                return torch.tanh(self.head(recurrent)).transpose(0, 1)

        torch.manual_seed(7)
        writer = TeacherSequenceWriter(sequence_length=4)
        for step in range(4):
            row = self._step(1, step, done=step == 3)
            row["teacher_command"] = torch.tensor([0.20, 0.05])
            writer.append(row)
        dataset = writer.finalize()
        policy = TinySRUPolicy()
        sequences = list(iter_imitation_sequences(dataset, sequence_length=4))
        batch = collate_imitation_sequences(sequences, hidden_dim=8)
        initial = float(imitation_loss(policy, batch).detach())
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)
        for epoch in range(25):
            train_imitation_epoch(policy, dataset, optimizer, batch_size=1, seed=epoch)
        final = float(imitation_loss(policy, batch).detach())
        self.assertLess(final, initial * 0.25)


if __name__ == "__main__":
    unittest.main()
