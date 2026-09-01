"""Sequence preparation and loss functions for V1 recurrent imitation."""

import random

import torch
import torch.nn.functional as F

from .direct_velocity_observation import build_direct_velocity_observation


V1_ACTOR_OBSERVATION_DIM = 275


def build_imitation_observations(episode, max_goal_distance=8.0):
    """Build the current V1 actor ABI from one stored teacher episode."""
    count = int(episode["depth"].shape[0])
    recovery = torch.zeros(count, dtype=torch.bool)
    observation = build_direct_velocity_observation(
        episode["proprioception"].float(),
        episode["goal_xy_robot"].float(),
        episode["previous_command"].float(),
        episode["depth"].float(),
        max_goal_distance=max_goal_distance,
        recovery_active=recovery,
        previous_actual_velocity=episode["previous_actual_velocity"].float(),
    )
    if observation.shape[1] != V1_ACTOR_OBSERVATION_DIM:
        raise ValueError(
            "unexpected V1 imitation observation width: %d"
            % observation.shape[1]
        )
    if not torch.isfinite(observation).all():
        raise ValueError("non-finite V1 imitation observation")
    return observation


def teacher_command_to_action(command, max_forward_speed, max_yaw_rate):
    """Convert physical teacher commands into the actor's bounded action domain."""
    if command.ndim != 2 or command.shape[1] != 2:
        raise ValueError("teacher command must have shape [N, 2]")
    if float(max_forward_speed) <= 0.0 or float(max_yaw_rate) <= 0.0:
        raise ValueError("command limits must be positive")
    if not torch.isfinite(command).all():
        raise ValueError("teacher command contains non-finite values")
    action = command.clone().float()
    action[:, 0] /= float(max_forward_speed)
    action[:, 1] /= float(max_yaw_rate)
    if torch.any(action.abs() > 1.0 + 1.0e-5):
        raise ValueError("teacher command exceeds the actor command domain")
    return action.clamp(-1.0, 1.0)


def iter_imitation_sequences(
    dataset,
    sequence_length=None,
    max_goal_distance=8.0,
    max_forward_speed=0.25,
    max_yaw_rate=0.10,
):
    """Yield ordered, episode-bounded truncated-BPTT sequences."""
    length = int(sequence_length or dataset["sequence_length"])
    if length <= 0:
        raise ValueError("sequence_length must be positive")
    for episode in dataset["episodes"]:
        steps = episode["step_id"].tolist()
        if steps != list(range(len(steps))):
            raise ValueError("teacher episode step ids are not chronological")
        if not bool(episode["done"][-1]):
            raise ValueError("teacher episode does not end at done=True")
        observations = build_imitation_observations(
            episode, max_goal_distance=max_goal_distance
        )
        targets = teacher_command_to_action(
            episode["teacher_command"].float(),
            max_forward_speed,
            max_yaw_rate,
        )
        for start in range(0, observations.shape[0], length):
            stop = min(start + length, observations.shape[0])
            yield {
                "episode_id": int(episode["episode_id"]),
                "start_step": int(start),
                "observations": observations[start:stop],
                "targets": targets[start:stop],
                "done": episode["done"].bool()[start:stop],
            }


def collate_imitation_sequences(sequences, hidden_dim=128, device=None):
    """Pad only within a batch and return explicit recurrent masks."""
    sequences = list(sequences)
    if not sequences:
        raise ValueError("cannot collate an empty sequence batch")
    max_steps = max(item["observations"].shape[0] for item in sequences)
    batch_size = len(sequences)
    observation_dim = sequences[0]["observations"].shape[1]
    observations = torch.zeros(max_steps, batch_size, observation_dim)
    targets = torch.zeros(max_steps, batch_size, 2)
    valid_mask = torch.zeros(max_steps, batch_size, dtype=torch.bool)
    done = torch.zeros(max_steps, batch_size, dtype=torch.bool)
    episode_ids = []
    starts = []
    for batch_index, item in enumerate(sequences):
        steps = int(item["observations"].shape[0])
        observations[:steps, batch_index] = item["observations"]
        targets[:steps, batch_index] = item["targets"]
        valid_mask[:steps, batch_index] = True
        done[:steps, batch_index] = item["done"]
        episode_ids.append(int(item["episode_id"]))
        starts.append(int(item["start_step"]))
    recurrent_masks = valid_mask.float()
    recurrent_masks[0] = 0.0
    batch = {
        "observations": observations,
        "targets": targets,
        "valid_mask": valid_mask,
        "recurrent_masks": recurrent_masks,
        "done": done,
        "episode_ids": torch.as_tensor(episode_ids, dtype=torch.long),
        "start_steps": torch.as_tensor(starts, dtype=torch.long),
        "initial_hidden": torch.zeros(batch_size, int(hidden_dim)),
    }
    if device is not None:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
    return batch


def masked_huber_loss(prediction, target, valid_mask, delta=1.0):
    """Compute elementwise Smooth-L1 loss over valid macro steps only."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have shape [T, B, 2]")
    if valid_mask.shape != prediction.shape[:2]:
        raise ValueError("valid mask must have shape [T, B]")
    if not torch.any(valid_mask):
        raise ValueError("masked Huber loss has no valid steps")
    elementwise = F.smooth_l1_loss(
        prediction, target, reduction="none", beta=float(delta)
    )
    mask = valid_mask.to(dtype=elementwise.dtype).unsqueeze(-1)
    return (elementwise * mask).sum() / mask.sum().clamp_min(1.0) / prediction.shape[-1]


def imitation_loss(model, batch):
    """Run the recurrent actor over one padded batch without state leakage."""
    prediction = model._mean(
        batch["observations"],
        hidden_states=batch["initial_hidden"],
        masks=batch["recurrent_masks"],
        update_state=False,
    )
    return masked_huber_loss(prediction, batch["targets"], batch["valid_mask"])


def make_imitation_batches(dataset, batch_size=32, sequence_length=None, seed=2026):
    """Materialize reproducibly shuffled sequence batches for one epoch."""
    sequences = list(iter_imitation_sequences(dataset, sequence_length=sequence_length))
    random.Random(int(seed)).shuffle(sequences)
    batch_size = max(1, int(batch_size))
    for start in range(0, len(sequences), batch_size):
        yield sequences[start:start + batch_size]


def train_imitation_epoch(model, dataset, optimizer, batch_size=32, seed=2026, device=None):
    """Train one masked recurrent imitation epoch and return its mean loss."""
    model.train()
    total_loss = 0.0
    batches = 0
    for sequences in make_imitation_batches(dataset, batch_size=batch_size, seed=seed):
        batch = collate_imitation_sequences(
            sequences, hidden_dim=model.memory.hidden_dim, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        loss = imitation_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
    if batches == 0:
        raise ValueError("teacher dataset produced no imitation batches")
    return total_loss / batches
