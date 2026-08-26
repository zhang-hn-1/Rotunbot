"""Torch-only utilities for the Robot-frame Local P2P observation."""

import torch


def _check_batched_tensor(name, value, width):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must have shape [N, {width}], got {tuple(value.shape)}")


def world_to_robot_xy(world_delta: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate planar world displacement into the robot yaw frame."""
    _check_batched_tensor("world_delta", world_delta, 2)
    if not isinstance(yaw, torch.Tensor):
        yaw = torch.as_tensor(yaw, dtype=world_delta.dtype, device=world_delta.device)
    else:
        yaw = yaw.to(device=world_delta.device, dtype=world_delta.dtype)
    if yaw.ndim not in (0, 1) or (yaw.ndim == 1 and yaw.shape[0] != world_delta.shape[0]):
        raise ValueError(
            f"yaw must be scalar or shape [{world_delta.shape[0]}], got {tuple(yaw.shape)}"
        )

    c = torch.cos(yaw).reshape(-1, 1) if yaw.ndim == 1 else torch.cos(yaw)
    s = torch.sin(yaw).reshape(-1, 1) if yaw.ndim == 1 else torch.sin(yaw)
    x = world_delta[:, 0:1]
    y = world_delta[:, 1:2]
    return torch.cat((c * x + s * y, -s * x + c * y), dim=1)


def build_local_observation(
    local_goal: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    previous_actions: torch.Tensor,
    max_goal_distance: float,
) -> torch.Tensor:
    """Build the exact 17-D single-frame observation contract."""
    _check_batched_tensor("local_goal", local_goal, 2)
    _check_batched_tensor("base_lin_vel", base_lin_vel, 3)
    _check_batched_tensor("base_ang_vel", base_ang_vel, 3)
    _check_batched_tensor("projected_gravity", projected_gravity, 3)
    _check_batched_tensor("dof_pos", dof_pos, 2)
    _check_batched_tensor("dof_vel", dof_vel, 2)
    _check_batched_tensor("previous_actions", previous_actions, 2)
    batch_size = local_goal.shape[0]
    values = (base_lin_vel, base_ang_vel, projected_gravity, dof_pos, dof_vel, previous_actions)
    if any(value.shape[0] != batch_size for value in values):
        raise ValueError("all observation fields must have the same batch size")
    if max_goal_distance <= 0:
        raise ValueError("max_goal_distance must be positive")
    scale = torch.as_tensor(max_goal_distance, dtype=local_goal.dtype, device=local_goal.device)
    return torch.cat(
        (
            local_goal / scale,
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            dof_pos,
            dof_vel,
            previous_actions,
        ),
        dim=1,
    )
