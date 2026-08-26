"""State-preserving temporary-goal updates for a frozen P2P environment."""

from dataclasses import dataclass

import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _goal(value):
    result = _array(value)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError("world_goal_xy must contain two finite values")
    return result


@dataclass(frozen=True)
class GoalSwitchEvent:
    switch_index: int
    time_s: float
    world_goal_xy: tuple


class GoalSwitchController:
    """Change only ``env.commands[env_index, :2]`` and measure continuity."""

    def __init__(self, env, env_index=0):
        self.env = env
        self.env_index = int(env_index)
        self.switches = []
        self.current_world_goal = None

    def update_world_goal(self, world_goal_xy, time_s=0.0):
        goal = _goal(world_goal_xy)
        commands = self.env.commands
        if hasattr(commands, "device"):
            import torch

            value = torch.as_tensor(goal, dtype=commands.dtype, device=commands.device)
            commands[self.env_index, :2] = value
        else:
            commands[self.env_index, :2] = goal
        event = GoalSwitchEvent(
            switch_index=len(self.switches),
            time_s=float(time_s),
            world_goal_xy=(float(goal[0]), float(goal[1])),
        )
        self.switches.append(event)
        self.current_world_goal = goal.copy()
        return event

    def measure_action_discontinuity(self, action):
        if not hasattr(self.env, "last_actions"):
            raise AttributeError("environment must expose last_actions")
        previous = _array(self.env.last_actions[self.env_index])[:2]
        current = _array(action)
        if current.shape == (1, 2):
            current = current[0]
        if current.shape != (2,) or not np.all(np.isfinite(current)):
            raise ValueError("action must contain two finite values")
        return float(np.linalg.norm(current - previous))

    def set_intermediate_goal_mode(self, enabled):
        """Delegate waypoint-only reset behavior to the evaluation wrapper."""
        setter = getattr(self.env, "set_intermediate_goal", None)
        if setter is None:
            raise AttributeError(
                "environment must expose set_intermediate_goal for Goal Switch"
            )
        setter(bool(enabled))
