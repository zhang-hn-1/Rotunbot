"""V62 evaluation task with corridor walls created before Isaac Gym preparation."""

import math

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import RotunbotVel
from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel_config import (
    RotunbotVelSRU50SafeYawResidualV62TransitionCfg,
    RotunbotVelSRU50SafeYawResidualV62TransitionCfgPPO,
)
from legged_gym.utils import task_registry


CORRIDOR_TASK_NAME = "rotunbot_vel_sru50_v62_corridor_eval"


def make_wall_segments(centerline, maximum_heading_change=0.15):
    """Coalesce straight edges and facet curved edges without excess actors."""
    points = np.asarray(centerline, dtype=np.float64)
    if len(points) < 2:
        return ()
    segments = []
    segment_start = points[0].copy()
    reference_direction = None
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-8:
            continue
        direction = delta / length
        if reference_direction is not None:
            change = math.atan2(
                abs(float(np.cross(reference_direction, direction))),
                float(np.dot(reference_direction, direction)),
            )
            if change > float(maximum_heading_change):
                if np.linalg.norm(start - segment_start) > 1.0e-8:
                    segments.append((segment_start.copy(), start.copy()))
                segment_start = start.copy()
                reference_direction = direction
        else:
            reference_direction = direction
        if index == len(points) - 2:
            if np.linalg.norm(end - segment_start) > 1.0e-8:
                segments.append((segment_start.copy(), end.copy()))
    return tuple(segments)


class RotunbotVelCorridor(RotunbotVel):
    """Keep robot actor state views valid while adding static wall actors."""

    def _create_envs(self):
        super()._create_envs()
        segments = tuple(getattr(self.cfg, "corridor_wall_segments", ()))
        self._corridor_wall_count = 2 * len(segments)
        self._corridor_actors_per_env = 1 + self._corridor_wall_count
        if not segments:
            return

        options = gymapi.AssetOptions()
        options.fix_base_link = True
        options.disable_gravity = True
        wall_assets = []
        for start, end in segments:
            dx = float(end[0]) - float(start[0])
            dy = float(end[1]) - float(start[1])
            length = math.hypot(dx, dy)
            if length <= 1.0e-6:
                raise ValueError("corridor wall segment must have positive length")
            wall_assets.append(
                (
                    self.gym.create_box(self.sim, length, 0.05, 0.40, options),
                    start,
                    end,
                    length,
                )
            )

        half_width = float(self.cfg.corridor_wall_width_m) / 2.0
        for env_index, env_handle in enumerate(self.envs):
            actor_index = 1
            for segment_index, (asset, start, end, length) in enumerate(wall_assets):
                dx = float(end[0]) - float(start[0])
                dy = float(end[1]) - float(start[1])
                yaw = math.atan2(dy, dx)
                normal = (-math.sin(yaw), math.cos(yaw))
                midpoint = (
                    0.5 * (float(start[0]) + float(end[0])),
                    0.5 * (float(start[1]) + float(end[1])),
                )
                for side in (-1.0, 1.0):
                    pose = gymapi.Transform()
                    pose.p = gymapi.Vec3(
                        midpoint[0] + side * half_width * normal[0],
                        midpoint[1] + side * half_width * normal[1],
                        0.20,
                    )
                    pose.r = gymapi.Quat(
                        0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)
                    )
                    self.gym.create_actor(
                        env_handle,
                        asset,
                        pose,
                        "corridor_wall_%d_%d" % (segment_index, int(side > 0)),
                        actor_index,
                        0,
                        0,
                    )
                    actor_index += 1
            if actor_index != self._corridor_actors_per_env:
                raise RuntimeError("corridor actor count mismatch")

    def _select_robot_root_states(self, actor_root_state):
        if not getattr(self, "_corridor_wall_count", 0):
            return super()._select_robot_root_states(actor_root_state)
        expected = self.num_envs * self._corridor_actors_per_env
        if actor_root_state.shape[0] != expected:
            raise RuntimeError(
                "corridor actor root tensor has unexpected size: %d != %d"
                % (actor_root_state.shape[0], expected)
            )
        return actor_root_state.view(self.num_envs, self._corridor_actors_per_env, 13)[:, 0, :]

    def _robot_actor_ids(self, env_ids):
        return env_ids * self._corridor_actors_per_env

    def _reset_root_states(self, env_ids):
        if not getattr(self, "_corridor_wall_count", 0):
            return super()._reset_root_states(env_ids)
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )
        if not bool(self.cfg.init_state.randomize_initial_velocity):
            self.root_states[env_ids, 7:13] = 0.0
        actor_ids = self._robot_actor_ids(env_ids).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._all_root_states),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )


def register_v62_corridor_eval_task():
    if CORRIDOR_TASK_NAME not in task_registry.task_classes:
        task_registry.register(
            CORRIDOR_TASK_NAME,
            RotunbotVelCorridor,
            RotunbotVelSRU50SafeYawResidualV62TransitionCfg(),
            RotunbotVelSRU50SafeYawResidualV62TransitionCfgPPO(),
        )
    return CORRIDOR_TASK_NAME
