# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import torch.distributed as dist


def _init_data_parallel(args):
    """Initialize optional torchrun data parallelism for one shared policy."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU training requires CUDA.")

    torch.cuda.set_device(local_rank)
    args.compute_device_id = local_rank
    args.sim_device_id = local_rank
    args.sim_device = f"cuda:{local_rank}"
    args.rl_device = f"cuda:{local_rank}"

    # --num_envs denotes the total number of environments across all ranks.
    configured_envs = args.num_envs
    if configured_envs is None:
        configured_envs = task_registry.get_cfgs(args.task)[0].env.num_envs
    if configured_envs % world_size != 0:
        raise ValueError(
            f"--num_envs ({configured_envs}) must be divisible by "
            f"WORLD_SIZE ({world_size})"
        )
    args.num_envs = configured_envs // world_size

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size


def _broadcast_policy(runner):
    if not dist.is_available() or not dist.is_initialized():
        return
    for parameter in runner.alg.actor_critic.parameters():
        dist.broadcast(parameter.data, src=0)

def train(args):
    rank, world_size = _init_data_parallel(args)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if world_size > 1 and env_cfg.seed != -1:
        # Avoid all ranks collecting identical trajectories.
        env_cfg.seed = int(env_cfg.seed) + rank
        train_cfg.seed = env_cfg.seed
    env, env_cfg = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg
    )
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root="default" if rank == 0 else None,
    )
    _broadcast_policy(ppo_runner)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    args = get_args()
    train(args)
