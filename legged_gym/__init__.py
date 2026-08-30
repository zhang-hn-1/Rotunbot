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

import os

# Isaac Gym Preview 4 and the pinned rsl_rl release target the Python 3.8 /
# NumPy APIs that are removed or lazily exposed by newer environments.  Keep
# the compatibility shim at package import time so all task entry points use
# the same runtime behavior.
import distutils
import distutils.version
import numpy as np

if not hasattr(np, "float"):
    np.float = float

# Isaac Gym's pinned TorchScript random helper asks NVRTC to compile for the
# host's SM 8.9 capability, which Torch 1.10/cu113 does not support.  Keep
# this tiny helper eager so RTX 40-series task construction does not JIT a
# random-number kernel with an invalid architecture flag.
import isaacgym.torch_utils as _isaacgym_torch_utils
import torch as _torch


def _eager_torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * _torch.rand(*shape, device=device) + lower


def _eager_quat_rotate_inverse(quaternion, vector):
    q_w = quaternion[:, -1]
    q_vec = quaternion[:, :3]
    a = vector * (2.0 * q_w.square() - 1.0).unsqueeze(-1)
    b = _torch.cross(q_vec, vector, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * _torch.bmm(
        q_vec.view(quaternion.shape[0], 1, 3),
        vector.view(quaternion.shape[0], 3, 1),
    ).squeeze(-1) * 2.0
    return a - b + c


def _eager_quat_apply(quaternion, vector):
    xyz = quaternion[:, :3]
    t = _torch.cross(xyz, vector, dim=-1) * 2.0
    return vector + quaternion[:, 3:] * t + _torch.cross(xyz, t, dim=-1)


def _eager_normalize(value, eps=1.0e-9):
    return value / value.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


_isaacgym_torch_utils.torch_rand_float = _eager_torch_rand_float
_isaacgym_torch_utils.quat_rotate_inverse = _eager_quat_rotate_inverse
_isaacgym_torch_utils.quat_apply = _eager_quat_apply
_isaacgym_torch_utils.normalize = _eager_normalize

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'envs')
