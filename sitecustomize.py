"""RTX-40 / numpy / distutils compatibility shims (this machine only).

Applied automatically at interpreter startup so the reference project's
scripts run on the RTX 4070 (sm_89) with torch 1.10/cu113.
"""
import distutils  # noqa: E402
import distutils.version  # noqa: E402
distutils.version = distutils.version

import numpy as np  # noqa: E402
np.float = float

import isaacgym  # noqa: F401,E402 (must precede torch for gymdeps)
import torch  # noqa: E402
import isaacgym.torch_utils as torch_utils  # noqa: E402


def _quat_rotate_inverse(q, v):
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(
        q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)
    ).squeeze(-1) * 2.0
    return a - b + c


def _quat_apply(a, b):
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2.0
    return b + a[:, 3:] * t + xyz.cross(t, dim=-1)


def _normalize(x, eps=1.0e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def _torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


torch_utils.quat_rotate_inverse = _quat_rotate_inverse
torch_utils.quat_apply = _quat_apply
torch_utils.normalize = _normalize
torch_utils.torch_rand_float = _torch_rand_float
