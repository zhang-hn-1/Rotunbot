"""Runtime compatibility boundary for the legacy Isaac Gym evaluation tools."""


def install_python_compat():
    """Install only compatibility aliases needed before importing rsl_rl."""
    import distutils
    import distutils.version
    import numpy as np

    # Python 3.8's setuptools distutils shim does not expose the imported
    # submodule as an attribute, while torch.utils.tensorboard expects it.
    distutils.version = distutils.version
    # Isaac Gym Preview 4 still references this removed NumPy alias.
    np.float = float


def install_isaac_gym_compat():
    """Install legacy Python/Isaac Gym shims without changing task behavior."""
    install_python_compat()
    import isaacgym  # noqa: F401
    import torch
    import isaacgym.torch_utils as torch_utils

    def quat_rotate_inverse(q, v):
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(
            q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)
        ).squeeze(-1) * 2.0
        return a - b + c

    def quat_apply(a, b):
        xyz = a[:, :3]
        t = xyz.cross(b, dim=-1) * 2.0
        return b + a[:, 3:] * t + xyz.cross(t, dim=-1)

    def normalize(x, eps=1.0e-9):
        return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

    def torch_rand_float(lower, upper, shape, device):
        return (upper - lower) * torch.rand(*shape, device=device) + lower

    # PyTorch 1.10/cu113 cannot JIT these helpers reliably on Ada GPUs.  The
    # eager equivalents preserve their math and leave the frozen task intact.
    torch_utils.quat_rotate_inverse = quat_rotate_inverse
    torch_utils.quat_apply = quat_apply
    torch_utils.normalize = normalize
    torch_utils.torch_rand_float = torch_rand_float
