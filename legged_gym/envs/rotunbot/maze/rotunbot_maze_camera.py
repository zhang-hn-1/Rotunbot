"""Reusable depth sensing for maze environments.

The mixin owns sensor lifecycle and ray/AABB math only.  The environment that
uses it owns the obstacle geometry through ``_get_depth_fallback_aabbs``.
"""

import math

import torch
import torch.nn.functional as F


def normalize_depth_image(depth, near, far):
    """Convert Isaac Gym depth values to finite ``[0, 1]`` far-is-open values."""
    # Isaac Gym IMAGE_DEPTH uses non-positive/no-return values for invalid
    # samples on some paths; treat zero like negative/NaN/Inf rather than
    # clamping it to the near plane and presenting a false close obstacle.
    invalid = (~torch.isfinite(depth)) | (depth <= 0.0)
    depth = torch.where(invalid, torch.full_like(depth, float(far)), depth)
    depth = depth.clamp(float(near), float(far))
    return ((depth - float(near)) / max(float(far) - float(near), 1.0e-6)).clamp(0.0, 1.0)


class DepthCameraMixin:
    """Camera/fallback implementation shared by depth-aware maze tasks."""

    def _init_depth_camera_state(self):
        camera_cfg = self.cfg.camera
        requested = getattr(camera_cfg, "depth_backend", None)
        if requested is None:
            requested = getattr(camera_cfg, "policy_source", "fallback")
            requested = "isaacgym" if requested == "camera" else requested
        requested = str(requested).lower()
        if requested not in ("fallback", "isaacgym"):
            raise ValueError("camera.depth_backend must be 'fallback' or 'isaacgym'")
        self.depth_backend_requested = requested
        self.depth_backend_actual = "unavailable"
        self._camera_handles = []
        self._camera_depth_tensors = []
        self._camera_ready = False
        self._camera_error_reported = False

    def _get_depth_fallback_aabbs(self):
        """Return local XY obstacle centers and XY half-extents.

        Geometry is deliberately supplied by the environment so this sensor
        layer cannot accidentally encode a particular maze layout.
        """
        raise NotImplementedError

    def _create_camera_sensors(self):
        """Create attached IMAGE_DEPTH sensors when the real backend is requested."""
        if self.depth_backend_requested != "isaacgym":
            self.depth_backend_actual = "fallback"
            return
        if not bool(getattr(self.cfg.camera, "enable", True)):
            raise RuntimeError("Isaac Gym depth backend requested but camera.enable is false")
        if getattr(self, "graphics_device_id", -1) < 0:
            raise RuntimeError(
                "Isaac Gym depth backend requested with graphics disabled; "
                "use an enabled graphics device for IMAGE_DEPTH evaluation"
            )
        from isaacgym import gymapi

        camera_cfg = self.cfg.camera
        try:
            props = gymapi.CameraProperties()
            props.width = int(camera_cfg.width)
            props.height = int(camera_cfg.height)
            props.horizontal_fov = float(camera_cfg.horizontal_fov)
            props.near_plane = float(camera_cfg.near_plane)
            props.far_plane = float(camera_cfg.far_plane)
            props.enable_tensors = True
            transform = gymapi.Transform()
            transform.p = gymapi.Vec3(*[float(v) for v in camera_cfg.position])
            transform.r = gymapi.Quat(*[float(v) for v in camera_cfg.rotation])
            self._camera_handles = []
            for env_handle, actor_handle in zip(self.envs, self.actor_handles):
                handle = self.gym.create_camera_sensor(env_handle, props)
                body = self.gym.find_actor_rigid_body_handle(
                    env_handle, actor_handle, getattr(camera_cfg, "body_name", "base_link")
                )
                self.gym.attach_camera_to_body(
                    handle, env_handle, body, transform, gymapi.FOLLOW_TRANSFORM
                )
                self._camera_handles.append(handle)
        except Exception as exc:  # pragma: no cover - simulator dependent
            self._camera_handles = []
            raise RuntimeError("failed to create Isaac Gym IMAGE_DEPTH sensors") from exc

    def _init_camera_tensors(self):
        if self.depth_backend_requested != "isaacgym":
            self.depth_backend_actual = "fallback"
            return
        from isaacgym import gymapi, gymtorch

        try:
            self._camera_depth_tensors = []
            for env_handle, camera_handle in zip(self.envs, self._camera_handles):
                raw = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_handle, camera_handle, gymapi.IMAGE_DEPTH
                )
                wrapped = gymtorch.wrap_tensor(raw)
                self._camera_depth_tensors.append(
                    wrapped.view(int(self.cfg.camera.height), int(self.cfg.camera.width))
                )
            self._camera_ready = len(self._camera_depth_tensors) == self.num_envs
        except Exception as exc:  # pragma: no cover - simulator dependent
            self._camera_depth_tensors = []
            self._camera_ready = False
            raise RuntimeError("failed to bind Isaac Gym IMAGE_DEPTH tensors") from exc

    def _apply_depth_noise(self, depth):
        cfg = self.cfg.camera
        depth = depth.clamp(0.0, 1.0)
        if not bool(getattr(cfg, "add_noise", False)):
            return depth
        depth = depth + torch.randn_like(depth) * float(getattr(cfg, "noise_std", 0.0))
        dropout = torch.rand_like(depth) < float(getattr(cfg, "dropout_probability", 0.0))
        depth = torch.where(dropout, torch.ones_like(depth), depth)
        quantization = float(getattr(cfg, "quantization", 0.0))
        if quantization > 0.0:
            depth = torch.round(depth / quantization) * quantization
        return depth.clamp(0.0, 1.0)

    def _resize_depth_to_observation(self, depth):
        target = (int(self.cfg.env.depth_height), int(self.cfg.env.depth_width))
        if tuple(depth.shape[-2:]) == target:
            return depth
        return F.interpolate(depth.unsqueeze(1), size=target, mode="bilinear", align_corners=False).squeeze(1)

    def depth_debug_stats(self, depth=None):
        """Return non-mutating statistics for the tensor sent to the encoder."""
        if depth is None:
            depth = self.depth_observation
        flat = depth.detach().float().reshape(-1)
        return {
            "shape": list(depth.shape),
            "dtype": str(depth.dtype),
            "min": float(flat.min().item()) if flat.numel() else None,
            "max": float(flat.max().item()) if flat.numel() else None,
            "mean": float(flat.mean().item()) if flat.numel() else None,
            "std": float(flat.std(unbiased=False).item()) if flat.numel() else None,
            "finite": bool(torch.isfinite(depth).all().item()),
            "backend_requested": str(getattr(self, "depth_backend_requested", "unknown")),
            "backend_actual": str(getattr(self, "depth_backend_actual", "unknown")),
            "near_plane": float(self.cfg.camera.near_plane),
            "far_plane": float(self.cfg.camera.far_plane),
        }

    def _fallback_depth(self):
        """Cast horizontal rays against environment-provided XY AABBs."""
        camera_cfg = self.cfg.camera
        height = int(self.cfg.env.depth_height)
        width = int(self.cfg.env.depth_width)
        yaw = self.base_euler_tensor[:, 2]
        offset_x, offset_y = [float(v) for v in getattr(camera_cfg, "position", (0.0, 0.0, 0.0))[:2]]
        camera_x = self.root_states[:, 0] + torch.cos(yaw) * offset_x - torch.sin(yaw) * offset_y
        camera_y = self.root_states[:, 1] + torch.sin(yaw) * offset_x + torch.cos(yaw) * offset_y
        camera_x = camera_x - self.env_origins[:, 0]
        camera_y = camera_y - self.env_origins[:, 1]

        half_fov = math.radians(float(camera_cfg.horizontal_fov)) / 2.0
        relative_angles = torch.linspace(-half_fov, half_fov, width, device=self.device)
        angles = yaw.unsqueeze(1) + relative_angles.unsqueeze(0)
        ray_x = torch.cos(angles)
        ray_y = torch.sin(angles)
        eps = torch.full_like(ray_x, 1.0e-6)
        ray_x = torch.where(ray_x.abs() < 1.0e-6, torch.where(ray_x < 0, -eps, eps), ray_x)
        ray_y = torch.where(ray_y.abs() < 1.0e-6, torch.where(ray_y < 0, -eps, eps), ray_y)
        nearest = torch.full((self.num_envs, width), float(camera_cfg.far_plane), device=self.device)
        centers, half_extents = self._get_depth_fallback_aabbs()
        centers = torch.as_tensor(centers, dtype=torch.float32, device=self.device)
        half_extents = torch.as_tensor(half_extents, dtype=torch.float32, device=self.device)
        if centers.numel() == 0:
            return ((nearest - float(camera_cfg.near_plane)) /
                    max(float(camera_cfg.far_plane) - float(camera_cfg.near_plane), 1.0e-6)).unsqueeze(1).expand(-1, height, -1).clone()
        for center, half_extent in zip(centers, half_extents):
            min_x, max_x = center[0] - half_extent[0], center[0] + half_extent[0]
            min_y, max_y = center[1] - half_extent[1], center[1] + half_extent[1]
            tx_a = (min_x - camera_x.unsqueeze(1)) / ray_x
            tx_b = (max_x - camera_x.unsqueeze(1)) / ray_x
            ty_a = (min_y - camera_y.unsqueeze(1)) / ray_y
            ty_b = (max_y - camera_y.unsqueeze(1)) / ray_y
            t_near = torch.maximum(torch.minimum(tx_a, tx_b), torch.minimum(ty_a, ty_b))
            t_far = torch.minimum(torch.maximum(tx_a, tx_b), torch.maximum(ty_a, ty_b))
            hit = (t_far >= torch.maximum(t_near, torch.zeros_like(t_near))) & (t_far > 0.0)
            nearest = torch.minimum(nearest, torch.where(hit, torch.clamp(t_near, min=0.0), nearest))
        normalized = normalize_depth_image(nearest, camera_cfg.near_plane, camera_cfg.far_plane)
        return normalized.unsqueeze(1).expand(-1, height, -1).clone()

    def capture_depth(self):
        """Capture the selected backend and return ``[N, 8, 32]`` normalized depth."""
        if self.depth_backend_requested == "fallback":
            depth = self._fallback_depth()
            self.depth_backend_actual = "fallback"
            return self._apply_depth_noise(self._resize_depth_to_observation(depth))
        if not self._camera_ready:
            raise RuntimeError("IMAGE_DEPTH backend requested but camera tensors are unavailable")
        from isaacgym import gymapi

        try:  # pragma: no cover - simulator dependent
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            raw = torch.stack(self._camera_depth_tensors, dim=0).to(self.device)
            depth = normalize_depth_image(raw, self.cfg.camera.near_plane, self.cfg.camera.far_plane)
            self.depth_backend_actual = "isaacgym"
            return self._apply_depth_noise(self._resize_depth_to_observation(depth))
        except Exception as exc:  # pragma: no cover - simulator dependent
            raise RuntimeError("Isaac Gym IMAGE_DEPTH capture failed") from exc
        finally:  # pragma: no cover - simulator dependent
            self.gym.end_access_image_tensors(self.sim)
