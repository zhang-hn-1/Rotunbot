"""Depth camera sensing for the Rotunbot maze SRU task.

Ported from the reference `SphericalRobot_LeggedGym-master-sru` project
(rotunbot_target_depth.py): a forward depth camera attached to the robot's
base_link plus a deterministic headless ray/AABB fallback so training stays
possible without a graphics device.

The observation encoding is normalized distance: 0 = near plane, 1 = far
plane / open space.  When the real Isaac Gym camera is unavailable (headless
training) the fallback computes a pinhole-like horizontal ray sweep against
the maze wall AABBs, which is fast, deterministic and matches the maze
geometry exactly.
"""

import math

import torch
import torch.nn.functional as F
from isaacgym import gymapi, gymtorch


class DepthCameraMixin:
    """Add forward depth sensing to a Rotunbot maze environment.

    The host env must provide: gym, sim, envs, actor_handles,
    graphics_device_id, device, num_envs, root_states, env_origins,
    base_euler_tensor, and (for the fallback) obstacle_centers /
    obstacle_sizes tensors (built from the maze wall grid).
    """

    def _init_camera(self):
        """Allocate depth buffers and create camera sensors when possible."""
        camera_cfg = self.cfg.camera
        self._camera_handles = []
        self._camera_depth_tensors = []
        self._camera_ready = False
        self._camera_error_reported = False
        self.depth_observation = torch.zeros(
            self.num_envs,
            int(self.cfg.env.depth_height),
            int(self.cfg.env.depth_width),
            dtype=torch.float32,
            device=self.device,
        )
        self.depth_camera_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if not bool(camera_cfg.enable):
            return
        if self.graphics_device_id < 0:
            return
        self._create_camera_sensors()
        self._init_camera_tensors()

    def _create_camera_sensors(self):
        camera_cfg = self.cfg.camera
        try:
            camera_props = gymapi.CameraProperties()
            camera_props.width = int(camera_cfg.width)
            camera_props.height = int(camera_cfg.height)
            camera_props.horizontal_fov = float(camera_cfg.horizontal_fov)
            camera_props.near_plane = float(camera_cfg.near_plane)
            camera_props.far_plane = float(camera_cfg.far_plane)
            camera_props.enable_tensors = True

            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(*[float(v) for v in camera_cfg.position])
            local_transform.r = gymapi.Quat(*[float(v) for v in camera_cfg.rotation])

            base_body_name = "base_link"
            for env_handle, actor_handle in zip(self.envs, self.actor_handles):
                camera_handle = self.gym.create_camera_sensor(
                    env_handle, camera_props
                )
                body_handle = self.gym.find_actor_rigid_body_handle(
                    env_handle, actor_handle, base_body_name
                )
                self.gym.attach_camera_to_body(
                    camera_handle,
                    env_handle,
                    body_handle,
                    local_transform,
                    gymapi.FOLLOW_TRANSFORM,
                )
                self._camera_handles.append(camera_handle)
        except Exception as exc:  # pragma: no cover - Isaac Gym build dependent
            self._camera_handles = []
            if not self._camera_error_reported:
                print(
                    "[DepthCamera] camera tensors unavailable; "
                    f"using headless depth fallback ({exc})"
                )
                self._camera_error_reported = True

    def _init_camera_tensors(self):
        if not self._camera_handles:
            return
        try:
            self._camera_depth_tensors = []
            for env_handle, camera_handle in zip(self.envs, self._camera_handles):
                raw_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_handle, camera_handle, gymapi.IMAGE_DEPTH
                )
                wrapped = gymtorch.wrap_tensor(raw_tensor)
                self._camera_depth_tensors.append(
                    wrapped.view(int(self.cfg.camera.height), int(self.cfg.camera.width))
                )
            self._camera_ready = len(self._camera_depth_tensors) == self.num_envs
        except Exception as exc:  # pragma: no cover - Isaac Gym build dependent
            self._camera_depth_tensors = []
            self._camera_ready = False
            if not self._camera_error_reported:
                print(
                    "[DepthCamera] failed to access camera tensors; "
                    f"using depth fallback ({exc})"
                )
                self._camera_error_reported = True

    # -------------------------------------------------------------- sensing
    def _apply_depth_noise(self, depth):
        camera_cfg = self.cfg.camera
        if not bool(camera_cfg.add_noise):
            return depth.clamp(0.0, 1.0)
        depth = depth + torch.randn_like(depth) * float(camera_cfg.noise_std)
        dropout = torch.rand_like(depth) < float(camera_cfg.dropout_probability)
        # Normalized distance: 1 = far/open space, so missing pixels fill
        # with the far-plane value.
        depth = torch.where(dropout, torch.ones_like(depth), depth)
        quantization = float(camera_cfg.quantization)
        if quantization > 0.0:
            depth = torch.round(depth / quantization) * quantization
        return depth.clamp(0.0, 1.0)

    def _resize_depth_to_observation(self, depth):
        target_height = int(self.cfg.env.depth_height)
        target_width = int(self.cfg.env.depth_width)
        if depth.shape[-2:] == (target_height, target_width):
            return depth
        return F.interpolate(
            depth.unsqueeze(1),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    def _fallback_depth(self):
        """Vectorized pinhole-like horizontal ray/AABB depth observation.

        Casts ``depth_width`` rays across the horizontal FOV in the robot
        body frame and intersects them with the maze wall AABBs.  The image
        is a normalized-distance scanline expanded to [N, H, W].
        """
        height = int(self.cfg.env.depth_height)
        width = int(self.cfg.env.depth_width)
        camera_cfg = self.cfg.camera
        yaw = self.base_euler_tensor[:, 2]
        camera_offset_x = float(camera_cfg.position[0])
        camera_offset_y = float(camera_cfg.position[1])
        camera_x = self.root_states[:, 0] + (
            torch.cos(yaw) * camera_offset_x - torch.sin(yaw) * camera_offset_y
        )
        camera_y = self.root_states[:, 1] + (
            torch.sin(yaw) * camera_offset_x + torch.cos(yaw) * camera_offset_y
        )
        camera_x = camera_x - self.env_origins[:, 0]
        camera_y = camera_y - self.env_origins[:, 1]

        half_fov = math.radians(float(camera_cfg.horizontal_fov)) / 2.0
        ray_angles = torch.linspace(
            -half_fov, half_fov, width, device=self.device
        ).unsqueeze(0)
        ray_angles = ray_angles + yaw.unsqueeze(1)
        ray_x = torch.cos(ray_angles)
        ray_y = torch.sin(ray_angles)
        eps = torch.full_like(ray_x, 1.0e-6)
        ray_x = torch.where(torch.abs(ray_x) < 1.0e-6, eps, ray_x)
        ray_y = torch.where(torch.abs(ray_y) < 1.0e-6, eps, ray_y)

        nearest = torch.full(
            (self.num_envs, width),
            float(camera_cfg.far_plane),
            device=self.device,
        )
        # obstacle_centers: [N, num_walls, 2], obstacle_sizes: [num_walls, 2]
        for obstacle_id in range(self.obstacle_sizes.shape[0]):
            size = self.obstacle_sizes[obstacle_id]
            center = self.obstacle_centers[:, obstacle_id, :]
            min_corner = center - 0.5 * size
            max_corner = center + 0.5 * size
            tx_a = (min_corner[:, 0].unsqueeze(1) - camera_x.unsqueeze(1)) / ray_x
            tx_b = (max_corner[:, 0].unsqueeze(1) - camera_x.unsqueeze(1)) / ray_x
            ty_a = (min_corner[:, 1].unsqueeze(1) - camera_y.unsqueeze(1)) / ray_y
            ty_b = (max_corner[:, 1].unsqueeze(1) - camera_y.unsqueeze(1)) / ray_y
            t_near = torch.maximum(
                torch.minimum(tx_a, tx_b), torch.minimum(ty_a, ty_b)
            )
            t_far = torch.minimum(
                torch.maximum(tx_a, tx_b), torch.maximum(ty_a, ty_b)
            )
            hit = (t_far >= torch.maximum(t_near, torch.zeros_like(t_near))) & (
                t_far > 0.0
            )
            distance = torch.where(hit, torch.clamp(t_near, min=0.0), nearest)
            nearest = torch.minimum(nearest, distance)

        near = float(camera_cfg.near_plane)
        far = float(camera_cfg.far_plane)
        normalized_distance = (nearest - near) / max(far - near, 1.0e-6)
        return normalized_distance.unsqueeze(1).expand(-1, height, -1).clone()

    def capture_depth(self):
        """Refresh self.depth_observation (real camera or fallback)."""
        camera_cfg = self.cfg.camera
        if not bool(camera_cfg.enable):
            # No camera configured: leave depth at open-space (1.0) values.
            self.depth_observation[:] = 1.0
            return
        if not self._camera_ready:
            fallback = self._apply_depth_noise(
                self._resize_depth_to_observation(self._fallback_depth())
            )
            self.depth_observation[:] = fallback
            return

        access_started = False
        try:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            access_started = True
            raw_depth = torch.stack(self._camera_depth_tensors, dim=0).to(
                self.device
            )
            raw_depth = torch.where(raw_depth < 0.0, -raw_depth, raw_depth)
            raw_depth = torch.where(
                torch.isfinite(raw_depth),
                raw_depth,
                torch.full_like(raw_depth, float(camera_cfg.far_plane)),
            )
            near = float(camera_cfg.near_plane)
            far = float(camera_cfg.far_plane)
            depth = (raw_depth.clamp(near, far) - near) / max(far - near, 1.0e-6)
            camera_depth = self._resize_depth_to_observation(depth)
            self.depth_observation[:] = self._apply_depth_noise(camera_depth)
        except Exception as exc:  # pragma: no cover - Isaac Gym build dependent
            self._camera_ready = False
            if not self._camera_error_reported:
                print(
                    "[DepthCamera] camera render failed; "
                    f"using depth fallback ({exc})"
                )
                self._camera_error_reported = True
            fallback = self._apply_depth_noise(
                self._resize_depth_to_observation(self._fallback_depth())
            )
            self.depth_observation[:] = fallback
        finally:
            if access_started:
                self.gym.end_access_image_tensors(self.sim)
