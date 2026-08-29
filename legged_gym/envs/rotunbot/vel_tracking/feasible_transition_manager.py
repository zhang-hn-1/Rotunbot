"""Batch-safe feasible command transitions for the frozen V62 controller."""

import torch


class TransitionState:
    """Integer states used by :class:`FeasibleVelocityTransitionManager`."""

    TRACK = 0
    BRAKE_TO_ORIGIN = 1
    WAIT_SETTLED = 2
    ACCELERATE_FROM_ORIGIN = 3


class FeasibleVelocityTransitionManager:
    """Keep every applied ``(v, w)`` reference inside V62's feasible domain.

    The manager owns only command-transition state.  Endpoint projection stays
    with the existing V62 ``project_velocity_commands`` function; callers pass
    already-projected targets to :meth:`update_target`.
    """

    def __init__(
        self,
        num_envs,
        device,
        dtype,
        dt,
        maximum_linear_acceleration,
        maximum_yaw_acceleration,
        maximum_forward_speed,
        maximum_yaw_rate,
        minimum_turn_radius,
        envelope_fraction=1.0,
        stationary_threshold=0.0,
        reversal_detection_v=0.05,
        reversal_detection_w=0.015,
        reversal_minimum_request_jump_v=0.10,
        reversal_minimum_request_jump_w=0.03,
        settle_v_threshold=0.01,
        settle_w_threshold=0.005,
        settle_time=0.10,
        curvature_fraction_breakpoints=None,
        curvature_max_speed_values=None,
    ):
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        if float(dt) <= 0.0:
            raise ValueError("dt must be positive")
        if float(minimum_turn_radius) <= 0.0:
            raise ValueError("minimum_turn_radius must be positive")
        if len(curvature_fraction_breakpoints or []) != len(
            curvature_max_speed_values or []
        ):
            raise ValueError("curvature schedule lengths must match")

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.dt = float(dt)
        self.maximum_linear_acceleration = float(maximum_linear_acceleration)
        self.maximum_yaw_acceleration = float(maximum_yaw_acceleration)
        self.maximum_forward_speed = float(maximum_forward_speed)
        self.maximum_yaw_rate = float(maximum_yaw_rate)
        self.minimum_turn_radius = float(minimum_turn_radius)
        self.envelope_fraction = float(envelope_fraction)
        self.stationary_threshold = float(stationary_threshold)
        self.reversal_detection_v = float(reversal_detection_v)
        self.reversal_detection_w = float(reversal_detection_w)
        self.reversal_minimum_request_jump_v = float(reversal_minimum_request_jump_v)
        self.reversal_minimum_request_jump_w = float(reversal_minimum_request_jump_w)
        self.settle_v_threshold = float(settle_v_threshold)
        self.settle_w_threshold = float(settle_w_threshold)
        self.settle_time = float(settle_time)
        self.settle_steps = max(1, int(torch.ceil(torch.tensor(self.settle_time / self.dt)).item()))
        self.curvature_fraction_breakpoints = tuple(
            float(value) for value in (curvature_fraction_breakpoints or [])
        )
        self.curvature_max_speed_values = tuple(
            float(value) for value in (curvature_max_speed_values or [])
        )

        shape = (self.num_envs,)
        command_shape = (self.num_envs, 2)
        self.transition_state = torch.full(
            shape, TransitionState.TRACK, dtype=torch.long, device=self.device
        )
        self.state = self.transition_state
        self.transition_anchor_command = torch.zeros(
            command_shape, dtype=self.dtype, device=self.device
        )
        self.transition_latest_target = torch.zeros_like(
            self.transition_anchor_command
        )
        self.transition_progress = torch.zeros(
            shape, dtype=self.dtype, device=self.device
        )
        self.settle_counter = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.transition_settle_counter = self.settle_counter
        self.transition_active = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids=None):
        """Reset all per-environment transition state to deterministic TRACK."""
        ids = self._normalize_env_ids(env_ids)
        if ids.numel() == 0:
            return
        self.transition_state[ids] = TransitionState.TRACK
        self.transition_anchor_command[ids] = 0.0
        self.transition_latest_target[ids] = 0.0
        self.transition_progress[ids] = 0.0
        self.settle_counter[ids] = 0
        self.transition_active[ids] = False

    def update_target(
        self,
        projected_target,
        current_command,
        measured_forward_velocity,
        measured_yaw_rate,
        env_ids=None,
    ):
        """Store the latest endpoint and latch only genuine linear reversals.

        A target update never restarts an existing brake or settle phase.  This
        is the latest-target-wins rule required by the 5 Hz SRU interface.
        """
        ids = self._normalize_env_ids(env_ids)
        if ids.numel() == 0:
            return
        targets = self._command_batch(projected_target, ids.numel())
        current = self._command_batch(current_command, ids.numel())
        measured_v = self._vector_batch(measured_forward_velocity, ids.numel())
        measured_w = self._vector_batch(measured_yaw_rate, ids.numel())

        old_target = self.transition_latest_target[ids].clone()
        self.transition_latest_target[ids] = targets
        changed = torch.any(torch.abs(targets - old_target) > 1.0e-6, dim=1)

        source_v = torch.where(
            torch.abs(measured_v) > torch.abs(current[:, 0]),
            measured_v,
            current[:, 0],
        )
        request_jump = (
            torch.abs(targets[:, 0] - old_target[:, 0])
            >= self.reversal_minimum_request_jump_v
        ) | (
            torch.abs(targets[:, 1] - old_target[:, 1])
            >= self.reversal_minimum_request_jump_w
        )
        reversal = (
            changed
            & request_jump
            & (torch.abs(source_v) >= self.reversal_detection_v)
            & (torch.abs(targets[:, 0]) >= self.reversal_detection_v)
            & (source_v * targets[:, 0] < 0.0)
        )
        eligible = reversal & (
            (self.transition_state[ids] == TransitionState.TRACK)
            | (
                self.transition_state[ids]
                == TransitionState.ACCELERATE_FROM_ORIGIN
            )
        )
        if torch.any(eligible):
            selected = ids[eligible]
            self.transition_state[selected] = TransitionState.BRAKE_TO_ORIGIN
            self.transition_anchor_command[selected] = current[eligible]
            self.transition_progress[selected] = 1.0
            self.settle_counter[selected] = 0
            self.transition_active[selected] = True

    def advance(self, current_command, measured_forward_velocity, measured_yaw_rate):
        """Advance all environments by one low-level policy period."""
        current = self._command_batch(current_command, self.num_envs)
        measured_v = self._vector_batch(measured_forward_velocity, self.num_envs)
        measured_w = self._vector_batch(measured_yaw_rate, self.num_envs)
        applied = current.clone()

        track_mask = self.transition_state == TransitionState.TRACK
        if torch.any(track_mask):
            applied[track_mask] = self._advance_bounded_feasible(
                current[track_mask], self.transition_latest_target[track_mask]
            )

        brake_mask = self.transition_state == TransitionState.BRAKE_TO_ORIGIN
        entered_wait = torch.zeros_like(brake_mask)
        if torch.any(brake_mask):
            anchor = self.transition_anchor_command[brake_mask]
            fraction_step = self._radial_fraction_step(anchor)
            progress = torch.clamp(
                self.transition_progress[brake_mask] - fraction_step,
                min=0.0,
                max=1.0,
            )
            # Do not snap a residual anchor command to zero early: the final
            # snap itself must also obey the per-step component bounds.
            finished = progress <= 1.0e-7
            progress = torch.where(finished, torch.zeros_like(progress), progress)
            applied[brake_mask] = anchor * progress.unsqueeze(1)
            brake_ids = brake_mask.nonzero(as_tuple=False).flatten()
            self.transition_progress[brake_ids] = progress
            entered_wait[brake_ids[finished]] = True
            if torch.any(finished):
                finished_ids = brake_ids[finished]
                self.transition_state[finished_ids] = TransitionState.WAIT_SETTLED
                self.settle_counter[finished_ids] = 0

        wait_mask = (self.transition_state == TransitionState.WAIT_SETTLED) & ~entered_wait
        entered_accelerate = torch.zeros_like(wait_mask)
        if torch.any(wait_mask):
            applied[wait_mask] = 0.0
            settled = (
                (torch.abs(measured_v[wait_mask]) <= self.settle_v_threshold)
                & (torch.abs(measured_w[wait_mask]) <= self.settle_w_threshold)
            )
            wait_ids = wait_mask.nonzero(as_tuple=False).flatten()
            next_counter = torch.where(
                settled,
                self.settle_counter[wait_ids] + 1,
                torch.zeros_like(self.settle_counter[wait_ids]),
            )
            self.settle_counter[wait_ids] = next_counter
            finished = next_counter >= self.settle_steps
            if torch.any(finished):
                finished_ids = wait_ids[finished]
                self.transition_state[finished_ids] = (
                    TransitionState.ACCELERATE_FROM_ORIGIN
                )
                self.transition_progress[finished_ids] = 0.0
                entered_accelerate[finished_ids] = True

        accelerate_mask = (
            self.transition_state == TransitionState.ACCELERATE_FROM_ORIGIN
        ) & ~entered_accelerate
        if torch.any(accelerate_mask):
            target = self.transition_latest_target[accelerate_mask]
            current_accel = applied[accelerate_mask]
            arrived_target = torch.linalg.norm(target - current_accel, dim=1) <= 2.0e-6
            accelerated = self._advance_bounded_feasible(current_accel, target)
            applied[accelerate_mask] = accelerated
            arrived_target = torch.linalg.norm(target - accelerated, dim=1) <= 2.0e-6
            accel_ids = accelerate_mask.nonzero(as_tuple=False).flatten()
            if torch.any(arrived_target):
                self.transition_state[accel_ids[arrived_target]] = TransitionState.TRACK
                self.transition_progress[accel_ids[arrived_target]] = 0.0

        self.transition_active.copy_(
            self.transition_state != TransitionState.TRACK
        )
        return applied, self.transition_state.clone(), self.transition_active.clone()

    def _advance_bounded_feasible(self, current, target):
        limits = self._rate_limits()
        delta = target - current
        bounded = current + torch.maximum(torch.minimum(delta, limits), -limits)
        valid = self._is_feasible(bounded)
        if torch.all(valid):
            return bounded

        # Backtrack inside the local rate box until the command is feasible.
        # The fixed iteration count is over the scalar search, never over envs.
        low = torch.zeros(current.shape[0], dtype=self.dtype, device=self.device)
        high = torch.ones_like(low)
        for _ in range(16):
            middle = 0.5 * (low + high)
            candidate = current + (bounded - current) * middle.unsqueeze(1)
            candidate_valid = self._is_feasible(candidate)
            low = torch.where(candidate_valid, middle, low)
            high = torch.where(candidate_valid, high, middle)
        safe_fraction = low.unsqueeze(1)
        return current + (bounded - current) * safe_fraction

    def _is_feasible(self, commands):
        speed = torch.abs(commands[:, 0])
        yaw = torch.abs(commands[:, 1])
        tolerance = 3.0e-6
        limit = torch.minimum(
            torch.full_like(speed, self.maximum_yaw_rate),
            speed / self.minimum_turn_radius,
        ) * self.envelope_fraction
        valid = (
            (speed <= self.maximum_forward_speed + tolerance)
            & (yaw <= self.maximum_yaw_rate + tolerance)
            & (yaw <= limit + tolerance)
        )
        if self.stationary_threshold > 0.0:
            stationary = speed < self.stationary_threshold
            valid &= (~stationary) | (yaw <= tolerance)
        if len(self.curvature_fraction_breakpoints) >= 2:
            epsilon = torch.finfo(commands.dtype).eps
            curvature = torch.clamp(
                yaw * self.minimum_turn_radius / torch.clamp(speed, min=epsilon),
                0.0,
                1.0,
            )
            stable_speed = torch.full_like(
                curvature, self.curvature_max_speed_values[0]
            )
            for index in range(len(self.curvature_fraction_breakpoints) - 1):
                left = self.curvature_fraction_breakpoints[index]
                right = self.curvature_fraction_breakpoints[index + 1]
                fraction = torch.clamp(
                    (curvature - left) / (right - left), 0.0, 1.0
                )
                segment = self.curvature_max_speed_values[index] + (
                    self.curvature_max_speed_values[index + 1]
                    - self.curvature_max_speed_values[index]
                ) * fraction
                stable_speed = torch.where(curvature >= left, segment, stable_speed)
            valid &= speed <= stable_speed + tolerance
        return valid

    def _radial_fraction_step(self, anchor):
        limits = self._rate_limits()
        component_steps = limits.unsqueeze(0) / torch.clamp(
            torch.abs(anchor), min=torch.finfo(self.dtype).eps
        )
        zero_component = torch.abs(anchor) <= torch.finfo(self.dtype).eps
        component_steps = torch.where(
            zero_component,
            torch.full_like(component_steps, float("inf")),
            component_steps,
        )
        return torch.clamp(torch.min(component_steps, dim=1).values, max=1.0)

    def _rate_limits(self):
        # Leave one small float margin so the externally observable delta is
        # strictly below the contractual bound after float32 arithmetic.
        margin = 1.0e-6
        return torch.as_tensor(
            [
                max(0.0, self.maximum_linear_acceleration * self.dt - margin),
                max(0.0, self.maximum_yaw_acceleration * self.dt - margin),
            ],
            dtype=self.dtype,
            device=self.device,
        )

    def _normalize_env_ids(self, env_ids):
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        return env_ids.to(device=self.device, dtype=torch.long).reshape(-1)

    def _command_batch(self, value, count):
        tensor = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0).expand(count, -1)
        if tuple(tensor.shape) != (count, 2):
            raise ValueError("command tensors must have shape (count, 2)")
        return tensor

    def _vector_batch(self, value, count):
        tensor = torch.as_tensor(value, dtype=self.dtype, device=self.device).reshape(-1)
        if tensor.numel() == 1 and count != 1:
            tensor = tensor.expand(count)
        if tensor.numel() != count:
            raise ValueError("velocity tensors must have shape (count,)")
        return tensor
