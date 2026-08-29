import inspect
import importlib.util
import os
import unittest

import torch


def _load_transition_manager_module():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(
        project_root,
        "legged_gym",
        "envs",
        "rotunbot",
        "vel_tracking",
        "feasible_transition_manager.py",
    )
    spec = importlib.util.spec_from_file_location("feasible_transition_manager", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_manager_module = _load_transition_manager_module()
FeasibleVelocityTransitionManager = _manager_module.FeasibleVelocityTransitionManager
TransitionState = _manager_module.TransitionState


class FeasibleTransitionManagerTests(unittest.TestCase):
    def make_manager(self, count=1):
        return FeasibleVelocityTransitionManager(
            num_envs=count,
            device="cpu",
            dtype=torch.float32,
            dt=0.02,
            maximum_linear_acceleration=0.10,
            maximum_yaw_acceleration=0.007,
            maximum_forward_speed=0.25,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=2.0,
            envelope_fraction=1.0,
            stationary_threshold=0.0,
            reversal_detection_v=0.05,
            reversal_detection_w=0.015,
            reversal_minimum_request_jump_v=0.10,
            reversal_minimum_request_jump_w=0.03,
            settle_v_threshold=0.01,
            settle_w_threshold=0.005,
            settle_time=0.10,
            curvature_fraction_breakpoints=[0.0, 0.25, 0.50, 1.0],
            curvature_max_speed_values=[0.25, 0.20, 0.15, 0.10],
        )

    def feasible(self, commands):
        v = torch.abs(commands[:, 0])
        w = torch.abs(commands[:, 1])
        return bool(
            torch.all(v <= 0.25 + 2.0e-6)
            and torch.all(w <= 0.10 + 2.0e-6)
            and torch.all(w <= v / 2.0 + 2.0e-6)
        )

    def test_forward_same_branch_matches_bounded_v62_step(self):
        manager = self.make_manager()
        current = torch.tensor([[0.10, 0.03]])
        target = torch.tensor([[0.15, 0.04]])
        manager.update_target(target, current, torch.tensor([0.10]), torch.tensor([0.03]))
        applied, state, active = manager.advance(
            current, torch.tensor([0.10]), torch.tensor([0.03])
        )
        self.assertTrue(torch.allclose(applied, torch.tensor([[0.102, 0.03014]]), atol=2e-6))
        self.assertEqual(int(state.item()), TransitionState.TRACK)
        self.assertFalse(bool(active.item()))

    def test_boundary_turn_can_release_yaw_before_accelerating(self):
        manager = self.make_manager()
        current = torch.tensor([[0.10, 0.05]])
        target = torch.tensor([[0.20, 0.0]])
        manager.update_target(target, current, current[:, 0], current[:, 1])
        next_command, _, _ = manager.advance(
            current, current[:, 0], current[:, 1]
        )
        self.assertLess(float(next_command[0, 1]), 0.05)
        self.assertLessEqual(float(torch.abs(next_command[0, 0] - current[0, 0])), 0.002)
        for _ in range(300):
            current, _, _ = manager.advance(current, current[:, 0], current[:, 1])
        self.assertGreater(float(current[0, 0]), 0.15)
        self.assertLess(float(current[0, 1]), 0.02)

    def test_backward_same_branch_does_not_enter_brake(self):
        manager = self.make_manager()
        current = torch.tensor([[-0.10, -0.03]])
        target = torch.tensor([[-0.15, -0.04]])
        manager.update_target(target, current, torch.tensor([-0.10]), torch.tensor([-0.03]))
        _, state, _ = manager.advance(current, torch.tensor([-0.10]), torch.tensor([-0.03]))
        self.assertEqual(int(state.item()), TransitionState.TRACK)

    def test_same_branch_output_remains_feasible(self):
        manager = self.make_manager()
        current = torch.tensor([[0.10, 0.03]])
        target = torch.tensor([[0.15, 0.04]])
        manager.update_target(target, current, torch.tensor([0.10]), torch.tensor([0.03]))
        for _ in range(40):
            current, _, _ = manager.advance(current, current[:, 0], current[:, 1])
            self.assertTrue(self.feasible(current))

    def test_constant_curvature_reversal_passes_through_origin(self):
        manager = self.make_manager()
        current = torch.tensor([[0.14, 0.035]])
        target = torch.tensor([[-0.14, -0.035]])
        manager.update_target(target, current, current[:, 0], current[:, 1])
        seen = [current.clone()]
        for _ in range(500):
            current, state, _ = manager.advance(current, current[:, 0], current[:, 1])
            seen.append(current.clone())
            if int(state.item()) == TransitionState.WAIT_SETTLED:
                break
        trace = torch.cat(seen, dim=0)
        self.assertEqual(int(state.item()), TransitionState.WAIT_SETTLED)
        self.assertLess(float(torch.min(torch.abs(trace[:, 0]))), 1.0e-6)
        self.assertLess(float(torch.min(torch.abs(trace[:, 1]))), 1.0e-6)

    def test_fixed_nonzero_yaw_reversal_brakes_yaw_with_linear_speed(self):
        manager = self.make_manager()
        current = torch.tensor([[0.14, 0.035]])
        target = torch.tensor([[-0.14, 0.035]])
        manager.update_target(target, current, current[:, 0], current[:, 1])
        for _ in range(500):
            current, state, _ = manager.advance(current, current[:, 0], current[:, 1])
            if int(state.item()) == TransitionState.WAIT_SETTLED:
                break
        self.assertEqual(int(state.item()), TransitionState.WAIT_SETTLED)
        self.assertTrue(torch.allclose(current, torch.zeros_like(current), atol=1e-7))

    def test_every_step_respects_linear_and_yaw_rate_bounds(self):
        manager = self.make_manager()
        current = torch.tensor([[0.14, 0.035]])
        manager.update_target(
            torch.tensor([[-0.14, 0.035]]), current, current[:, 0], current[:, 1]
        )
        previous = current.clone()
        for _ in range(500):
            current, state, _ = manager.advance(current, current[:, 0], current[:, 1])
            delta = torch.abs(current - previous)
            self.assertLessEqual(float(delta[0, 0]), 0.002)
            self.assertLessEqual(float(delta[0, 1]), 0.00014)
            previous = current.clone()
            if int(state.item()) == TransitionState.WAIT_SETTLED:
                break

    def test_every_applied_command_is_projection_invariant(self):
        manager = self.make_manager()
        current = torch.tensor([[0.14, 0.035]])
        manager.update_target(
            torch.tensor([[-0.14, 0.035]]), current, current[:, 0], current[:, 1]
        )
        for _ in range(500):
            current, state, _ = manager.advance(current, current[:, 0], current[:, 1])
            self.assertTrue(self.feasible(current))
            if int(state.item()) == TransitionState.WAIT_SETTLED:
                break

    def test_latest_target_wins_without_restarting_brake_progress(self):
        manager = self.make_manager()
        current = torch.tensor([[0.14, 0.035]])
        manager.update_target(
            torch.tensor([[-0.14, -0.035]]), current, current[:, 0], current[:, 1]
        )
        current, state, _ = manager.advance(current, current[:, 0], current[:, 1])
        progress_after_first_step = float(manager.transition_progress.item())
        manager.update_target(
            torch.tensor([[-0.10, 0.025]]), current, current[:, 0], current[:, 1]
        )
        self.assertEqual(int(manager.state.item()), TransitionState.BRAKE_TO_ORIGIN)
        self.assertLess(float(manager.transition_progress.item()), 1.0)
        self.assertAlmostEqual(float(manager.transition_progress.item()), progress_after_first_step, places=6)
        self.assertEqual(int(state.item()), TransitionState.BRAKE_TO_ORIGIN)

    def test_wait_settled_requires_measured_velocity(self):
        manager = self.make_manager()
        manager.state.fill_(TransitionState.WAIT_SETTLED)
        command = torch.zeros((1, 2))
        for _ in range(5):
            command, state, _ = manager.advance(command, torch.tensor([0.02]), torch.tensor([0.0]))
            self.assertEqual(int(state.item()), TransitionState.WAIT_SETTLED)
        command, state, _ = manager.advance(command, torch.tensor([0.0]), torch.tensor([0.0]))
        self.assertEqual(int(state.item()), TransitionState.WAIT_SETTLED)
        for _ in range(3):
            command, state, _ = manager.advance(command, torch.tensor([0.0]), torch.tensor([0.0]))
            self.assertEqual(int(state.item()), TransitionState.WAIT_SETTLED)
        command, state, _ = manager.advance(command, torch.tensor([0.0]), torch.tensor([0.0]))
        self.assertEqual(int(state.item()), TransitionState.ACCELERATE_FROM_ORIGIN)

    def test_acceleration_starts_from_origin_and_reaches_latest_target(self):
        manager = self.make_manager()
        current = torch.tensor([[0.0, 0.0]])
        manager.state.fill_(TransitionState.WAIT_SETTLED)
        manager.update_target(torch.tensor([[-0.10, -0.025]]), current, torch.zeros(1), torch.zeros(1))
        manager.settle_counter.fill_(manager.settle_steps)
        current, state, _ = manager.advance(current, torch.zeros(1), torch.zeros(1))
        self.assertEqual(int(state.item()), TransitionState.ACCELERATE_FROM_ORIGIN)
        self.assertTrue(torch.allclose(current, torch.zeros_like(current)))
        current, state, _ = manager.advance(current, torch.zeros(1), torch.zeros(1))
        self.assertLess(float(current[0, 0]), 0.0)
        self.assertLess(float(current[0, 1]), 0.0)
        for _ in range(200):
            current, state, _ = manager.advance(current, current[:, 0], current[:, 1])
            if int(state.item()) == TransitionState.TRACK:
                break
        self.assertTrue(torch.allclose(current, torch.tensor([[-0.10, -0.025]]), atol=2e-5))

    def test_reversal_during_acceleration_returns_to_brake(self):
        manager = self.make_manager()
        manager.state.fill_(TransitionState.ACCELERATE_FROM_ORIGIN)
        current = torch.tensor([[-0.08, -0.020]])
        manager.transition_latest_target.copy_(torch.tensor([[-0.10, -0.025]]))
        manager.update_target(torch.tensor([[0.10, 0.025]]), current, current[:, 0], current[:, 1])
        self.assertEqual(int(manager.state.item()), TransitionState.BRAKE_TO_ORIGIN)

    def test_reset_clears_all_transition_runtime_state(self):
        manager = self.make_manager(count=2)
        current = torch.tensor([[0.14, 0.035], [-0.14, -0.035]])
        manager.update_target(
            torch.tensor([[-0.14, -0.035], [0.14, 0.035]]),
            current,
            current[:, 0],
            current[:, 1],
        )
        manager.advance(current, current[:, 0], current[:, 1])
        manager.reset(torch.tensor([0]))
        self.assertEqual(int(manager.state[0].item()), TransitionState.TRACK)
        self.assertTrue(torch.allclose(manager.transition_anchor_command[0], torch.zeros(2)))
        self.assertTrue(torch.allclose(manager.transition_latest_target[0], torch.zeros(2)))
        self.assertEqual(int(manager.settle_counter[0].item()), 0)
        self.assertFalse(bool(manager.transition_active[0].item()))
        self.assertEqual(int(manager.state[1].item()), TransitionState.BRAKE_TO_ORIGIN)

    def test_batch_execution_uses_tensor_state_for_2048_environments(self):
        manager = self.make_manager(count=2048)
        current = torch.zeros((2048, 2))
        targets = torch.zeros_like(current)
        targets[:1024, 0] = 0.10
        targets[1024:, 0] = -0.10
        manager.update_target(targets, current, current[:, 0], current[:, 1])
        applied, state, active = manager.advance(current, current[:, 0], current[:, 1])
        self.assertEqual(tuple(applied.shape), (2048, 2))
        self.assertEqual(tuple(state.shape), (2048,))
        self.assertEqual(tuple(active.shape), (2048,))
        self.assertNotIn("for env_id in", inspect.getsource(manager.advance))


if __name__ == "__main__":
    unittest.main()
