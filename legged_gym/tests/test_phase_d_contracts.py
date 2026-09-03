import unittest

from legged_gym.navigation.phase_d_contracts import (
    classify_phase_d_failure,
    require_isaacgym_depth,
    resolve_phase_d_timing,
    transition_manager_stall_evidence,
)


class PhaseDContractTests(unittest.TestCase):
    def test_runtime_timing_is_derived(self):
        timing = resolve_phase_d_timing(0.005, 4, 5.0)
        self.assertAlmostEqual(timing.policy_dt_s, 0.02)
        self.assertEqual(timing.hold_policy_steps, 10)
        self.assertEqual(timing.hold_physics_steps, 40)
        self.assertAlmostEqual(timing.upper_command_hz, 5.0)

    def test_non_integral_high_level_period_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_phase_d_timing(0.005, 4, 7.0)

    def test_formal_depth_contract_is_fail_closed(self):
        self.assertTrue(require_isaacgym_depth("isaacgym", "isaacgym"))
        for requested, actual in (("fallback", "fallback"), ("isaacgym", "fallback"), ("fallback", "isaacgym")):
            with self.assertRaises(RuntimeError):
                require_isaacgym_depth(requested, actual)

    def test_failure_precedence_does_not_infer_stall_from_timeout(self):
        self.assertEqual(classify_phase_d_failure(timeout=True), "TIMEOUT")
        self.assertEqual(
            classify_phase_d_failure(timeout=True, transition_manager_stall=True),
            "TRANSITION_MANAGER_STALL",
        )
        self.assertEqual(
            classify_phase_d_failure(collision=True, transition_manager_stall=True),
            "COLLISION",
        )

    def test_stall_requires_sustained_evidence(self):
        rows = [
            {
                "command_target_v_mps": 0.10,
                "command_target_w_rps": 0.0,
                "applied_v_mps": 0.0,
                "applied_w_rps": 0.0,
                "actual_v_mps": 0.0,
                "actual_w_rps": 0.0,
                "global_goal_distance_m": 1.0,
                "transition_active": True,
                "goal_success_radius_m": 0.35,
            }
            for _ in range(5)
        ]
        self.assertTrue(transition_manager_stall_evidence(rows, minimum_window=5))
        rows[-1]["transition_active"] = False
        self.assertFalse(transition_manager_stall_evidence(rows, minimum_window=5))


if __name__ == "__main__":
    unittest.main()
