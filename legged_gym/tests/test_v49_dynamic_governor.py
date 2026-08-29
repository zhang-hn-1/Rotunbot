import unittest

from legged_gym.navigation.v49_dynamic_governor import (
    DynamicGovernorConfig,
    StateDependentReachabilityGovernor,
)
from legged_gym.navigation.v49_dynamic_reachability import (
    DynamicReachabilityTable,
    ReachabilityState,
)


def _rows():
    rows = []
    for current_v in (0.0, 0.10):
        for command_v in (0.0, 0.04, 0.08, 0.10):
            for command_w in (-0.02, 0.0, 0.02):
                yaw_response = command_w if abs(command_v) >= 0.08 else command_w * 0.10
                rows.append({
                    "current_v": current_v,
                    "projected_v": command_v,
                    "projected_w": command_w,
                    **{
                        field: value
                        for horizon in (50, 100, 150, 200)
                        for field, value in (
                            ("predicted_forward_velocity_%dms" % horizon, command_v),
                            ("predicted_yaw_rate_%dms" % horizon, yaw_response),
                        )
                    },
                })
    return rows


def _governor(**overrides):
    values = dict(
        maximum_forward_speed=0.10,
        maximum_yaw_rate=0.04,
        minimum_turn_radius=0.50,
        envelope_fraction=1.0,
        candidate_forward_offsets=(-0.04, 0.0, 0.04),
        candidate_yaw_offsets=(0.0,),
        maximum_forward_command_step=0.05,
        maximum_yaw_command_step=0.04,
        weight_forward_error=1.0,
        weight_yaw_error=8.0,
        weight_command_delta=0.05,
    )
    values.update(overrides)
    config = DynamicGovernorConfig(**values)
    return StateDependentReachabilityGovernor(
        DynamicReachabilityTable.from_rows(_rows()), config
    )


class DynamicGovernorTests(unittest.TestCase):
    def test_reachable_command_is_preserved(self):
        decision = _governor().select_command(
            ReachabilityState(0.10), (0.08, 0.02), (0.08, 0.02)
        )
        self.assertEqual(decision.command, (0.08, 0.02))
        self.assertFalse(decision.fallback)

    def test_low_speed_high_yaw_can_raise_forward_command(self):
        decision = _governor().select_command(
            ReachabilityState(0.0), (0.04, 0.02), (0.0, 0.0)
        )
        self.assertEqual(decision.command, (0.05, 0.02))
        self.assertTrue(decision.forward_modified)
        self.assertFalse(decision.yaw_modified)

    def test_hard_projection_and_command_rate_bounds(self):
        decision = _governor().select_command(
            ReachabilityState(0.10), (0.30, 0.30), (0.0, 0.0)
        )
        self.assertLessEqual(abs(decision.command[0]), 0.10)
        self.assertLessEqual(abs(decision.command[1]), 0.04)
        self.assertLessEqual(abs(decision.command[0]), 0.05)
        self.assertLessEqual(abs(decision.command[1]), 0.04)

    def test_no_direction_reversal(self):
        decision = _governor().select_command(
            ReachabilityState(-0.08), (0.08, 0.02), (-0.08, 0.0)
        )
        self.assertLessEqual(decision.command[0], 0.0)

    def test_tie_break_is_deterministic(self):
        governor = _governor(weight_yaw_error=0.0, weight_command_delta=0.0)
        first = governor.select_command(ReachabilityState(0.10), (0.08, 0.0), (0.0, 0.0))
        second = governor.select_command(ReachabilityState(0.10), (0.08, 0.0), (0.0, 0.0))
        self.assertEqual(first.command, second.command)

    def test_out_of_coverage_uses_static_fallback(self):
        decision = _governor().select_command(
            ReachabilityState(0.50), (0.08, 0.02), (0.08, 0.02)
        )
        self.assertTrue(decision.fallback)
        self.assertEqual(decision.command, (0.08, 0.02))
        self.assertTrue(decision.prediction.out_of_coverage)


if __name__ == "__main__":
    unittest.main()
