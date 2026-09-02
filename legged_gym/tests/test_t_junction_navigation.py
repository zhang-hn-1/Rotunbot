import math
import unittest

import numpy as np

from legged_gym.navigation.v1_t_junction import (
    build_t_junction_geometry,
    classify_t_branch,
    wall_actor_centers,
)
from legged_gym.navigation.v1_t_junction_metrics import aggregate_t_gate


class _TorchLikeFinite:
    """Minimal tensor-like finite predicate without importing Isaac Gym."""

    def __init__(self, finite):
        self._finite = finite

    def isfinite(self):
        return self

    def all(self):
        return self

    def item(self):
        return self._finite


def _record(side, index, **overrides):
    expected = "LEFT" if side == "T_LEFT" else "RIGHT"
    row = {
        "episode_id": "%s-%02d" % (side, index),
        "scenario": side,
        "policy_role": "student",
        "success": 1,
        "collision": 0,
        "timeout": 0,
        "wrong_turn": 0,
        "turn_completion": 1,
        "exit": 1,
        "branch_prediction": expected,
        "expected_branch": expected,
        "depth_backend_actual": "isaacgym",
        "seed": 17,
        "initial_pose": (0.0, 0.0, 0.28),
        "initial_yaw": 0.0,
        "horizon": 1000,
    }
    row.update(overrides)
    return row


def _paired_student_records(successes_per_side=20):
    records = []
    pairs = []
    for index in range(20):
        for side in ("T_LEFT", "T_RIGHT"):
            records.append(
                _record(
                    side,
                    index,
                    success=int(index < successes_per_side),
                    turn_completion=int(index < successes_per_side),
                    exit=int(index < successes_per_side),
                )
            )
        pairs.append(("T_LEFT-%02d" % index, "T_RIGHT-%02d" % index))
    return records, pairs


class TJunctionNavigationTests(unittest.TestCase):
    def test_left_and_right_share_walls_but_mirror_goals(self):
        left = build_t_junction_geometry("T_LEFT")
        right = build_t_junction_geometry("T_RIGHT")

        self.assertEqual(left.scenario.start_xy.tolist(), right.scenario.start_xy.tolist())
        np.testing.assert_allclose(left.wall_segments, right.wall_segments)
        np.testing.assert_allclose(left.obstacle_aabbs, right.obstacle_aabbs)
        np.testing.assert_allclose(left.waypoints[:, 0], right.waypoints[:, 0])
        np.testing.assert_allclose(left.waypoints[:, 1], -right.waypoints[:, 1])
        self.assertEqual(left.branch_direction, 1)
        self.assertEqual(right.branch_direction, -1)

    def test_geometry_uses_center_half_extent_aabbs_and_open_t_topology(self):
        geometry = build_t_junction_geometry("T_LEFT")

        expected_centers = {
            (1.25, -1.5),
            (1.25, 1.5),
            (1.0, -1.25),
            (1.0, 1.25),
            (4.0, -1.25),
            (4.0, 1.25),
        }
        self.assertEqual(set(wall_actor_centers(geometry.wall_segments, 3.0)), expected_centers)
        self.assertNotIn((1.25, 0.0), expected_centers)
        self.assertEqual(
            geometry.obstacle_aabbs,
            (
                ((1.25, -1.5), (1.25, 0.05)),
                ((1.25, 1.5), (1.25, 0.05)),
                ((4.0, 1.25), (0.05, 1.25)),
                ((1.0, 1.25), (0.05, 1.25)),
                ((1.0, -1.25), (0.05, 1.25)),
                ((4.0, -1.25), (0.05, 1.25)),
            ),
        )
        for center, half_extent in geometry.obstacle_aabbs:
            self.assertEqual(len(center), 2)
            self.assertTrue(all(value > 0.0 for value in half_extent))

    def test_geometry_has_fixed_dimensions_and_finite_values(self):
        geometry = build_t_junction_geometry("left")
        self.assertEqual(geometry.scenario.width_m, 3.0)
        self.assertEqual(geometry.waypoints.shape, (3, 2))
        self.assertEqual(geometry.reach_radius_m, 0.35)
        self.assertTrue(np.isfinite(geometry.waypoints).all())
        self.assertTrue(np.isfinite(np.asarray(geometry.wall_segments)).all())
        self.assertTrue(np.isfinite(np.asarray(geometry.obstacle_aabbs)).all())

    def test_branch_classifier_has_deadband(self):
        self.assertEqual(classify_t_branch((2.5, 1.0)), "LEFT")
        self.assertEqual(classify_t_branch((2.5, -1.0)), "RIGHT")
        self.assertEqual(classify_t_branch((2.5, 0.1), deadband_m=0.35), "UNDECIDED")

    def test_geometry_rejects_nonfinite_or_nonpositive_inputs(self):
        for kwargs in (
            {"width_m": math.nan},
            {"stem_length_m": math.inf},
            {"branch_length_m": 0.0},
            {"reach_radius_m": -0.1},
        ):
            with self.assertRaises(ValueError):
                build_t_junction_geometry("left", **kwargs)

    def test_student_gate_passes_exactly_nineteen_of_twenty_per_side(self):
        records, pairs = _paired_student_records(successes_per_side=19)

        result = aggregate_t_gate(records, pairs=pairs, ablations={"zero_goal": {}})

        self.assertTrue(result["pass"])
        self.assertEqual(result["by_scenario"]["T_LEFT"]["success_rate"], 0.95)
        self.assertEqual(result["by_scenario"]["T_RIGHT"]["turn_completion_rate"], 0.95)
        self.assertTrue(result["checks"]["by_scenario"]["T_LEFT"]["success_rate_ge_0.95"])
        self.assertTrue(result["checks"]["goal_consistency_rate_ge_0.95"])

    def test_student_gate_fails_eighteen_of_twenty_on_one_side(self):
        records, pairs = _paired_student_records(successes_per_side=20)
        for record in records:
            if record["scenario"] == "T_LEFT" and record["episode_id"].endswith(("18", "19")):
                record.update(success=0, turn_completion=0, exit=0)

        result = aggregate_t_gate(records, pairs=pairs, ablations={})

        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["by_scenario"]["T_LEFT"]["success_rate_ge_0.95"])
        self.assertTrue(result["checks"]["by_scenario"]["T_RIGHT"]["success_rate_ge_0.95"])

    def test_student_gate_rejects_any_collision_or_wrong_turn(self):
        for key in ("collision", "wrong_turn"):
            with self.subTest(key=key):
                records, pairs = _paired_student_records()
                records[0][key] = 1
                result = aggregate_t_gate(records, pairs=pairs, ablations={})
                self.assertFalse(result["pass"])
                self.assertFalse(
                    result["checks"]["by_scenario"]["T_LEFT"]["%s_rate_eq_0" % key]
                )

    def test_teacher_gate_has_zero_collision_and_wrong_turn_requirements(self):
        records, pairs = _paired_student_records()
        for record in records:
            record["policy_role"] = "teacher"
        records[0].update(timeout=1, turn_completion=0)

        result = aggregate_t_gate(records, pairs=pairs, ablations={})
        self.assertTrue(result["pass"])
        records[0]["collision"] = 1
        self.assertFalse(aggregate_t_gate(records, pairs=pairs, ablations={})["pass"])

    def test_gate_rejects_missing_or_non_isaacgym_backend_and_nested_nonfinite_values(self):
        records, pairs = _paired_student_records()
        for overrides in (
            {"depth_backend_actual": "fallback"},
            {"depth_backend_actual": None},
            {"details": {"scores": [np.float32(math.nan)]}},
            {"details": {"tensor": _TorchLikeFinite(False)}},
        ):
            with self.subTest(overrides=overrides):
                altered = [dict(record) for record in records]
                altered[0].update(overrides)
                with self.assertRaises(ValueError):
                    aggregate_t_gate(altered, pairs=pairs, ablations={})
        with self.assertRaises(ValueError):
            aggregate_t_gate(records, pairs=pairs, ablations={"zero_goal": _TorchLikeFinite(False)})

    def test_pair_rejects_duplicate_side_identity_and_missing_metadata(self):
        records, pairs = _paired_student_records()
        left, right = pairs[0]
        cases = (
            ((left, left), records),
            ((left, "T_LEFT-01"), records),
            ((left, right), records + [dict(records[0])]),
            ((left, right), [
                {key: value for key, value in row.items() if key != "seed"}
                if row["episode_id"] == right else row
                for row in records
            ]),
        )
        for pair, case_records in cases:
            with self.subTest(pair=pair):
                with self.assertRaises(ValueError):
                    aggregate_t_gate(case_records, pairs=[pair], ablations={})

    def test_pair_accuracy_requires_opposite_expected_predictions(self):
        records, pairs = _paired_student_records()
        records[1]["branch_prediction"] = "LEFT"

        result = aggregate_t_gate(records, pairs=pairs, ablations={})

        self.assertEqual(result["goal_consistency_rate"], 19.0 / 20.0)
        self.assertTrue(result["pass"])


if __name__ == "__main__":
    unittest.main()
