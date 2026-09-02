import copy
import math
import importlib
import unittest

import numpy as np

from legged_gym.navigation.v1_t_junction import (
    build_t_junction_geometry,
    classify_t_branch,
    wall_actor_centers,
)
from legged_gym.navigation.v1_t_junction_metrics import aggregate_t_gate


def _trainer_module():
    return importlib.import_module(
        "legged_gym.scripts.train_sru_visual_t_junction_imitation"
    )


def _student_module():
    return importlib.import_module(
        "legged_gym.scripts.eval_sru_visual_t_junction"
    )


def _collector_module():
    return importlib.import_module(
        "legged_gym.scripts.collect_sru_visual_t_junction_teacher"
    )


def _audit_module():
    return importlib.import_module(
        "legged_gym.scripts.audit_t_junction_teacher_dataset"
    )


def _dataset_step(episode_id, step_id, done=False):
    return {
        "episode_id": episode_id,
        "step_id": step_id,
        "depth": [[0.5] * 32 for _ in range(8)],
        "goal_xy_robot": [1.0, 0.0],
        "proprioception": [0.0] * 12,
        "previous_command": [0.0, 0.0],
        "previous_actual_velocity": [0.0, 0.0],
        "teacher_command": [0.2, 0.0],
        "actual_velocity": [0.0, 0.0],
        "governor_command": [0.0, 0.0],
        "projection_command": [0.2, 0.0],
        "done": done,
        "success": done,
        "collision": False,
        "goal_distance": 0.3,
    }


def _materialized_episode(episode_id, rows):
    """Build the V1 episode shape emitted by TeacherSequenceWriter."""
    return {
        "episode_id": episode_id,
        "episode_ids": [episode_id] * len(rows),
        "sequence_length": len(rows),
        "num_sequences": int(math.ceil(len(rows) / 16)),
        **{
            key: [row[key] for row in rows]
            for key in rows[0]
            if key != "episode_id"
        },
    }


def _t_dataset():
    first = _dataset_step(0, 0, done=True)
    second = _dataset_step(1, 0, done=True)

    dataset = {
        "schema_version": 1,
        "step_fields": [
            "episode_id",
            "step_id",
            "depth",
            "goal_xy_robot",
            "proprioception",
            "previous_command",
            "previous_actual_velocity",
            "teacher_command",
            "actual_velocity",
            "governor_command",
            "projection_command",
            "done",
            "success",
            "collision",
            "goal_distance",
        ],
        "sequence_length": 16,
        "episodes": [
            _materialized_episode(0, [first]),
            _materialized_episode(1, [second]),
        ],
        "metadata": {
            "depth_backend_requested": "isaacgym",
            "depth_backend_actual": "isaacgym",
            "scenarios": ["T_LEFT", "T_RIGHT"],
            "episode_scenarios": ["T_LEFT", "T_RIGHT"],
            "seed": 2026,
            "episodes_per_scene": 1,
            "geometry": {
                "T_LEFT": {"goal_xy": [2.5, 2.5], "width_m": 3.0},
                "T_RIGHT": {"goal_xy": [2.5, -2.5], "width_m": 3.0},
            },
            "command_ranges": {
                "v_cmd": [-0.25, 0.25],
                "w_cmd": [-0.10, 0.10],
            },
            "episode_provenance": {
                "0": {
                    "scenario": "T_LEFT",
                    "goal": [2.5, 2.5],
                    "initial_pose": [0.0, 0.0, 0.28],
                    "initial_yaw": 0.0,
                    "horizon": 2250,
                },
                "1": {
                    "scenario": "T_RIGHT",
                    "goal": [2.5, -2.5],
                    "initial_pose": [0.0, 0.0, 0.28],
                    "initial_yaw": 0.0,
                    "horizon": 2250,
                },
            },
        },
    }
    return dataset


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


def _aabb_occupies(aabbs, point):
    point = np.asarray(point, dtype=np.float64)
    return any(
        np.all(np.abs(point - np.asarray(center)) <= np.asarray(half_extent))
        for center, half_extent in aabbs
    )


class TJunctionNavigationTests(unittest.TestCase):
    def test_t_teacher_collector_module_is_available(self):
        """Breaks if the requested T teacher entrypoint is not shipped."""
        self.assertIsNotNone(
            importlib.util.find_spec(
                "legged_gym.scripts.collect_sru_visual_t_junction_teacher"
            )
        )

    def test_t_mixed_bc_sampler_enforces_integer_1_3_5_ratio(self):
        """Breaks if mixed BC silently uses an unweighted or fractional sampler."""
        trainer = _trainer_module()
        datasets = {
            "straight": {"sequence_length": 16, "episodes": [{"episode_id": 0}]},
            "l_turn": {"sequence_length": 16, "episodes": [{"episode_id": 0}]},
            "t_junction": {"sequence_length": 16, "episodes": [{"episode_id": 0}]},
        }
        sampled = trainer.build_weighted_dataset(datasets)
        self.assertEqual(sampled["effective_sample_distribution"], {
            "straight": 1, "l_turn": 3, "t_junction": 5,
        })
        self.assertEqual(
            [item["source"] for item in sampled["samples"]],
            ["straight"] + ["l_turn"] * 3 + ["t_junction"] * 5,
        )
        with self.assertRaises(ValueError):
            trainer.build_weighted_dataset({**datasets, "t_junction": {
                "sequence_length": 15, "episodes": [{"episode_id": 0}]
            }})

    def test_t_student_episode_record_and_goal_modes_are_auditable(self):
        """Breaks if student evaluation loses release evidence or ablation meaning."""
        student = _student_module()
        record = student.make_t_student_episode_record(
            scenario="T_RIGHT", episode_id=3, pair_id="pair-03", seed=7,
            goal=(2.5, -2.5), initial_pose=(0.0, 0.0, 0.28), initial_yaw=0.0,
            horizon=2250, episode_steps=42, success=True, collision=False,
            timeout=False, branch_prediction="RIGHT", wrong_turn=False,
            turn_completion=True, exit_reached=True, failure_trace=[],
        )
        self.assertEqual(record["policy_role"], "student")
        self.assertEqual(record["expected_branch"], "RIGHT")
        self.assertTrue({
            "scenario", "episode_id", "pair_id", "seed", "goal", "initial_pose",
            "initial_yaw", "horizon", "success", "collision", "timeout",
            "wrong_turn", "turn_completion", "exit", "failure_trace",
            "depth_backend_actual",
        }.issubset(record))
        self.assertEqual(student.goal_for_observation((2.5, 2.5), (2.5, -2.5), "normal"), (2.5, 2.5))
        self.assertEqual(student.goal_for_observation((2.5, 2.5), (2.5, -2.5), "zero"), (0.0, 0.0))
        self.assertEqual(student.goal_for_observation((2.5, 2.5), (2.5, -2.5), "swapped"), (2.5, -2.5))
        with self.assertRaises(ValueError):
            student.goal_for_observation((2.5, 2.5), (2.5, -2.5), "bad")

    def test_t_student_pair_builder_requires_exact_complete_coverage(self):
        """Breaks if paired counterfactuals reuse or omit gate records."""
        student = _student_module()
        records = []
        for index in range(20):
            records.extend([
                {
                    "episode_id": "L-%02d" % index, "scenario": "T_LEFT",
                    "pair_id": "p-%02d" % index, "seed": 2026,
                    "initial_pose": (0.0, 0.0, 0.4), "initial_yaw": 0.0,
                    "horizon": 2250,
                },
                {
                    "episode_id": "R-%02d" % index, "scenario": "T_RIGHT",
                    "pair_id": "p-%02d" % index, "seed": 2026,
                    "initial_pose": (0.0, 0.0, 0.4), "initial_yaw": 0.0,
                    "horizon": 2250,
                },
            ])
        pairs = student.build_counterfactual_pairs(records)
        self.assertEqual(len(pairs), 20)
        self.assertEqual({item for pair in pairs for item in pair}, {
            record["episode_id"] for record in records
        })
        with self.assertRaises(ValueError):
            student.build_counterfactual_pairs(records[:-1])

    def test_t_teacher_branch_commands_require_opposite_yaw_signs(self):
        """Breaks if a T_LEFT turn is commanded with the T_RIGHT yaw sign."""
        collector = _collector_module()
        self.assertEqual(collector.expected_branch_yaw_sign("T_LEFT"), 1)
        self.assertEqual(collector.expected_branch_yaw_sign("T_RIGHT"), -1)
        geometry = build_t_junction_geometry("T_LEFT")
        waypoints = collector.t_navigation_waypoints(geometry)
        self.assertEqual(waypoints.shape, (4, 2))
        np.testing.assert_allclose(waypoints[-1], geometry.scenario.goal_xy)
        self.assertTrue(np.allclose(waypoints[2], (1.75, 2.5)))
        self.assertFalse(collector.has_wrong_turn_command(0.03, "T_LEFT"))
        self.assertTrue(collector.has_wrong_turn_command(-0.03, "T_LEFT"))
        self.assertFalse(collector.has_wrong_turn_command(-0.03, "T_RIGHT"))
        self.assertTrue(collector.has_wrong_turn_command(0.03, "T_RIGHT"))

    def test_t_teacher_classifies_wrong_turn_turn_completion_and_exit(self):
        """Breaks if branch selection or terminal progress are misclassified."""
        collector = _collector_module()
        left = collector.classify_t_episode_progress(
            "T_LEFT", (2.5, 1.0), waypoint_index=2, exit_reached=True
        )
        self.assertEqual(left["branch_prediction"], "LEFT")
        self.assertFalse(left["wrong_turn"])
        self.assertTrue(left["turn_completed"])
        self.assertTrue(left["exit_reached"])
        wrong = collector.classify_t_episode_progress(
            "T_RIGHT", (2.5, 1.0), waypoint_index=1, exit_reached=False
        )
        self.assertEqual(wrong["branch_prediction"], "LEFT")
        self.assertTrue(wrong["wrong_turn"])
        self.assertFalse(wrong["turn_completed"])
        self.assertFalse(wrong["exit_reached"])

    def test_t_teacher_uses_terminal_snapshot_and_observed_branch_evidence(self):
        """Breaks if automatic reset root pose can erase terminal branch evidence."""
        collector = _collector_module()

        class FakeTerminalEnv:
            env_origins = np.asarray([[10.0, -4.0, 0.0]])
            terminal_position = np.asarray([[12.5, -3.0]])
            root_states = np.asarray([[10.0, -4.0, 0.28]])

        self.assertEqual(
            collector.terminal_local_xy(FakeTerminalEnv()), (2.5, 1.0)
        )
        progress = collector.classify_t_episode_progress(
            "T_RIGHT",
            terminal_local_xy=(2.5, -1.0),
            waypoint_index=2,
            exit_reached=True,
            observed_branches=("LEFT",),
        )
        self.assertEqual(progress["branch_prediction"], "RIGHT")
        self.assertTrue(progress["wrong_turn"])
        self.assertTrue(progress["turn_completed"])

    def test_t_teacher_preserves_primitive_wrong_branch_excursion_after_recovery(self):
        """Breaks if a wrong branch between macro boundaries can be forgotten."""
        collector = _collector_module()
        observed = collector.record_t_branch_occupancy((), (2.5, -1.0))
        observed = collector.record_t_branch_occupancy(observed, (2.5, 1.0))
        progress = collector.classify_t_episode_progress(
            "T_LEFT",
            (2.5, 1.0),
            waypoint_index=2,
            exit_reached=True,
            observed_branches=observed,
        )
        self.assertEqual(observed, ("RIGHT", "LEFT"))
        self.assertTrue(progress["wrong_turn"])

    def test_t_teacher_installs_waypoint_goal_before_capture(self):
        """Breaks if stored waypoint goals diverge from actor observations."""
        collector = _collector_module()

        class FakeObservationEnv:
            def __init__(self):
                self.calls = []

            def set_observation_goal_world(self, goal_world):
                self.calls.append(("set", np.asarray(goal_world).copy()))

            def compute_observations(self):
                self.calls.append(("compute", None))

            def _goal_xy_robot(self):
                return np.asarray([[1.25, -0.5]])

        env = FakeObservationEnv()
        goal_xy_robot = collector.install_observation_goal(env, np.asarray([12.5, -3.0]))
        self.assertEqual(env.calls[0][0], "set")
        np.testing.assert_allclose(env.calls[0][1], [[12.5, -3.0]])
        self.assertEqual(env.calls[1], ("compute", None))
        self.assertIsNotNone(goal_xy_robot)
        np.testing.assert_allclose(goal_xy_robot, [[1.25, -0.5]])

    def test_t_teacher_episode_record_contains_auditable_required_fields(self):
        """Breaks if a T run omits a release-gate field from episode evidence."""
        collector = _collector_module()
        record = collector.make_t_episode_record(
            scenario="T_LEFT",
            episode_id=4,
            seed=2026,
            goal=(2.5, 2.5),
            initial_pose=(0.0, 0.0, 0.28),
            initial_yaw=0.0,
            horizon=2250,
            episode_steps=111,
            macro_steps=12,
            success=True,
            collision=False,
            timeout=False,
            progress={
                "branch_prediction": "LEFT",
                "wrong_turn": False,
                "turn_completed": True,
                "exit_reached": True,
            },
        )
        self.assertTrue(
            {
                "timeout",
                "turn_completed",
                "exit_reached",
                "scenario",
                "seed",
                "goal",
                "initial_pose",
                "episode_steps",
                "macro_steps",
                "failure_trace",
            }.issubset(record)
        )
        self.assertEqual(record["expected_branch"], "LEFT")
        self.assertEqual(record["policy_role"], "teacher")

    def test_t_teacher_rejects_non_isaacgym_depth_backend(self):
        """Breaks if the collector could silently label fallback depth."""
        collector = _collector_module()
        self.assertEqual(collector.require_isaacgym_depth_backend("isaacgym"), "isaacgym")
        with self.assertRaises(RuntimeError):
            collector.require_isaacgym_depth_backend("fallback")

    def test_t_dataset_audit_checks_v1_schema_finite_terminal_chronology_and_counts(self):
        """Breaks if the audit accepts malformed or non-real-depth T labels."""
        audit = _audit_module()
        result = audit.audit_t_teacher_dataset(_t_dataset())
        self.assertTrue(result["finite"])
        self.assertTrue(result["terminal_done"])
        self.assertTrue(result["chronological_step_ids"])
        self.assertEqual(result["scenario_counts"], {"T_LEFT": 1, "T_RIGHT": 1})
        self.assertEqual(result["macro_steps"], 2)
        malformed = _t_dataset()
        malformed["metadata"]["depth_backend_actual"] = "fallback"
        with self.assertRaises(ValueError):
            audit.audit_t_teacher_dataset(malformed)
        nonfinite = _t_dataset()
        nonfinite["episodes"][0]["depth"][0][0][0] = math.nan
        with self.assertRaises(ValueError):
            audit.audit_t_teacher_dataset(nonfinite)
        early_done = _t_dataset()
        episode = early_done["episodes"][0]
        for field, values in episode.items():
            if isinstance(values, list):
                values.append(values[-1])
        episode["step_id"] = [0, 1]
        episode["done"] = [True, True]
        episode["sequence_length"] = 2
        with self.assertRaises(ValueError):
            audit.audit_t_teacher_dataset(early_done)

    def test_t_dataset_audit_rejects_non_t16_lengths_shapes_and_scalar_values(self):
        """Breaks if malformed V1 fields can pass a nominal T dataset audit."""
        audit = _audit_module()
        cases = []

        non_t16 = _t_dataset()
        non_t16["sequence_length"] = 15
        cases.append(non_t16)

        mismatched_length = _t_dataset()
        mismatched_length["episodes"][0]["teacher_command"].append([0.2, 0.0])
        cases.append(mismatched_length)

        string_episode_length = _t_dataset()
        string_episode_length["episodes"][0]["sequence_length"] = "1"
        cases.append(string_episode_length)

        wrong_depth_shape = _t_dataset()
        wrong_depth_shape["episodes"][0]["depth"] = [[[0.5] * 32 for _ in range(7)]]
        cases.append(wrong_depth_shape)

        wrong_command_shape = _t_dataset()
        wrong_command_shape["episodes"][0]["projection_command"] = [[0.2]]
        cases.append(wrong_command_shape)

        string_value = _t_dataset()
        string_value["episodes"][0]["teacher_command"][0][0] = "0.2"
        cases.append(string_value)

        none_value = _t_dataset()
        none_value["episodes"][0]["goal_xy_robot"][0][1] = None
        cases.append(none_value)

        for malformed in cases:
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    audit.audit_t_teacher_dataset(malformed)

    def test_t_dataset_audit_requires_writer_num_sequences_for_each_episode(self):
        """Breaks if a malformed writer materialization can pass the T audit."""
        audit = _audit_module()
        cases = []

        missing = _t_dataset()
        del missing["episodes"][0]["num_sequences"]
        cases.append(missing)

        non_integral = _t_dataset()
        non_integral["episodes"][0]["num_sequences"] = 1.0
        cases.append(non_integral)

        wrong_count = _t_dataset()
        rows = [_dataset_step(0, step, done=step == 16) for step in range(17)]
        wrong_count["episodes"][0] = _materialized_episode(0, rows)
        wrong_count["episodes"][0]["num_sequences"] = 1
        cases.append(wrong_count)

        for malformed in cases:
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    audit.audit_t_teacher_dataset(malformed)

    def test_t_dataset_audit_rejects_duplicate_or_non_increasing_episode_ids(self):
        """Breaks if provenance coverage can collapse multiple episodes onto one ID."""
        audit = _audit_module()
        duplicate_ids = _t_dataset()
        left, right = duplicate_ids["episodes"]
        duplicate_ids["episodes"] = [
            copy.deepcopy(left),
            copy.deepcopy(left),
            copy.deepcopy(right),
            copy.deepcopy(right),
        ]
        duplicate_ids["metadata"]["episodes_per_scene"] = 2
        duplicate_ids["metadata"]["episode_scenarios"] = [
            "T_LEFT", "T_LEFT", "T_RIGHT", "T_RIGHT"
        ]

        decreasing_ids = _t_dataset()
        decreasing_ids["episodes"] = [
            copy.deepcopy(decreasing_ids["episodes"][1]),
            copy.deepcopy(decreasing_ids["episodes"][0]),
        ]
        decreasing_ids["episodes"][0]["episode_id"] = 1
        decreasing_ids["episodes"][0]["episode_ids"] = [1]
        decreasing_ids["episodes"][1]["episode_id"] = 0
        decreasing_ids["episodes"][1]["episode_ids"] = [0]
        decreasing_ids["metadata"]["episode_provenance"] = {
            "1": copy.deepcopy(_t_dataset()["metadata"]["episode_provenance"]["0"]),
            "0": copy.deepcopy(_t_dataset()["metadata"]["episode_provenance"]["1"]),
        }

        for malformed in (duplicate_ids, decreasing_ids):
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    audit.audit_t_teacher_dataset(malformed)

    def test_t_dataset_audit_rejects_swapped_episode_provenance_side_or_goal(self):
        """Breaks if a dataset episode can be attributed to the opposite T side."""
        audit = _audit_module()

        swapped_side = _t_dataset()
        swapped_side["metadata"]["episode_provenance"]["0"]["scenario"] = "T_RIGHT"
        with self.assertRaises(ValueError):
            audit.audit_t_teacher_dataset(swapped_side)

        swapped_goal = _t_dataset()
        swapped_goal["metadata"]["episode_provenance"]["0"]["goal"] = [2.5, -2.5]
        with self.assertRaises(ValueError):
            audit.audit_t_teacher_dataset(swapped_goal)

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

    def test_geometry_uses_explicit_actor_aabbs_with_an_open_stem_and_front_wall(self):
        geometry = build_t_junction_geometry("T_LEFT")

        expected_centers = {
            (1.25, 1.5),
            (1.25, -1.5),
            (1.0, 2.75),
            (4.0, 2.75),
            (2.5, 4.0),
            (1.0, -2.75),
            (4.0, -2.75),
            (2.5, -4.0),
        }
        self.assertEqual(set(wall_actor_centers(geometry.wall_segments, 3.0)), expected_centers)
        self.assertEqual(len(geometry.wall_segments), len(geometry.obstacle_aabbs))
        for (start, end), (center, half_extent) in zip(
            geometry.wall_segments, geometry.obstacle_aabbs
        ):
            np.testing.assert_allclose(center, 0.5 * (np.asarray(start) + np.asarray(end)))
            self.assertEqual(len(center), 2)
            self.assertTrue(all(value > 0.0 for value in half_extent))

        # The stem is traversable through the junction, while both branch
        # corridors remain open between their side walls and terminate at caps.
        for x in np.linspace(0.0, 2.5, 11):
            with self.subTest(stem_x=x):
                self.assertFalse(_aabb_occupies(geometry.obstacle_aabbs, (x, 0.0)))
        self.assertFalse(_aabb_occupies(geometry.obstacle_aabbs, (2.5, 0.0)))
        self.assertFalse(_aabb_occupies(geometry.obstacle_aabbs, (2.5, 2.5)))
        self.assertFalse(_aabb_occupies(geometry.obstacle_aabbs, (2.5, -2.5)))
        self.assertTrue(_aabb_occupies(geometry.obstacle_aabbs, (2.5, 4.01)))

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

    def test_pair_rejects_reused_episode_ids_across_pairs(self):
        records, pairs = _paired_student_records()

        with self.assertRaises(ValueError):
            aggregate_t_gate(records, pairs=pairs + [pairs[0]], ablations={})

    def test_student_pairing_must_cover_every_record_exactly_once(self):
        records, pairs = _paired_student_records()

        with self.assertRaises(ValueError):
            aggregate_t_gate(records, pairs=pairs[:-1], ablations={})

    def test_pair_accuracy_requires_opposite_expected_predictions(self):
        records, pairs = _paired_student_records()
        records[1]["branch_prediction"] = "LEFT"

        result = aggregate_t_gate(records, pairs=pairs, ablations={})

        self.assertEqual(result["goal_consistency_rate"], 19.0 / 20.0)
        self.assertTrue(result["pass"])


if __name__ == "__main__":
    unittest.main()
