import importlib.util
import os
import sys
import unittest

import numpy as np


def _load_evaluator_module():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    scripts_dir = os.path.join(project_root, "legged_gym", "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    path = os.path.join(scripts_dir, "evaluate_vel_sru50_structured_random.py")
    spec = importlib.util.spec_from_file_location("structured_random_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StructuredVelocityCommandCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_evaluator_module()
        cls.count = 64
        cls.radius = 2.0
        cls.max_v = 0.25
        cls.max_w = 0.10

    def generate(self, family, seed=1234):
        return self.module.generate_transition_family(
            family,
            self.count,
            seed,
            self.radius,
            self.max_v,
            self.max_w,
        )

    def assert_feasible(self, commands):
        self.assertTrue(np.all(np.abs(commands[:, 0]) <= self.max_v + 1.0e-6))
        self.assertTrue(np.all(np.abs(commands[:, 1]) <= self.max_w + 1.0e-6))
        self.assertTrue(
            np.all(
                np.abs(commands[:, 1])
                <= np.abs(commands[:, 0]) / self.radius + 1.0e-6
            )
        )

    def test_all_families_are_deterministic_and_have_expected_shape(self):
        self.assertEqual(len(self.module.FAMILY_ORDER), 12)
        for family in self.module.FAMILY_ORDER:
            first_from, first_to = self.generate(family, seed=44)
            second_from, second_to = self.generate(family, seed=44)
            self.assertEqual(first_from.shape, (self.count, 2), family)
            self.assertEqual(first_to.shape, (self.count, 2), family)
            np.testing.assert_array_equal(first_from, second_from)
            np.testing.assert_array_equal(first_to, second_to)
            self.assert_feasible(first_from)

    def test_fixed_w_velocity_reversal_is_guaranteed(self):
        before, after = self.generate("fixed_w_v_reversal")
        np.testing.assert_allclose(before[:, 1], after[:, 1], atol=1.0e-7)
        self.assertTrue(np.all(before[:, 0] * after[:, 0] < 0.0))
        self.assertTrue(np.all(np.abs(before[:, 1]) >= 0.012 - 1.0e-6))
        self.assert_feasible(before)
        self.assert_feasible(after)

    def test_constant_curvature_reversal_flips_both_channels(self):
        before, after = self.generate("constant_curvature_reversal")
        self.assertTrue(np.all(before[:, 0] * after[:, 0] < 0.0))
        self.assertTrue(np.all(before[:, 1] * after[:, 1] < 0.0))
        np.testing.assert_allclose(
            before[:, 1] / before[:, 0],
            after[:, 1] / after[:, 0],
            rtol=1.0e-5,
            atol=1.0e-6,
        )

    def test_fixed_v_yaw_reversal_is_guaranteed(self):
        before, after = self.generate("fixed_v_w_reversal")
        np.testing.assert_allclose(before[:, 0], after[:, 0], atol=1.0e-7)
        np.testing.assert_allclose(before[:, 1], -after[:, 1], atol=1.0e-7)
        self.assertTrue(np.all(np.abs(before[:, 1]) > 0.0))

    def test_infeasible_family_really_contains_projection_requests(self):
        _, after = self.generate("infeasible_low_speed_high_yaw")
        violates_radius = np.abs(after[:, 1]) > np.abs(after[:, 0]) / self.radius
        self.assertGreater(float(np.mean(violates_radius)), 0.80)

    def test_turning_stop_family_contains_in_place_yaw_requests(self):
        _, after = self.generate("turn_stop_or_restart")
        in_place = (np.abs(after[:, 0]) < 1.0e-8) & (np.abs(after[:, 1]) > 0.0)
        self.assertGreaterEqual(int(np.count_nonzero(in_place)), self.count // 2)

    def test_quadrant_jump_contains_all_sign_change_patterns(self):
        before, after = self.generate("all_quadrant_jump")
        v_changed = np.sign(before[:, 0]) != np.sign(after[:, 0])
        w_changed = np.sign(before[:, 1]) != np.sign(after[:, 1])
        patterns = set(zip(v_changed.tolist(), w_changed.tolist()))
        self.assertEqual(
            patterns,
            {(False, False), (False, True), (True, False), (True, True)},
        )


if __name__ == "__main__":
    unittest.main()
