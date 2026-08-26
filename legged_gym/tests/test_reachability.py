import tempfile
import unittest
from pathlib import Path

import numpy as np

from legged_gym.scripts.measure_reachability import summarize_bearing_dependence

from legged_gym.navigation.reachability import (
    ReachabilityEnvelope,
    ReachabilitySample,
    load_samples,
    save_samples,
)


class ReachabilityTests(unittest.TestCase):
    def _sample(self, displacement):
        return ReachabilitySample(
            action0=0.5,
            action1=-0.25,
            displacement_body_xy=tuple(displacement),
            steady_state_velocity_body_xy=(0.4, 0.0),
            rise_time_s=0.3,
            cross_axis_coupling=0.1,
            action_clipping=False,
            joint_response=(0.2, -0.1),
        )

    def test_raw_samples_round_trip_as_json(self):
        samples = [self._sample((1.0, 0.0)), self._sample((0.0, 0.5))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            save_samples(path, samples)
            loaded = load_samples(path)
        self.assertEqual(loaded, samples)

    def test_envelope_comes_from_measured_radial_limits(self):
        envelope = ReachabilityEnvelope.from_samples(
            [self._sample((1.0, 0.0)), self._sample((0.0, 0.5))],
            angular_bins=8,
        )
        inside = envelope.filter([0.75, 0.0])
        outside = envelope.filter([2.0, 0.0])
        np.testing.assert_allclose(inside, [0.75, 0.0])
        np.testing.assert_allclose(outside, [1.0, 0.0])

    def test_filter_preserves_bearing_when_clipping(self):
        envelope = ReachabilityEnvelope(
            bearings_rad=(0.0, np.pi / 2.0),
            max_radius_m=(1.0, 0.5),
        )
        result = envelope.filter([0.0, 1.0])
        np.testing.assert_allclose(result, [0.0, 0.5], atol=1e-12)

    def test_bearing_summary_combines_sweep_and_coverage(self):
        samples = [self._sample((1.0, 0.0)), self._sample((0.0, 0.2))]
        coverage = {
            "case_summaries": [
                {"distance_m": 1.0, "bearing_deg": 0.0, "episodes": 3,
                 "success_rate": 1.0, "timeout_count": 0},
                {"distance_m": 1.0, "bearing_deg": 90.0, "episodes": 3,
                 "success_rate": 0.0, "timeout_count": 3},
            ]
        }
        summary = summarize_bearing_dependence(samples, coverage)
        self.assertEqual(summary["0"]["measured_action_count"], 1)
        self.assertEqual(summary["0"]["reachable_action_count"], 1)
        self.assertEqual(summary["0"]["coverage_success_rate"], 1.0)
        self.assertEqual(summary["90"]["coverage_timeout_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
