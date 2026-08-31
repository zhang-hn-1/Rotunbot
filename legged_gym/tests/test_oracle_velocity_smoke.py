"""Pure gate-validation tests for the Oracle -> V62 GPU smoke runner."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from legged_gym.scripts.smoke_test_oracle_velocity_stack import (
    validate_approved_s2b_checkpoint,
)


class OracleVelocitySmokeGateTests(unittest.TestCase):
    def _approved_summary(self, checkpoint):
        return {
            "stage": "S2B",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "gate": {"stage": "S2B", "pass": True},
            "collision_count": 0,
            "divergence_count": 0,
            "feasible_domain_violation_count": 0,
            "hidden_projection_jump_count": 0,
            "rate_violation_count": 0,
        }

    def test_accepts_checkpoint_only_when_approved_summary_matches_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pt"
            checkpoint.write_bytes(b"approved checkpoint")
            summary = Path(temp_dir) / "approved_s2b.json"
            summary.write_text(json.dumps(self._approved_summary(checkpoint)))

            approved = validate_approved_s2b_checkpoint(checkpoint, summary)

            self.assertEqual(approved, summary.resolve())

    def test_rejects_arbitrary_checkpoint_even_with_an_approved_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            approved_checkpoint = Path(temp_dir) / "approved.pt"
            approved_checkpoint.write_bytes(b"approved checkpoint")
            experiment_checkpoint = Path(temp_dir) / "experiment.pt"
            experiment_checkpoint.write_bytes(b"arbitrary experiment")
            summary = Path(temp_dir) / "approved_s2b.json"
            summary.write_text(json.dumps(self._approved_summary(approved_checkpoint)))

            with self.assertRaisesRegex(RuntimeError, "does not resolve"):
                validate_approved_s2b_checkpoint(experiment_checkpoint, summary)

    def test_rejects_gate_with_any_nonzero_safety_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pt"
            checkpoint.write_bytes(b"approved checkpoint")
            summary = Path(temp_dir) / "approved_s2b.json"
            payload = self._approved_summary(checkpoint)
            payload["rate_violation_count"] = 1
            summary.write_text(json.dumps(payload))

            with self.assertRaisesRegex(RuntimeError, "rate_violation_count"):
                validate_approved_s2b_checkpoint(checkpoint, summary)

    def test_rejects_malformed_gate_payload_types_with_controlled_error(self):
        """Catch unchecked JSON shapes that previously leaked AttributeError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pt"
            checkpoint.write_bytes(b"approved checkpoint")
            summary = Path(temp_dir) / "approved_s2b.json"
            malformed_gates = (
                [],
                {"stage": "S2B", "pass": 1},
                {"stage": "s2b", "pass": True},
            )

            for malformed_gate in malformed_gates:
                with self.subTest(gate=malformed_gate):
                    payload = self._approved_summary(checkpoint)
                    payload["gate"] = malformed_gate
                    summary.write_text(json.dumps(payload))

                    with self.assertRaisesRegex(RuntimeError, "gate"):
                        validate_approved_s2b_checkpoint(checkpoint, summary)

    def test_rejects_boolean_safety_counts_with_controlled_error(self):
        """Catch JSON booleans accepted accidentally as integer zero counts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pt"
            checkpoint.write_bytes(b"approved checkpoint")
            summary = Path(temp_dir) / "approved_s2b.json"

            for field in (
                "collision_count",
                "divergence_count",
                "feasible_domain_violation_count",
                "hidden_projection_jump_count",
                "rate_violation_count",
            ):
                with self.subTest(field=field):
                    payload = self._approved_summary(checkpoint)
                    payload[field] = False
                    summary.write_text(json.dumps(payload))

                    with self.assertRaisesRegex(RuntimeError, field):
                        validate_approved_s2b_checkpoint(checkpoint, summary)


if __name__ == "__main__":
    unittest.main()
