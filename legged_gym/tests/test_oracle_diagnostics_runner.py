import unittest


class OracleDiagnosticsRunnerTests(unittest.TestCase):
    def test_diagnostic_runner_reuses_oracle_gate_entrypoint(self):
        from legged_gym.scripts import evaluate_oracle_diagnostics as runner

        self.assertTrue(callable(runner.run_gate))
        self.assertTrue(callable(runner.parse_script_args))


if __name__ == "__main__":
    unittest.main()
