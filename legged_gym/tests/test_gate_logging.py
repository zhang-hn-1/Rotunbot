import json
import tempfile
import unittest
from pathlib import Path

from legged_gym.navigation.evaluation_logging import EpisodeLogger


class GateLoggingTests(unittest.TestCase):
    def test_logger_round_trips_summary_and_steps_to_json(self):
        logger = EpisodeLogger({"gate": "single_local_goal", "seed": 3})
        logger.record_step(
            time_s=0.02,
            robot_xy=[0.0, 0.0],
            world_goal_xy=[1.0, 0.0],
            distance=1.0,
            speed=0.0,
            action=[0.1, -0.2],
        )
        logger.finish(success=True, reason="local_goal", completion_time_s=1.2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.json"
            logger.write_json(path)
            payload = json.loads(path.read_text())
        self.assertEqual(payload["summary"]["success"], True)
        self.assertEqual(payload["summary"]["reason"], "local_goal")
        self.assertEqual(len(payload["trajectory"]), 1)
        self.assertEqual(payload["trajectory"][0]["action"], [0.1, -0.2])

    def test_logger_writes_csv_with_action_and_distance_fields(self):
        logger = EpisodeLogger({"gate": "single_local_goal"})
        logger.record_step(
            time_s=0.0,
            robot_xy=[0.0, 0.0],
            world_goal_xy=[0.5, 0.0],
            distance=0.5,
            speed=0.0,
            action=[0.0, 0.0],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            logger.write_csv(path)
            text = path.read_text()
        self.assertIn("time_s", text.splitlines()[0])
        self.assertIn("distance", text.splitlines()[0])
        self.assertIn("0.5", text)


if __name__ == "__main__":
    unittest.main()
