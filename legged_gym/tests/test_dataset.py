import tempfile
import unittest
from pathlib import Path

import numpy as np

from legged_gym.navigation.dataset import (
    ClosedLoopDatasetWriter,
    OracleSample,
)


class DatasetTests(unittest.TestCase):
    def test_writer_saves_depth_and_metadata(self):
        sample = OracleSample(
            depth=np.ones((2, 3), dtype=np.float32),
            robot_xy=(1.0, 2.0),
            robot_yaw=0.5,
            global_goal_xy=(4.0, 5.0),
            local_goal_xy=(1.0, 0.0),
            temporary_world_goal_xy=(2.0, 2.0),
            previous_local_goal_xy=(0.5, 0.0),
            collision=False,
            timestamp_s=1.25,
            episode_id=7,
            waypoint_index=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            writer = ClosedLoopDatasetWriter(directory)
            record = writer.append(sample)
            writer.close()
            self.assertTrue((Path(directory) / record["depth_file"]).is_file())
            self.assertEqual(record["episode_id"], 7)
            self.assertEqual(record["waypoint_index"], 2)
            self.assertIn("temporary_world_goal_xy", (Path(directory) / "records.jsonl").read_text())

    def test_provider_is_not_required_for_serialization(self):
        self.assertTrue(hasattr(OracleSample, "__dataclass_fields__"))


if __name__ == "__main__":
    unittest.main()
