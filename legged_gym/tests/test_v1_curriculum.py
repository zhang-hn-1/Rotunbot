import json
import tempfile
import unittest


class V1CurriculumTests(unittest.TestCase):
    def test_sampling_uses_replay_and_frontier_bounds(self):
        from legged_gym.navigation.v1_curriculum import V1PerformanceCurriculum

        curriculum = V1PerformanceCurriculum(seed=4)
        distances, kinds = curriculum.sample_distances(1000)
        self.assertTrue(all(2.0 <= value <= 2.5 for value in distances))
        self.assertGreater(sum(kind == "replay" for kind in kinds), 600)
        self.assertGreater(sum(kind == "frontier" for kind in kinds), 200)

    def test_failed_level_freezes_and_passes_promote_after_two_evaluations(self):
        from legged_gym.navigation.v1_curriculum import V1PerformanceCurriculum

        curriculum = V1PerformanceCurriculum(seed=4)
        failed = curriculum.record_evaluation(
            iteration=50,
            frontier_success=16,
            replay_success=30,
            collision_count=0,
            rate_violation_count=0,
            domain_violation_count=0,
            hidden_projection_jump_count=0,
        )
        self.assertFalse(failed["pass"])
        self.assertEqual(curriculum.current_max_distance, 2.5)
        self.assertEqual(curriculum.consecutive_pass_count, 0)
        for iteration in (100, 150):
            result = curriculum.record_evaluation(
                iteration=iteration,
                frontier_success=26,
                replay_success=27,
                collision_count=1,
                rate_violation_count=0,
                domain_violation_count=0,
                hidden_projection_jump_count=0,
            )
        self.assertTrue(result["pass"])
        self.assertEqual(curriculum.current_level, 1)
        self.assertEqual(curriculum.current_max_distance, 3.0)
        self.assertEqual(curriculum.level_start_iteration, 150)

    def test_state_round_trip_preserves_resume_position_and_history(self):
        from legged_gym.navigation.v1_curriculum import V1PerformanceCurriculum

        curriculum = V1PerformanceCurriculum(seed=4)
        curriculum.record_evaluation(50, 26, 27, 0, 0, 0, 0)
        payload = curriculum.to_dict()
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as handle:
            json.dump(payload, handle)
            handle.seek(0)
            restored = V1PerformanceCurriculum.from_dict(json.load(handle))
        self.assertEqual(restored.to_dict(), payload)

    def test_promotion_requires_minimum_level_iterations(self):
        from legged_gym.navigation.v1_curriculum import V1PerformanceCurriculum

        curriculum = V1PerformanceCurriculum(seed=4)
        result = curriculum.record_evaluation(49, 30, 30, 0, 0, 0, 0)
        self.assertFalse(result["pass"])
        self.assertEqual(curriculum.current_level, 0)


if __name__ == "__main__":
    unittest.main()
