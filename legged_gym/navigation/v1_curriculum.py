"""Performance-gated distance curriculum for the V1 visual corridor."""

import ast
import random


V1_CURRICULUM_LEVELS = (2.5, 3.0, 4.0, 5.0, 6.0)
V1_CURRICULUM_BASE_DISTANCE = 2.0


class V1PerformanceCurriculum:
    """Sample V1 goals and promote distance only after measured performance."""

    FRONTIER_FRACTION = 0.30
    MIN_LEVEL_ITERATIONS = 50
    FRONTIER_SUCCESS_THRESHOLD = 26
    REPLAY_SUCCESS_THRESHOLD = 27
    MAX_COLLISIONS = 1

    def __init__(self, seed=0):
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self.current_level = 0
        self.current_max_distance = V1_CURRICULUM_LEVELS[0]
        self.level_start_iteration = 0
        self.consecutive_pass_count = 0
        self.internal_eval_history = []

    def sample_distances(self, count):
        """Return sampled goal distances and their replay/frontier labels."""
        count = int(count)
        if count < 0:
            raise ValueError("count must be non-negative")
        distances = []
        kinds = []
        for _ in range(count):
            if self._rng.random() < self.FRONTIER_FRACTION:
                kinds.append("frontier")
                distances.append(
                    self._rng.uniform(
                        max(V1_CURRICULUM_BASE_DISTANCE, self.current_max_distance - 0.25),
                        self.current_max_distance,
                    )
                )
            else:
                kinds.append("replay")
                distances.append(
                    self._rng.uniform(
                        V1_CURRICULUM_BASE_DISTANCE,
                        self.current_max_distance,
                    )
                )
        return distances, kinds

    def record_evaluation(
        self,
        iteration,
        frontier_success,
        replay_success,
        collision_count,
        rate_violation_count,
        domain_violation_count,
        hidden_projection_jump_count,
    ):
        """Record one internal evaluation and update the promotion state."""
        iteration = int(iteration)
        metrics = {
            "iteration": iteration,
            "level": self.current_level,
            "max_distance": self.current_max_distance,
            "frontier_success": int(frontier_success),
            "replay_success": int(replay_success),
            "collision_count": int(collision_count),
            "rate_violation_count": int(rate_violation_count),
            "domain_violation_count": int(domain_violation_count),
            "hidden_projection_jump_count": int(hidden_projection_jump_count),
        }
        enough_iterations = (
            iteration - self.level_start_iteration >= self.MIN_LEVEL_ITERATIONS
        )
        passed = (
            int(frontier_success) >= self.FRONTIER_SUCCESS_THRESHOLD
            and int(replay_success) >= self.REPLAY_SUCCESS_THRESHOLD
            and int(collision_count) <= self.MAX_COLLISIONS
            and int(rate_violation_count) == 0
            and int(domain_violation_count) == 0
            and int(hidden_projection_jump_count) == 0
            and enough_iterations
        )
        metrics["minimum_iterations_passed"] = enough_iterations
        metrics["pass"] = passed
        if passed:
            self.consecutive_pass_count += 1
            if (
                self.consecutive_pass_count >= 2
                and self.current_level < len(V1_CURRICULUM_LEVELS) - 1
            ):
                self.current_level += 1
                self.current_max_distance = V1_CURRICULUM_LEVELS[self.current_level]
                self.level_start_iteration = iteration
                self.consecutive_pass_count = 0
                metrics["promoted"] = True
            else:
                metrics["promoted"] = False
        else:
            self.consecutive_pass_count = 0
            metrics["promoted"] = False
        metrics["consecutive_pass_count"] = self.consecutive_pass_count
        self.internal_eval_history.append(metrics)
        return metrics

    def to_dict(self):
        """Return JSON-serializable state for checkpoint/resume."""
        return {
            "seed": self.seed,
            "current_level": self.current_level,
            "current_max_distance": self.current_max_distance,
            "level_start_iteration": self.level_start_iteration,
            "consecutive_pass_count": self.consecutive_pass_count,
            "internal_eval_history": list(self.internal_eval_history),
            "rng_state": repr(self._rng.getstate()),
        }

    @classmethod
    def from_dict(cls, payload):
        curriculum = cls(seed=payload.get("seed", 0))
        curriculum.current_level = int(payload["current_level"])
        curriculum.current_max_distance = float(payload["current_max_distance"])
        curriculum.level_start_iteration = int(payload["level_start_iteration"])
        curriculum.consecutive_pass_count = int(payload["consecutive_pass_count"])
        curriculum.internal_eval_history = list(
            payload.get("internal_eval_history", payload.get("history", []))
        )
        if payload.get("rng_state"):
            curriculum._rng.setstate(ast.literal_eval(payload["rng_state"]))
        return curriculum
