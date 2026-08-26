"""A thin evaluation wrapper around the unchanged uniform-4150 P2P task."""

from legged_gym.envs.rotunbot.target_point.rotunbot_target_repro import (
    RotunbotTargetRepro,
)


class HierarchicalP2P(RotunbotTargetRepro):
    """Preserve P2P math while allowing intermediate waypoint continuation."""

    def __init__(self, *args, **kwargs):
        self._intermediate_goal = False
        super().__init__(*args, **kwargs)

    def set_intermediate_goal(self, enabled):
        """Suppress only success-triggered reset for non-global waypoints."""
        self._intermediate_goal = bool(enabled)

    def check_termination(self):
        super().check_termination()
        if self._intermediate_goal:
            # success_buf remains the original P2P success result and is still
            # available to the evaluator; only its automatic reset side effect
            # is removed for an intermediate waypoint.
            self.reset_buf &= ~self.success_buf
