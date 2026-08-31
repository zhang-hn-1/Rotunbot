"""Continuous geometry-only local waypoint Oracle for corridor scenarios."""

import math

import numpy as np


class CorridorWaypointOracle:
    """Return capability-bounded world-frame local goals for a centerline.

    This component owns no policy or actuator interface.  Consumers receive a
    two-coordinate world waypoint and must route it through the Local Goal and
    frozen V62 velocity stack themselves.  The preferred candidate is always
    on the centerline.  If it exceeds the trained local-distance or bearing
    capability, the returned point is a bounded local-goal fallback and need
    not itself lie on the centerline.
    """

    def __init__(
        self, scenario, local_distance_limit, bearing_limit_deg, lookahead_m=0.6
    ):
        self.scenario = scenario
        self.local_distance_limit = float(local_distance_limit)
        self.bearing_limit_rad = math.radians(float(bearing_limit_deg))
        self.lookahead_m = float(lookahead_m)
        if self.local_distance_limit <= 0.0:
            raise ValueError("local_distance_limit must be positive")
        if not 0.0 < self.bearing_limit_rad <= math.pi:
            raise ValueError("bearing_limit_deg must be in (0, 180]")
        if self.lookahead_m <= 0.0:
            raise ValueError("lookahead_m must be positive")

        self._centerline = np.asarray(scenario.centerline, dtype=np.float64)
        if self._centerline.ndim != 2 or self._centerline.shape[1] != 2:
            raise ValueError("scenario centerline must have shape [N, 2]")
        if len(self._centerline) < 2:
            raise ValueError("scenario centerline must contain at least two points")
        self._segment_delta = np.diff(self._centerline, axis=0)
        self._segment_lengths = np.linalg.norm(self._segment_delta, axis=1)
        if np.any(self._segment_lengths <= 1.0e-9):
            raise ValueError("scenario centerline cannot contain duplicate points")
        self._arc_lengths = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(self._segment_lengths))
        )
        self._turn_start_arc_lengths = tuple(
            float(self._arc_lengths[int(turn.start_index)])
            for turn in scenario.turns
        )

    @staticmethod
    def _pose(pose):
        values = np.asarray(pose, dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("pose must be finite [x, y, yaw]")
        return values

    def _project_arc_length(self, position_xy):
        start = self._centerline[:-1]
        relative = position_xy.reshape(1, 2) - start
        fractions = np.clip(
            np.sum(relative * self._segment_delta, axis=1)
            / np.square(self._segment_lengths),
            0.0,
            1.0,
        )
        projections = start + fractions[:, None] * self._segment_delta
        index = int(np.argmin(np.linalg.norm(projections - position_xy, axis=1)))
        return float(self._arc_lengths[index] + fractions[index] * self._segment_lengths[index])

    def _centerline_point(self, arc_length):
        arc_length = float(np.clip(arc_length, 0.0, self._arc_lengths[-1]))
        index = min(
            int(np.searchsorted(self._arc_lengths, arc_length, side="right") - 1),
            len(self._segment_lengths) - 1,
        )
        fraction = (arc_length - self._arc_lengths[index]) / self._segment_lengths[index]
        return self._centerline[index] + fraction * self._segment_delta[index]

    def _shortened_lookahead_arc_length(self, current_arc_length):
        target = min(current_arc_length + self.lookahead_m, self._arc_lengths[-1])
        for turn_start in self._turn_start_arc_lengths:
            if current_arc_length < turn_start < target:
                return turn_start
            if turn_start <= current_arc_length < turn_start + self.lookahead_m:
                # Ramp the lookahead in from zero after the centerline starts
                # turning.  This joins the pre-turn target at turn_start and
                # reaches the normal lookahead without a target jump.
                return min(
                    current_arc_length + (current_arc_length - turn_start),
                    self._arc_lengths[-1],
                )
        return target

    def _cap_to_local_goal_capability(self, position_xy, yaw, candidate_xy):
        delta = candidate_xy - position_xy
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-12:
            return candidate_xy.copy()
        distance = min(distance, self.local_distance_limit)
        bearing = math.atan2(float(delta[1]), float(delta[0])) - float(yaw)
        bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
        bearing = float(np.clip(bearing, -self.bearing_limit_rad, self.bearing_limit_rad))
        heading = float(yaw) + bearing
        return position_xy + distance * np.asarray((math.cos(heading), math.sin(heading)))

    def next_waypoint(self, pose):
        """Return one finite world-frame XY waypoint, never a control command."""
        position_x, position_y, yaw = self._pose(pose)
        position = np.asarray((position_x, position_y), dtype=np.float64)
        current_arc = self._project_arc_length(position)
        target_arc = self._shortened_lookahead_arc_length(current_arc)
        candidate = self._centerline_point(target_arc)
        return self._cap_to_local_goal_capability(position, yaw, candidate)
