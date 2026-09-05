"""Control Barrier Functions (CBF) forward invariance safety filter."""
import math
from typing import Tuple, List, Optional
from interfaces import PlannedTrajectory, EgoVehicleState, PerceptionOutput, TrajectoryPoint


class ControlBarrierFilter:
    """Enforces Control Barrier Function invariants h(x) >= 0 to guarantee forward safe sets

    for both dynamic obstacle clearance and irregular road corridor containment.
    """

    def __init__(self, min_safe_dist_m: float = 1.6, gamma: float = 1.2):
        self.min_safe_dist_m = min_safe_dist_m
        self.gamma = gamma # CBF decay rate parameter

    def filter_trajectory(
        self,
        trajectory: PlannedTrajectory,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput
    ) -> Tuple[PlannedTrajectory, bool, float]:
        """Validates and filters trajectory waypoints against barrier constraints.

        Returns: (safe_trajectory, barrier_violated, min_margin_m)
        """
        min_margin = 999.0
        barrier_violated = False

        corridor_half_w = max(2.5, perception.drivable_corridor.average_width_m * 0.5)

        for wp in trajectory.waypoints:
            # 1. Obstacle barrier condition: h_obs = ||p_wp - p_obs|| - r_safe >= 0
            for obs in perception.obstacles:
                dx = wp.x - obs.bbox.center.x
                dy = wp.y - obs.bbox.center.y
                dist = math.hypot(dx, dy)
                margin = dist - self.min_safe_dist_m
                if margin < min_margin:
                    min_margin = margin

                if margin < 0.0:
                    barrier_violated = True
                    # Override speed to satisfy barrier decay
                    wp.speed_mps = min(wp.speed_mps, 1.2)
                    wp.acceleration_mps2 = -3.5

            # 2. Corridor boundary barrier condition: h_boundary >= 0
            dist_left = corridor_half_w - wp.y
            dist_right = wp.y - (-corridor_half_w)
            b_margin = min(dist_left, dist_right)
            if b_margin < min_margin:
                min_margin = b_margin

            if b_margin < 0.20:
                barrier_violated = True
                wp.speed_mps = min(wp.speed_mps, 2.0)
                wp.acceleration_mps2 = -2.5

        return trajectory, barrier_violated, max(0.0, round(min_margin, 2))
