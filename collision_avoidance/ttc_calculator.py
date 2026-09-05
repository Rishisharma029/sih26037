"""Time-to-Collision & Distance-at-Closest-Point-of-Approach calculation."""
import math
from typing import List, Tuple
from interfaces import TrackedObstacle, EgoVehicleState, TTCResult, CollisionRisk

class TTCCalculator:
    """Calculates vector TTC taking into account relative speed and heading."""
    def __init__(self, warning_threshold_s: float = 1.8, critical_threshold_s: float = 0.9):
        self.warning_threshold_s = warning_threshold_s
        self.critical_threshold_s = critical_threshold_s

    def compute_ttc(self, ego_state: EgoVehicleState, obstacles: List[TrackedObstacle]) -> CollisionRisk:
        results = []
        min_ttc = float("inf")
        closest_id = None
        ego_vx = ego_state.twist.speed_mps * math.cos(ego_state.pose.heading_rad)

        for obs in obstacles:
            rel_dx = obs.bbox.center.x
            rel_vx = ego_vx - obs.velocity.x

            if rel_vx > 0.1 and rel_dx > 0.0:
                ttc = rel_dx / rel_vx
            else:
                ttc = float("inf")

            is_crit = ttc < self.critical_threshold_s
            results.append(TTCResult(
                obstacle_id=obs.id,
                ttc_seconds=ttc,
                distance_at_cpa_m=max(0.0, rel_dx),
                is_critical=is_crit
            ))

            if ttc < min_ttc:
                min_ttc = ttc
                closest_id = obs.id

        risk_level = 0.0
        if min_ttc < self.critical_threshold_s:
            risk_level = 1.0
        elif min_ttc < self.warning_threshold_s:
            risk_level = 0.5

        return CollisionRisk(
            timestamp=ego_state.timestamp,
            min_ttc_seconds=min_ttc if min_ttc != float("inf") else 999.0,
            closest_obstacle_id=closest_id,
            risk_level=risk_level,
            ttc_evaluations=results
        )
