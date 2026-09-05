"""Time-to-Collision & Distance-at-Closest-Point-of-Approach calculation."""
import math
from typing import List, Tuple, Optional
from interfaces import TrackedObstacle, EgoVehicleState, TTCResult, CollisionRisk


class TTCCalculator:
    """Calculates vector TTC taking into account 2D relative closing speed and collision swaths."""

    def __init__(self, warning_threshold_s: float = 1.8, critical_threshold_s: float = 0.85):
        self.warning_threshold_s = warning_threshold_s
        self.critical_threshold_s = critical_threshold_s

    def compute_ttc(self, ego_state: EgoVehicleState, obstacles: List[TrackedObstacle]) -> CollisionRisk:
        results: List[TTCResult] = []
        min_ttc = float("inf")
        closest_id: Optional[str] = None

        ego_x = ego_state.pose.position.x
        ego_y = ego_state.pose.position.y
        ego_vx = ego_state.twist.speed_mps * math.cos(ego_state.pose.heading_rad)
        ego_vy = ego_state.twist.speed_mps * math.sin(ego_state.pose.heading_rad)

        for obs in obstacles:
            dx = obs.bbox.center.x - ego_x
            dy = obs.bbox.center.y - ego_y
            dist = math.hypot(dx, dy)

            rel_vx = ego_vx - obs.velocity.x
            rel_vy = ego_vy - obs.velocity.y

            # 2D Closing velocity projected along range vector
            closing_speed = (dx * rel_vx + dy * rel_vy) / max(0.1, dist)

            # Collision swath for ego path (within 1.1m lateral corridor of vehicle trajectory)
            if dx > 0.0 and abs(dy) < 1.10 and closing_speed > 0.15:
                ttc = dist / closing_speed
            elif 0.0 < dx < 25.0 and abs(dy) < 1.10 and ego_vx > 0.3 and obs.is_static:
                ttc = dx / ego_vx
            else:
                ttc = float("inf")

            is_crit = ttc < self.critical_threshold_s
            results.append(TTCResult(
                obstacle_id=obs.id,
                ttc_seconds=ttc if ttc != float("inf") else 999.0,
                distance_at_cpa_m=dist,
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
