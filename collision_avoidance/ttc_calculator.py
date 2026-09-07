"""Time-to-Collision & Distance-at-Closest-Point-of-Approach calculation."""
import math
from typing import List, Tuple, Optional
from interfaces import TrackedObstacle, EgoVehicleState, TTCResult, CollisionRisk


class TTCCalculator:
    """Calculates vector TTC taking into account 2D relative closing speed and collision swaths."""

    def __init__(
        self,
        caution_threshold_s: float = 4.0,
        warning_threshold_s: float = 2.0,
        critical_threshold_s: float = 1.0
    ):
        self.caution_threshold_s = caution_threshold_s
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
            # Obstacle is already in standardized ego coordinates
            x_ego = obs.bbox.center.x
            y_ego = obs.bbox.center.y
            dist = math.hypot(x_ego, y_ego)

            # Relative velocity in ego frame:
            # obs.velocity.x is forward relative velocity (negative = closing in)
            vx_ego = obs.velocity.x
            vy_ego = obs.velocity.y

            ego_speed = ego_state.twist.speed_mps

            # 2D Closing velocity projected along line-of-sight range vector
            if obs.is_static:
                closing_speed = ego_speed * (x_ego / max(0.1, dist)) if x_ego > 0.0 else 0.0
            elif vx_ego < 0.0:
                # Direct relative approach velocity in ego frame (negative = closing in)
                closing_speed = -(x_ego * vx_ego + y_ego * vy_ego) / max(0.1, dist)
            else:
                # Forward speed specified (e.g. lead vehicle slower than ego)
                rel_vx = ego_speed - vx_ego
                closing_speed = (x_ego * rel_vx - y_ego * vy_ego) / max(0.1, dist) if rel_vx > 0.0 else 0.0

            # Collision swath evaluation (vehicle width ~ 1.8m + safety envelope -> half swath 1.15m)
            # Active only for obstacles ahead in forward corridor (x_ego > 0)
            if x_ego > 0.0 and abs(y_ego) < 1.20 and closing_speed > 0.15:
                ttc = dist / closing_speed
            elif 0.0 < x_ego < 30.0 and abs(y_ego) < 1.20 and ego_speed > 0.3 and obs.is_static:
                ttc = x_ego / ego_speed
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

        # Graded overall collision risk assessment [0.0 - 1.0]
        if min_ttc < self.critical_threshold_s:
            risk_level = 1.0
        elif min_ttc <= self.warning_threshold_s:
            # Interpolate risk between 0.50 and 0.95
            ratio = (self.warning_threshold_s - min_ttc) / max(0.01, self.warning_threshold_s - self.critical_threshold_s)
            risk_level = round(0.50 + 0.45 * ratio, 2)
        elif min_ttc <= self.caution_threshold_s:
            # Interpolate risk between 0.10 and 0.49
            ratio = (self.caution_threshold_s - min_ttc) / max(0.01, self.caution_threshold_s - self.warning_threshold_s)
            risk_level = round(0.10 + 0.39 * ratio, 2)
        else:
            risk_level = 0.0

        return CollisionRisk(
            timestamp=ego_state.timestamp,
            min_ttc_seconds=min_ttc if min_ttc != float("inf") else 999.0,
            closest_obstacle_id=closest_id,
            risk_level=risk_level,
            ttc_evaluations=results
        )
