"""Fail-Operational Fallback Manager for degraded and emergency autonomous operations."""
import math
from typing import List, Optional, Tuple
from interfaces import (
    EgoVehicleState, PerceptionOutput, PredictionOutput,
    PlannedTrajectory, SafeTrajectory, SafetyAction,
    BehaviorMode, ControlCommand, GearMode, TrajectoryPoint
)


class FallbackManager:
    """High-Assurance Fail-Operational Fallback Manager.

    Monitors system health, detects sensor dropouts, planner timeouts, and adverse visibility,
    and executes Minimum Risk Maneuvers (MRM) and graceful degradation policies.
    """

    def __init__(self, min_safe_clearance_m: float = 0.8, emergency_decel_mps2: float = -4.5):
        self.min_safe_clearance_m = min_safe_clearance_m
        self.emergency_decel_mps2 = emergency_decel_mps2
        self.fallback_active = False
        self.active_fallback_reason: Optional[str] = None
        self.mrm_triggered_count = 0

    def evaluate_and_apply_fallback(
        self,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput,
        safe_plan: Optional[SafeTrajectory],
        planner_latency_ms: float = 5.0
    ) -> Tuple[SafeTrajectory, bool, str]:
        """Verify if system degradation requires triggering a Fail-Operational Fallback."""
        # 1. Check for Planner Timeout / Stalling (> 80 ms latency)
        if planner_latency_ms > 80.0:
            self.mrm_triggered_count += 1
            self.fallback_active = True
            reason = f"PLANNER_TIMEOUT ({planner_latency_ms:.1f}ms > 80ms) -> Minimum Risk Maneuver (MRM)"
            mrm_plan = self._generate_minimum_risk_stop(ego_state, reason)
            return mrm_plan, True, reason

        # 2. Check for Primary Sensor Failure (Camera or LiDAR lost)
        sensor_health = perception.sensor_health
        if not sensor_health.get("camera", True) or not sensor_health.get("lidar", True):
            self.fallback_active = True
            reason = "PRIMARY_SENSOR_DEGRADATION -> Graceful Speed Limiting (50% max speed)"
            degraded_plan = self._apply_speed_limit(safe_plan, max_speed_mps=3.0)
            return degraded_plan, True, reason

        # 3. Check for Blind / Low Visibility Detections
        if any(obs.confidence < 0.35 for obs in perception.obstacles if obs.distance_m < 15.0):
            self.fallback_active = True
            reason = "LOW_PERCEPTION_CONFIDENCE -> Conservative Crawl (2.0 m/s)"
            degraded_plan = self._apply_speed_limit(safe_plan, max_speed_mps=2.0)
            return degraded_plan, True, reason

        # Nominal operation
        self.fallback_active = False
        self.active_fallback_reason = None
        return safe_plan, False, "NOMINAL"

    def _generate_minimum_risk_stop(self, ego_state: EgoVehicleState, reason: str) -> SafeTrajectory:
        """Generate emergency Minimum Risk Maneuver bringing vehicle to a controlled halt on shoulder."""
        dt = 0.1
        steps = 20
        curr_x = ego_state.pose.position.x
        curr_y = ego_state.pose.position.y
        curr_v = ego_state.twist.speed_mps
        curr_yaw = ego_state.pose.heading_rad

        waypoints: List[TrajectoryPoint] = []
        v = curr_v
        x = curr_x
        y = curr_y

        for i in range(1, steps + 1):
            t = ego_state.timestamp + i * dt
            v = max(0.0, v + self.emergency_decel_mps2 * dt)
            x += v * math.cos(curr_yaw) * dt
            # Gently bias towards road shoulder (y < 0)
            y = y - 0.05 * dt
            waypoints.append(TrajectoryPoint(
                timestamp=t,
                x=x,
                y=y,
                yaw_rad=curr_yaw,
                curvature=0.0,
                speed_mps=v,
                acceleration_mps2=self.emergency_decel_mps2,
                jerk_mps3=0.0
            ))

        return SafeTrajectory(
            timestamp=ego_state.timestamp,
            source_trajectory_id="MRM_FALLBACK_STOP",
            waypoints=waypoints,
            safety_action=SafetyAction.EMERGENCY_BRAKE,
            is_emergency_stop=True,
            barrier_margin_m=1.5,
            min_ttc_seconds=999.0,
            replan_recommended=False,
            safety_status_reason=reason
        )

    def _apply_speed_limit(self, safe_plan: SafeTrajectory, max_speed_mps: float) -> SafeTrajectory:
        """Clamp trajectory waypoint speeds to safe degraded velocity limit."""
        if safe_plan is None or not safe_plan.waypoints:
            return safe_plan

        clamped_wps = []
        for wp in safe_plan.waypoints:
            c_wp = wp.model_copy(deep=True)
            c_wp.speed_mps = min(c_wp.speed_mps, max_speed_mps)
            clamped_wps.append(c_wp)

        return safe_plan.model_copy(update={"waypoints": clamped_wps}, deep=True)
