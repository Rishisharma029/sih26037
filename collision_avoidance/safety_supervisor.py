"""Dedicated Safety Supervisory Layer for Trajectory Arbitration & Collision Avoidance."""
import math
from typing import Optional, List, Tuple
from interfaces import (
    PlannedTrajectory, SafeTrajectory, SafetyAction,
    EgoVehicleState, PerceptionOutput, PredictionOutput,
    MotionIntent, ObstacleClass
)
from .ttc_calculator import TTCCalculator
from .control_barrier_functions import ControlBarrierFilter


class SafetySupervisoryLayer:
    """Dedicated Independent Safety Supervisory Layer.

    Acts as an external arbiter outside the planner to verify candidate trajectories,
    enforce Control Barrier Function invariants, trigger Emergency Re-plans for unexpected
    actor incursions, and activate Autonomous Emergency Braking (AEB) when TTC breaches thresholds.
    """

    def __init__(
        self,
        aeb_ttc_threshold_s: float = 1.0,
        replan_ttc_threshold_s: float = 2.0,
        slowdown_ttc_threshold_s: float = 4.0,
        min_barrier_dist_m: float = 1.5,
        warning_ttc_threshold_s: Optional[float] = None
    ):
        self.aeb_ttc_threshold_s = aeb_ttc_threshold_s
        self.replan_ttc_threshold_s = replan_ttc_threshold_s if warning_ttc_threshold_s is None else warning_ttc_threshold_s
        self.slowdown_ttc_threshold_s = slowdown_ttc_threshold_s
        self.min_barrier_dist_m = min_barrier_dist_m
        
        self.ttc_calc = TTCCalculator(
            caution_threshold_s=self.slowdown_ttc_threshold_s,
            warning_threshold_s=self.replan_ttc_threshold_s,
            critical_threshold_s=self.aeb_ttc_threshold_s
        )
        self.cbf_filter = ControlBarrierFilter(min_safe_dist_m=min_barrier_dist_m)
        self.aeb_trigger_count = 0
        self.emergency_replan_count = 0

    def supervise(
        self,
        planned: PlannedTrajectory,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput,
        prediction: PredictionOutput
    ) -> SafeTrajectory:
        """Independently verifies planned trajectory against hard safety constraints.
        
        Enforces hierarchical safety arbitration rules:
        - Tier 1: TTC < 1.0s or Critical Physical Collision Barrier Breach -> Emergency Brake (AEB)
        - Tier 2: 1.0s <= TTC < 2.0s or Dynamic Incursion -> Emergency Replan
        - Tier 3: 2.0s <= TTC <= 4.0s or CBF Verge/Obstacle Proximity -> Slowdown / Caution
        - Tier 4: TTC > 4.0s -> Normal Pass-through / Planned Action
        """
        # 1. Evaluate Dynamic Time-to-Collision
        risk = self.ttc_calc.compute_ttc(ego_state, perception.obstacles)

        # 2. Evaluate Control Barrier Functions (CBF)
        filtered_traj, barrier_violated, barrier_margin = self.cbf_filter.filter_trajectory(
            planned, ego_state, perception
        )

        # 3. Check for Unexpected Obstacle Incursion / Rapid Cut-ins
        unexpected_incursion, incursion_id = self._check_unexpected_incursion(
            planned, ego_state, perception, prediction
        )

        closest_dist = min([obs.distance_m for obs in perception.obstacles], default=999.0)

        safety_action = SafetyAction.NONE
        is_e_stop = False
        replan_recommended = False
        safety_status_reason = "TRAJECTORY_VERIFIED_SAFE"

        # Tier 1: Critical TTC (< 1.0s) or Imminent Physical Impact -> Autonomous Emergency Braking (AEB)
        if risk.min_ttc_seconds < self.aeb_ttc_threshold_s or (closest_dist < 1.2 and ego_state.twist.speed_mps > 0.5):
            self.aeb_trigger_count += 1
            safety_action = SafetyAction.EMERGENCY_BRAKE
            is_e_stop = True
            replan_recommended = False
            safety_status_reason = f"CRITICAL_TTC_AEB_TRIGGERED (TTC={risk.min_ttc_seconds:0.2f}s)"
            for wp in filtered_traj.waypoints:
                wp.speed_mps = 0.0
                wp.acceleration_mps2 = -6.5

        # Tier 2: 1.0s <= TTC < 2.0s or Unexpected Dynamic Incursion -> Emergency Re-plan
        elif unexpected_incursion or risk.min_ttc_seconds < self.replan_ttc_threshold_s:
            self.emergency_replan_count += 1
            safety_action = SafetyAction.EMERGENCY_REPLAN
            replan_recommended = True
            if unexpected_incursion:
                safety_status_reason = f"UNEXPECTED_OBSTACLE_INCURSION ({incursion_id}) -> EMERGENCY_REPLAN"
            else:
                safety_status_reason = f"EMERGENCY_REPLAN_TRIGGERED (TTC={risk.min_ttc_seconds:0.2f}s)"
            # Apply defensive slowdown crawl while awaiting re-plan
            for wp in filtered_traj.waypoints:
                wp.speed_mps = min(wp.speed_mps, max(1.5, ego_state.twist.speed_mps * 0.5))
                wp.acceleration_mps2 = -2.5

        # Tier 3: 2.0s <= TTC <= 4.0s or CBF Verge/Obstacle Proximity -> Slowdown / Caution
        elif risk.min_ttc_seconds <= self.slowdown_ttc_threshold_s or barrier_violated:
            if barrier_violated:
                safety_action = SafetyAction.CONTROL_BARRIER_OVERRIDE
                safety_status_reason = f"CONTROL_BARRIER_OVERRIDE (Margin={barrier_margin:0.2f}m)"
            else:
                safety_action = SafetyAction.ADAPTIVE_CRUISE_SLOWDOWN
                safety_status_reason = f"CAUTION_SLOWDOWN (TTC={risk.min_ttc_seconds:0.2f}s)"
                # Scale speed down proportionally to TTC
                target_v_safe = max(1.5, risk.min_ttc_seconds * 1.35)
                for wp in filtered_traj.waypoints:
                    if wp.speed_mps > target_v_safe:
                        wp.speed_mps = round(target_v_safe, 2)
                        wp.acceleration_mps2 = min(wp.acceleration_mps2, -1.8)

        # Tier 4: TTC > 4.0s -> Normal Pass-through / Planned Action Confirmation
        elif planned.behavior_mode.value.startswith("NUDGE"):
            safety_action = SafetyAction.CORRIDOR_NUDGE
            safety_status_reason = "PLANNED_NUDGE_VERIFIED"
        elif planned.behavior_mode.value == "FOLLOW":
            safety_action = SafetyAction.ADAPTIVE_CRUISE_SLOWDOWN
            safety_status_reason = "SAFE_FOLLOW_VERIFIED"

        return SafeTrajectory(
            timestamp=ego_state.timestamp,
            source_trajectory_id=planned.trajectory_id,
            waypoints=filtered_traj.waypoints,
            safety_action=safety_action,
            is_emergency_stop=is_e_stop,
            barrier_margin_m=min(closest_dist, barrier_margin),
            min_ttc_seconds=risk.min_ttc_seconds,
            replan_recommended=replan_recommended,
            safety_status_reason=safety_status_reason
        )

    def _check_unexpected_incursion(
        self,
        planned: PlannedTrajectory,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput,
        prediction: PredictionOutput
    ) -> Tuple[bool, Optional[str]]:
        """Detects if an actor is cutting-in or crossing directly into the planned path swath."""
        for agent in prediction.agents:
            if agent.is_high_risk or agent.primary_intent in [
                MotionIntent.CUTTING_IN,
                MotionIntent.ERRATIC_SWERVE,
                MotionIntent.CROSSING_PATH
            ]:
                # Check intersection between agent's trajectories and planned waypoints
                for traj in agent.trajectories:
                    if traj.probability >= 0.30:
                        for step_idx, ego_wp in enumerate(planned.waypoints[:8]):
                            if step_idx < len(traj.waypoints):
                                ag_wp = traj.waypoints[step_idx]
                                dist = math.hypot(ego_wp.x - ag_wp.position.x, ego_wp.y - ag_wp.position.y)
                                if dist < 1.8: # Incursion into travel envelope
                                    return True, agent.id

        return False, None


# Backward-compatible alias
EmergencyBrakeSupervisory = SafetySupervisoryLayer
