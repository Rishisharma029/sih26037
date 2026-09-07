"""Dedicated Safety Supervisory Layer for Trajectory Arbitration & Collision Avoidance."""
import math
from typing import Optional, List, Tuple, Dict
from interfaces import (
    PlannedTrajectory, SafeTrajectory, SafetyAction, TrajectoryPoint,
    EgoVehicleState, PerceptionOutput, PredictionOutput,
    MotionIntent, ObstacleClass
)
from .ttc_calculator import TTCCalculator
from .control_barrier_functions import ControlBarrierFilter


class SafetySupervisoryLayer:
    """Dedicated Independent Hard Safety Supervisor & Trajectory Gatekeeper.

    Enforces absolute hard safety boundaries:
      Planner proposes -> Safety supervisor validates -> SAFE: pass to control / UNSAFE: reject & override.
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
        
        # Diagnostics and intervention telemetry
        self.total_audited = 0
        self.passed_count = 0
        self.rejected_count = 0
        self.aeb_trigger_count = 0
        self.emergency_replan_count = 0
        self.slowdown_count = 0
        self.last_gate_status = "PASSED_SAFE"
        self.last_rejection_reason: Optional[str] = None

    def supervise(
        self,
        planned: PlannedTrajectory,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput,
        prediction: PredictionOutput
    ) -> SafeTrajectory:
        """Independently verifies planned trajectory against hard safety constraints."""
        self.total_audited += 1

        # 1. Dynamic Time-to-Collision assessment
        risk = self.ttc_calc.compute_ttc(ego_state, perception.obstacles)

        # 2. Control Barrier Functions (CBF) verification
        filtered_traj, barrier_violated, barrier_margin = self.cbf_filter.filter_trajectory(
            planned, ego_state, perception
        )
        min_ditch_margin = self.cbf_filter.last_min_ditch_margin
        min_obs_margin = self.cbf_filter.last_min_obstacle_margin

        # 3. Check for Unexpected Dynamic Incursions / Rapid Cut-ins
        unexpected_incursion, incursion_id = self._check_unexpected_incursion(
            planned, ego_state, perception, prediction
        )

        closest_dist = min([obs.distance_m for obs in perception.obstacles], default=999.0)

        # -------------------------------------------------------------------
        # TIER 1: CRITICAL TTC (< 1.0s) OR IMMINENT COLLISION -> AEB
        # -------------------------------------------------------------------
        if risk.min_ttc_seconds < self.aeb_ttc_threshold_s or (closest_dist < 1.2 and ego_state.twist.speed_mps > 0.5):
            self.rejected_count += 1
            self.aeb_trigger_count += 1
            self.last_gate_status = "REJECTED_AEB"
            self.last_rejection_reason = "REJECTED_CRITICAL_TTC"
            
            for wp in filtered_traj.waypoints:
                wp.speed_mps = 0.0
                wp.acceleration_mps2 = -6.5

            return SafeTrajectory(
                timestamp=ego_state.timestamp,
                source_trajectory_id=planned.trajectory_id,
                waypoints=filtered_traj.waypoints,
                safety_action=SafetyAction.EMERGENCY_BRAKE,
                is_emergency_stop=True,
                is_rejected=True,
                rejection_reason="REJECTED_CRITICAL_TTC",
                supervisor_gate_status="REJECTED_AEB",
                barrier_margin_m=min(closest_dist, barrier_margin),
                min_ttc_seconds=risk.min_ttc_seconds,
                replan_recommended=False,
                safety_status_reason=f"CRITICAL_TTC_AEB_TRIGGERED (TTC={risk.min_ttc_seconds:0.2f}s)"
            )

        # -------------------------------------------------------------------
        # TIER 2: HARD DITCH BREACH (ditch_margin < 0.0m)
        # -------------------------------------------------------------------
        if min_ditch_margin < 0.0:
            self.rejected_count += 1
            self.aeb_trigger_count += 1
            self.last_gate_status = "REJECTED_AEB"
            self.last_rejection_reason = "REJECTED_DITCH_BREACH"
            
            for wp in filtered_traj.waypoints:
                wp.speed_mps = 0.0
                wp.acceleration_mps2 = -6.5

            return SafeTrajectory(
                timestamp=ego_state.timestamp,
                source_trajectory_id=planned.trajectory_id,
                waypoints=filtered_traj.waypoints,
                safety_action=SafetyAction.EMERGENCY_BRAKE,
                is_emergency_stop=True,
                is_rejected=True,
                rejection_reason="REJECTED_DITCH_BREACH",
                supervisor_gate_status="REJECTED_AEB",
                barrier_margin_m=min_ditch_margin,
                min_ttc_seconds=risk.min_ttc_seconds,
                replan_recommended=False,
                safety_status_reason=f"HARD REJECTION: Road boundary breach invariant violated (margin={min_ditch_margin:0.2f}m)"
            )

        # -------------------------------------------------------------------
        # TIER 3: DYNAMIC INCURSION / EMERGENCY REPLAN (1.0s <= TTC < 2.0s)
        # -------------------------------------------------------------------
        if unexpected_incursion or risk.min_ttc_seconds < self.replan_ttc_threshold_s:
            self.rejected_count += 1
            self.emergency_replan_count += 1
            self.last_gate_status = "REJECTED_REPLAN"
            self.last_rejection_reason = "REJECTED_UNEXPECTED_INCURSION" if unexpected_incursion else "REJECTED_REPLAN_TTC"

            for wp in filtered_traj.waypoints:
                wp.speed_mps = min(wp.speed_mps, max(1.5, ego_state.twist.speed_mps * 0.5))
                wp.acceleration_mps2 = -2.5

            if unexpected_incursion:
                status_reason = f"UNEXPECTED_OBSTACLE_INCURSION ({incursion_id}) -> EMERGENCY_REPLAN"
            else:
                status_reason = f"EMERGENCY_REPLAN_TRIGGERED (TTC={risk.min_ttc_seconds:0.2f}s)"

            return SafeTrajectory(
                timestamp=ego_state.timestamp,
                source_trajectory_id=planned.trajectory_id,
                waypoints=filtered_traj.waypoints,
                safety_action=SafetyAction.EMERGENCY_REPLAN,
                is_emergency_stop=False,
                is_rejected=True,
                rejection_reason=self.last_rejection_reason,
                supervisor_gate_status="REJECTED_REPLAN",
                barrier_margin_m=min(closest_dist, barrier_margin),
                min_ttc_seconds=risk.min_ttc_seconds,
                replan_recommended=True,
                safety_status_reason=status_reason
            )

        # -------------------------------------------------------------------
        # TIER 4: CAUTION SLOWDOWN (2.0s <= TTC <= 4.0s) OR CBF BARRIER MODULATION
        # -------------------------------------------------------------------
        if risk.min_ttc_seconds <= self.slowdown_ttc_threshold_s or barrier_violated:
            self.passed_count += 1
            self.slowdown_count += 1
            self.last_gate_status = "PASSED_WITH_SLOWDOWN"
            self.last_rejection_reason = None

            if barrier_violated:
                safety_action = SafetyAction.CONTROL_BARRIER_OVERRIDE
                safety_status_reason = f"CONTROL_BARRIER_OVERRIDE (Margin={barrier_margin:0.2f}m)"
            else:
                safety_action = SafetyAction.ADAPTIVE_CRUISE_SLOWDOWN
                safety_status_reason = f"CAUTION_SLOWDOWN (TTC={risk.min_ttc_seconds:0.2f}s)"
                target_v_safe = max(1.5, risk.min_ttc_seconds * 1.35)
                for wp in filtered_traj.waypoints:
                    if wp.speed_mps > target_v_safe:
                        wp.speed_mps = round(target_v_safe, 2)
                        wp.acceleration_mps2 = min(wp.acceleration_mps2, -1.8)

            return SafeTrajectory(
                timestamp=ego_state.timestamp,
                source_trajectory_id=planned.trajectory_id,
                waypoints=filtered_traj.waypoints,
                safety_action=safety_action,
                is_emergency_stop=False,
                is_rejected=False,
                rejection_reason=None,
                supervisor_gate_status="PASSED_WITH_SLOWDOWN",
                barrier_margin_m=min(closest_dist, barrier_margin),
                min_ttc_seconds=risk.min_ttc_seconds,
                replan_recommended=False,
                safety_status_reason=safety_status_reason
            )

        # -------------------------------------------------------------------
        # TIER 5: NOMINAL SAFE TRAJECTORY (TTC > 4.0s)
        # -------------------------------------------------------------------
        self.passed_count += 1
        self.last_gate_status = "PASSED_SAFE"
        self.last_rejection_reason = None

        safety_action = SafetyAction.NONE
        if planned.behavior_mode.value.startswith("NUDGE"):
            safety_action = SafetyAction.CORRIDOR_NUDGE
        elif planned.behavior_mode.value == "FOLLOW":
            safety_action = SafetyAction.ADAPTIVE_CRUISE_SLOWDOWN

        return SafeTrajectory(
            timestamp=ego_state.timestamp,
            source_trajectory_id=planned.trajectory_id,
            waypoints=planned.waypoints,
            safety_action=safety_action,
            is_emergency_stop=False,
            is_rejected=False,
            rejection_reason=None,
            supervisor_gate_status="PASSED_SAFE",
            barrier_margin_m=min(closest_dist, barrier_margin),
            min_ttc_seconds=risk.min_ttc_seconds,
            replan_recommended=False,
            safety_status_reason="TRAJECTORY_VERIFIED_SAFE"
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
                for traj in agent.trajectories:
                    if traj.probability >= 0.25:
                        for step_idx, ego_wp in enumerate(planned.waypoints[:10]):
                            if step_idx < len(traj.waypoints):
                                ag_wp = traj.waypoints[step_idx]
                                dist = math.hypot(ego_wp.x - ag_wp.position.x, ego_wp.y - ag_wp.position.y)
                                if dist < 1.8:
                                    return True, agent.id

        return False, None


# Backward-compatible alias
EmergencyBrakeSupervisory = SafetySupervisoryLayer
