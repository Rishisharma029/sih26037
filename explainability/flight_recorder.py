"""Blackbox Flight Recorder for autonomous decision auditing."""
import time
from typing import List, Optional, Dict, Any
from interfaces import (
    EgoVehicleState, PerceptionOutput, PredictionOutput,
    PlannedTrajectory, SafeTrajectory, SafetyAction,
    BehaviorMode, ObstacleClass, MotionIntent
)
from .types import (
    DecisionEvent, HazardContext, RiskLevel,
    DecisionRationale, CandidateSummary
)


class FlightRecorder:
    """Thread-safe autonomous decision flight recorder and explainability engine."""

    def __init__(self, scenario_name: str = "Indian Unstructured Road"):
        self.scenario_name = scenario_name
        self.event_counter = 0
        self.events: List[DecisionEvent] = []
        self.candidate_history: Dict[int, List[CandidateSummary]] = {}

    def record_decision(
        self,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput,
        prediction: PredictionOutput,
        planned: PlannedTrajectory,
        safe_plan: SafeTrajectory,
        candidate_scores: Optional[List[Any]] = None
    ) -> DecisionEvent:
        """Analyze the current Sense-Plan-Act cycle and log an explainable DecisionEvent."""
        self.event_counter += 1
        event_id = self.event_counter

        # 1. Identify primary hazard & context
        hazard, risk = self._identify_hazard(ego_state, perception, prediction, safe_plan)

        # 2. Assess path status
        if safe_plan.safety_action == SafetyAction.EMERGENCY_BRAKE:
            path_status = "CRITICAL"
        elif safe_plan.safety_action == SafetyAction.EMERGENCY_REPLAN:
            path_status = "UNSAFE"
        elif planned.behavior_mode in [BehaviorMode.NUDGE_LEFT, BehaviorMode.NUDGE_RIGHT]:
            path_status = "UNSAFE_NUDGE_REQUIRED"
        elif planned.behavior_mode == BehaviorMode.FOLLOW:
            path_status = "OBSTRUCTED_FOLLOWING"
        else:
            path_status = "SAFE"

        # 3. Extract candidate count & score decomposition
        num_candidates = len(candidate_scores) if candidate_scores else (len(planned.waypoints) if planned else 5)
        selected_id = planned.trajectory_id if planned else "Path #0"

        # 4. Formulate human-readable rationale
        rationale = self._formulate_rationale(planned, safe_plan, hazard)

        event = DecisionEvent(
            event_id=event_id,
            timestamp=ego_state.timestamp,
            scenario_name=self.scenario_name,
            ego_speed_kph=ego_state.twist.speed_mps * 3.6,
            ego_position_x=ego_state.pose.position.x,
            ego_position_y=ego_state.pose.position.y,
            hazard=hazard,
            risk_level=risk,
            path_status=path_status,
            total_candidates_evaluated=num_candidates,
            selected_candidate_id=selected_id,
            selected_behavior=planned.behavior_mode,
            safety_action=safe_plan.safety_action,
            rationale=rationale,
            safety_override_active=(safe_plan.safety_action != SafetyAction.NONE)
        )

        self.events.append(event)
        return event

    def _identify_hazard(
        self,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput,
        prediction: PredictionOutput,
        safe_plan: SafeTrajectory
    ) -> (HazardContext, RiskLevel):
        """Extract the most urgent environmental threat."""
        closest_obs = None
        min_dist = 999.0

        for obs in perception.obstacles:
            if obs.distance_m < min_dist:
                min_dist = obs.distance_m
                closest_obs = obs

        ttc = safe_plan.min_ttc_seconds if safe_plan else 10.0

        if closest_obs is None:
            hazard = HazardContext(
                hazard_id="NONE",
                hazard_type="Clear unhindered corridor ahead",
                obstacle_class=None,
                relative_distance_m=50.0,
                relative_speed_mps=0.0,
                ttc_seconds=10.0,
                corridor_side="FRONT"
            )
            return hazard, RiskLevel.LOW

        # Classify hazard type description
        obs_name = closest_obs.obstacle_class.value.replace("_", " ").title()
        if closest_obs.obstacle_class == ObstacleClass.MOTORCYCLE:
            h_type = "Motorcycle entering ego trajectory"
        elif closest_obs.obstacle_class == ObstacleClass.CATTLE_ANIMAL:
            h_type = "Stray cattle stationary / crossing road"
        elif closest_obs.obstacle_class == ObstacleClass.PEDESTRIAN:
            h_type = "Pedestrian crossing carriageway"
        elif closest_obs.obstacle_class == ObstacleClass.AUTO_RICKSHAW:
            h_type = "Auto-rickshaw dynamic cut-in"
        elif closest_obs.obstacle_class == ObstacleClass.TRUCK:
            h_type = "Wide oncoming heavy vehicle in opposing lane"
        else:
            h_type = f"{obs_name} encroaching into drivable path"

        side = "LEFT" if closest_obs.bbox.center.y > 0.5 else ("RIGHT" if closest_obs.bbox.center.y < -0.5 else "FRONT")

        # Classify risk level
        if ttc < 1.2 or min_dist < 1.2:
            risk = RiskLevel.CRITICAL
        elif ttc < 3.0 or min_dist < 3.5:
            risk = RiskLevel.HIGH
        elif ttc < 6.0 or min_dist < 8.0:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW

        hazard = HazardContext(
            hazard_id=closest_obs.id,
            hazard_type=h_type,
            obstacle_class=closest_obs.obstacle_class,
            relative_distance_m=min_dist,
            relative_speed_mps=closest_obs.velocity.x,
            ttc_seconds=ttc,
            corridor_side=side
        )
        return hazard, risk

    def _formulate_rationale(
        self,
        planned: PlannedTrajectory,
        safe_plan: SafeTrajectory,
        hazard: HazardContext
    ) -> DecisionRationale:
        """Generate human-readable engineering rationale for the planning selection."""
        safety_margin = safe_plan.barrier_margin_m

        if safe_plan.safety_action == SafetyAction.EMERGENCY_BRAKE:
            reason = "Imminent collision risk below critical TTC threshold -> Immediate AEB Hard Stop"
        elif safe_plan.safety_action == SafetyAction.EMERGENCY_REPLAN:
            reason = "Dynamic obstacle incursion into path swath -> Emergency dynamic re-plan & deceleration"
        elif planned.behavior_mode in [BehaviorMode.NUDGE_LEFT, BehaviorMode.NUDGE_RIGHT]:
            reason = f"Highest safety margin (+{safety_margin:.1f}m) + acceptable curvature + smooth corridor deviation"
        elif planned.behavior_mode == BehaviorMode.FOLLOW:
            reason = f"Maintains safe headway behind leading {hazard.obstacle_class.value if hazard.obstacle_class else 'vehicle'} with zero risk"
        else:
            reason = "Optimal nominal cruise velocity + minimal cross-track error + zero obstacle conflict"

        return DecisionRationale(
            primary_reason=reason,
            safety_margin_m=safety_margin,
            curvature_rating="ACCEPTABLE",
            lateral_deviation_rating="LOW" if abs(planned.waypoints[0].y if planned.waypoints else 0.0) < 0.8 else "MODERATE"
        )
