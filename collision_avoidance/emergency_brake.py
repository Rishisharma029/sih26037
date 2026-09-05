"""Supervisory Emergency Braking & Safety Envelope coordinator."""
from interfaces import (
    PlannedTrajectory, SafeTrajectory, SafetyAction,
    EgoVehicleState, PerceptionOutput, PredictionOutput
)
from .ttc_calculator import TTCCalculator
from .control_barrier_functions import ControlBarrierFilter

class EmergencyBrakeSupervisory:
    """Supervises planned trajectory and outputs a verified SafeTrajectory."""
    def __init__(self, aeb_ttc_threshold_s: float = 0.85):
        self.aeb_ttc_threshold_s = aeb_ttc_threshold_s
        self.ttc_calc = TTCCalculator()
        self.cbf_filter = ControlBarrierFilter()

    def supervise(
        self,
        planned: PlannedTrajectory,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput,
        prediction: PredictionOutput
    ) -> SafeTrajectory:
        risk = self.ttc_calc.compute_ttc(ego_state, perception.obstacles)
        filtered_traj, barrier_violated = self.cbf_filter.filter_trajectory(planned, ego_state, perception)

        safety_action = SafetyAction.NONE
        is_e_stop = False

        if risk.min_ttc_seconds < self.aeb_ttc_threshold_s:
            safety_action = SafetyAction.EMERGENCY_BRAKE
            is_e_stop = True
            for wp in filtered_traj.waypoints:
                wp.speed_mps = 0.0
                wp.acceleration_mps2 = -6.0
        elif barrier_violated:
            safety_action = SafetyAction.CONTROL_BARRIER_OVERRIDE
        elif planned.behavior_mode.value.startswith("NUDGE"):
            safety_action = SafetyAction.CORRIDOR_NUDGE

        closest_dist = min([obs.distance_m for obs in perception.obstacles], default=999.0)

        return SafeTrajectory(
            timestamp=ego_state.timestamp,
            source_trajectory_id=planned.trajectory_id,
            waypoints=filtered_traj.waypoints,
            safety_action=safety_action,
            is_emergency_stop=is_e_stop,
            barrier_margin_m=closest_dist,
            min_ttc_seconds=risk.min_ttc_seconds
        )
