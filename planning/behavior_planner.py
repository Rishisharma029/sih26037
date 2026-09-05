"""Behavioral finite state machine for Indian road maneuvers."""
from interfaces import BehaviorMode, PerceptionOutput, PredictionOutput, EgoVehicleState

class BehaviorPlanner:
    """Decides operational tactical state: CRUISE, NUDGE, FOLLOW, YIELD, EMERGENCY."""
    def __init__(self, safe_follow_dist_m: float = 12.0):
        self.safe_follow_dist_m = safe_follow_dist_m
        self.current_mode = BehaviorMode.CRUISE

    def decide_behavior(
        self,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput,
        prediction: PredictionOutput
    ) -> BehaviorMode:
        # Check for immediate critical obstacles ahead
        lead_obstacles = [
            obs for obs in perception.obstacles
            if 0.0 < obs.bbox.center.x < 30.0 and abs(obs.bbox.center.y) < 1.8
        ]

        if not lead_obstacles:
            self.current_mode = BehaviorMode.CRUISE
            return self.current_mode

        closest = min(lead_obstacles, key=lambda o: o.bbox.center.x)

        if closest.distance_m < 4.0:
            self.current_mode = BehaviorMode.EMERGENCY_STOP
        elif closest.is_static:
            # Check nudge room on left or right
            if closest.bbox.center.y > 0.0:
                self.current_mode = BehaviorMode.NUDGE_RIGHT
            else:
                self.current_mode = BehaviorMode.NUDGE_LEFT
        elif closest.distance_m < self.safe_follow_dist_m:
            self.current_mode = BehaviorMode.FOLLOW
        else:
            self.current_mode = BehaviorMode.CRUISE

        return self.current_mode
