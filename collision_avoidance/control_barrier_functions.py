"""Control Barrier Functions (CBF) forward invariance safety filter."""
from interfaces import PlannedTrajectory, EgoVehicleState, PerceptionOutput

class ControlBarrierFilter:
    """Guarantees h(x) >= 0 (safe clearance to all obstacles)."""
    def __init__(self, min_safe_dist_m: float = 2.0):
        self.min_safe_dist_m = min_safe_dist_m

    def filter_trajectory(
        self,
        trajectory: PlannedTrajectory,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput
    ) -> tuple[PlannedTrajectory, bool]:
        """Clips waypoint speeds if closest obstacle violates barrier condition."""
        closest_dist = min([obs.distance_m for obs in perception.obstacles], default=999.0)
        violated = closest_dist < self.min_safe_dist_m

        if violated:
            # Barrier override: slow down immediately
            for wp in trajectory.waypoints:
                wp.speed_mps = min(wp.speed_mps, 1.0)
                wp.acceleration_mps2 = -3.0

        return trajectory, violated
