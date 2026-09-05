"""
Closed-loop discrete-time simulator connecting road environment, actors, physics, and sensors.
"""
from interfaces import (
    EgoVehicleState, ControlCommand, Pose3D, Point3D,
    Twist3D, Vector3D, RawSensorFrame, PerceptionOutput
)
from .vehicle_model import KinematicBicycleModel
from .environment import RoadEnvironment
from .actors import SimulationActor

class ClosedLoopSimulator:
    """Main simulation coordinator managing environment, ego vehicle, and actor stepping."""
    def __init__(self, env: RoadEnvironment = None, dt: float = 0.05, initial_pose: Pose3D = None):
        self.env = env or RoadEnvironment()
        self.dt = dt
        self.vehicle_model = KinematicBicycleModel()
        init_p = initial_pose or Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0)
        self.state = EgoVehicleState(
            timestamp=0.0,
            pose=init_p,
            twist=Twist3D(speed_mps=0.0),
            acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
        )
        self.frame_counter = 0

    def step(self, command: ControlCommand) -> tuple[EgoVehicleState, RawSensorFrame]:
        """Step ego vehicle physics, step all other scene actors, and synthesize sensor frame."""
        self.frame_counter += 1
        self.state = self.vehicle_model.step(self.state, command, self.dt)
        self.env.step_actors(self.dt)

        raw_sensor = RawSensorFrame(
            timestamp=self.state.timestamp,
            frame_id=self.frame_counter,
            lidar_points_count=len(self.env.actors) * 45 + 150,
            camera_detections_count=len(self.env.actors),
            radar_targets_count=len([a for a in self.env.actors if not a.is_static]),
            gnss_fix=True,
            imu_angular_velocity=self.state.twist.angular,
            imu_linear_acceleration=self.state.acceleration
        )
        return self.state, raw_sensor

    def get_ground_truth_obstacles(self) -> list:
        """Returns TrackedObstacle list from ground-truth actor states relative to ego."""
        ego_x = self.state.pose.position.x
        ego_y = self.state.pose.position.y
        return [actor.to_tracked_obstacle(ego_x, ego_y) for actor in self.env.actors]
