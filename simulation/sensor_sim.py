"""Synthetic multi-sensor generator (LiDAR, Radar, Camera, GNSS/IMU)."""
import math
from interfaces import RawSensorFrame, EgoVehicleState, Vector3D
from .environment import RoadEnvironment

class SyntheticSensorSuite:
    """Generates raw sensor measurements from ground truth environment."""
    def __init__(self, env: RoadEnvironment):
        self.env = env
        self.frame_id = 0

    def capture(self, ego_state: EgoVehicleState) -> RawSensorFrame:
        self.frame_id += 1
        return RawSensorFrame(
            timestamp=ego_state.timestamp,
            frame_id=self.frame_id,
            lidar_points_count=len(self.env.obstacles) * 45 + 120,
            camera_detections_count=len(self.env.obstacles),
            radar_targets_count=len([o for o in self.env.obstacles if not o.is_static]),
            gnss_fix=True,
            imu_angular_velocity=ego_state.twist.angular,
            imu_linear_acceleration=ego_state.acceleration
        )
