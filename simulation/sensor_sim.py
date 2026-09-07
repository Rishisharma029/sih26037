"""
Synthetic Multi-Sensor Simulator generating realistic Camera, LiDAR, and Radar data.
Includes sensor noise, field-of-view (FOV) clipping, and Doppler range-rate physics.
"""
import math
import random
from typing import List, Dict, Any, Tuple
from interfaces import RawSensorFrame, EgoVehicleState, Point3D, Vector3D, ObstacleClass
from coordinates import world_to_ego_2d, world_to_ego_velocity
from .environment import RoadEnvironment
from .actors import SimulationActor

class CameraDetection:
    def __init__(self, class_name: ObstacleClass, bbox_2d: tuple, confidence: float, estimated_depth: float):
        self.class_name = class_name
        self.bbox_2d = bbox_2d # (xmin, ymin, xmax, ymax)
        self.confidence = confidence
        self.estimated_depth = estimated_depth

class LidarCluster:
    def __init__(self, centroid: Point3D, size: Vector3D, num_points: int):
        self.centroid = centroid
        self.size = size
        self.num_points = num_points

class RadarTarget:
    def __init__(self, target_id: str, range_m: float, azimuth_rad: float, doppler_speed_mps: float, rcs_db: float):
        self.target_id = target_id
        self.range_m = range_m
        self.azimuth_rad = azimuth_rad
        self.doppler_speed_mps = doppler_speed_mps
        self.rcs_db = rcs_db

class SyntheticSensorSuite:
    """Simulates realistic Camera, 3D LiDAR, and 4D Radar feeds."""
    def __init__(self, env: RoadEnvironment):
        self.env = env
        self.frame_id = 0

        # Sensor FOV specifications
        self.camera_fov_deg = 80.0
        self.lidar_range_m = 90.0
        self.radar_range_m = 120.0

    def capture(self, ego_state: EgoVehicleState) -> RawSensorFrame:
        self.frame_id += 1
        return RawSensorFrame(
            timestamp=ego_state.timestamp,
            frame_id=self.frame_id,
            lidar_points_count=len(self.env.actors) * 55 + 180,
            camera_detections_count=len(self.env.actors),
            radar_targets_count=len([a for a in self.env.actors if not a.is_static]),
            gnss_fix=True,
            imu_angular_velocity=ego_state.twist.angular,
            imu_linear_acceleration=ego_state.acceleration
        )

    def capture_sensor_measurements(self, ego_state: EgoVehicleState) -> tuple[List[CameraDetection], List[LidarCluster], List[RadarTarget]]:
        """Generates raw measurement packets from Camera, LiDAR, and Radar."""
        ego_x = ego_state.pose.position.x
        ego_y = ego_state.pose.position.y
        ego_yaw = ego_state.pose.heading_rad
        ego_speed = ego_state.twist.speed_mps

        camera_detections: List[CameraDetection] = []
        lidar_clusters: List[LidarCluster] = []
        radar_targets: List[RadarTarget] = []

        for actor in self.env.actors:
            # Transform to vehicle body frame using canonical transformation function
            local_x, local_y = world_to_ego_2d(actor.x, actor.y, ego_x, ego_y, ego_yaw)
            dist = math.hypot(local_x, local_y)

            if local_x <= 0.0 or dist > self.radar_range_m:
                continue

            azimuth = math.atan2(local_y, local_x)

            # 1. Camera Detection (Visual FOV & occlusion)
            if dist < 65.0 and abs(math.degrees(azimuth)) < (self.camera_fov_deg / 2.0):
                noisy_depth = dist + random.gauss(0.0, 0.2)
                camera_detections.append(CameraDetection(
                    class_name=actor.obstacle_class,
                    bbox_2d=(100, 100, 200, 200),
                    confidence=random.uniform(0.88, 0.98),
                    estimated_depth=noisy_depth
                ))

            # 2. LiDAR Cluster (Precise geometry + point reflections)
            if dist < self.lidar_range_m:
                noisy_centroid = Point3D(
                    x=local_x + random.gauss(0.0, 0.05),
                    y=local_y + random.gauss(0.0, 0.05),
                    z=actor.z + actor.height_m / 2.0
                )
                lidar_clusters.append(LidarCluster(
                    centroid=noisy_centroid,
                    size=Vector3D(x=actor.length_m, y=actor.width_m, z=actor.height_m),
                    num_points=int(450 / max(1.0, dist))
                ))

            # 3. Radar Target (Doppler velocity measurement)
            if dist < self.radar_range_m:
                # Relative velocity along line of sight in vehicle frame
                rel_vx, rel_vy = world_to_ego_velocity(
                    vx_world=actor.speed_mps * math.cos(actor.yaw_rad),
                    vy_world=actor.speed_mps * math.sin(actor.yaw_rad),
                    ego_heading_rad=ego_yaw,
                    ego_vx_world=ego_speed * math.cos(ego_yaw),
                    ego_vy_world=ego_speed * math.sin(ego_yaw)
                )
                doppler_radial_v = (local_x * rel_vx + local_y * rel_vy) / max(0.1, dist)

                radar_targets.append(RadarTarget(
                    target_id=f"rad_{actor.id}",
                    range_m=dist + random.gauss(0.0, 0.15),
                    azimuth_rad=azimuth + random.gauss(0.0, 0.02),
                    doppler_speed_mps=doppler_radial_v + random.gauss(0.0, 0.1),
                    rcs_db=15.0 if actor.obstacle_class == ObstacleClass.TRUCK else 8.0
                ))

        return camera_detections, lidar_clusters, radar_targets
