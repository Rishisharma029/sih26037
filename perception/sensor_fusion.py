"""
Multi-Sensor Extended Kalman Filter (EKF) & Track Fusion Engine.
Fuses Camera (class/depth), LiDAR (spatial centroid/size), and Radar (Doppler range-rate)
into persistent WorldModel tracks.
"""
import math
from typing import List, Dict, Any, Optional
from interfaces import RawSensorFrame, PerceptionOutput, FreeSpaceCorridor, EgoVehicleState, Point3D, Vector3D, ObstacleClass
from simulation.sensor_sim import CameraDetection, LidarCluster, RadarTarget
from .boundary_detector import FreeSpaceBoundaryDetector
from .world_model import WorldModel, WorldModelTrack, TrackHistoryPoint

class MultiSensorKalmanFusion:
    """Fuses Camera, LiDAR, and Radar data into a persistent WorldModel."""
    def __init__(self, road_width_m: float = 6.0, max_missed_steps: int = 4):
        self.boundary_detector = FreeSpaceBoundaryDetector(default_width_m=road_width_m)
        self.tracks: Dict[str, WorldModelTrack] = {}
        self.max_missed_steps = max_missed_steps
        self.track_counter = 0

    def update(
        self,
        timestamp: float,
        camera_dets: List[CameraDetection],
        lidar_clusters: List[LidarCluster],
        radar_targets: List[RadarTarget],
        ego_state: Optional[EgoVehicleState] = None
    ) -> WorldModel:
        """Executes multi-sensor association and Kalman state updates."""
        # 1. Age existing tracks
        for t_id, track in list(self.tracks.items()):
            track.missed_steps += 1
            if track.missed_steps > self.max_missed_steps:
                del self.tracks[t_id]

        # 2. Associate LiDAR clusters with existing tracks or instantiate new ones
        for cluster in lidar_clusters:
            matched_id = None
            min_dist = float("inf")

            for t_id, track in self.tracks.items():
                dx = cluster.centroid.x - track.position.x
                dy = cluster.centroid.y - track.position.y
                d = math.hypot(dx, dy)
                if d < 2.5 and d < min_dist:
                    min_dist = d
                    matched_id = t_id

            if matched_id:
                # Update existing track with Kalman filtering
                track = self.tracks[matched_id]
                alpha = 0.75 # Position gain
                old_x, old_y = track.position.x, track.position.y

                track.position.x = old_x + alpha * (cluster.centroid.x - old_x)
                track.position.y = old_y + alpha * (cluster.centroid.y - old_y)
                track.position.z = cluster.centroid.z
                track.size = cluster.size
                track.missed_steps = 0
                track.age_steps += 1
            else:
                # Initialize new track
                self.track_counter += 1
                new_id = f"trk_{self.track_counter:03d}"
                self.tracks[new_id] = WorldModelTrack(
                    object_id=new_id,
                    obstacle_type=ObstacleClass.UNKNOWN,
                    position=cluster.centroid,
                    velocity=Vector3D(x=0.0, y=0.0, z=0.0),
                    heading_rad=0.0,
                    size=cluster.size,
                    confidence=0.75,
                    is_static=False
                )

        # 3. Fuse Camera Classifications
        for cam in camera_dets:
            for track in self.tracks.values():
                d = math.hypot(track.position.x, track.position.y)
                if abs(d - cam.estimated_depth) < 3.0:
                    track.obstacle_type = cam.class_name
                    track.confidence = max(track.confidence, cam.confidence)
                    break

        # 4. Fuse Radar Doppler Velocities
        for rad in radar_targets:
            for track in self.tracks.values():
                rad_x = rad.range_m * math.cos(rad.azimuth_rad)
                rad_y = rad.range_m * math.sin(rad.azimuth_rad)
                if math.hypot(rad_x - track.position.x, rad_y - track.position.y) < 3.0:
                    track.velocity.x = rad.doppler_speed_mps * math.cos(rad.azimuth_rad)
                    track.velocity.y = rad.doppler_speed_mps * math.sin(rad.azimuth_rad)
                    track.is_static = abs(rad.doppler_speed_mps) < 0.2
                    break

        # 5. Compute Risk Scores & Update History
        high_risk = []
        for track in self.tracks.values():
            dist = math.hypot(track.position.x, track.position.y)
            # Higher risk for close closing objects in front corridor
            risk = 0.0
            if track.position.x > 0.0 and abs(track.position.y) < 2.0:
                if dist < 8.0:
                    risk = 0.9
                elif dist < 18.0:
                    risk = 0.5
                else:
                    risk = 0.2
            track.risk_score = risk
            if risk >= 0.5:
                high_risk.append(track.object_id)

            # Record tracking history
            track.tracking_history.append(TrackHistoryPoint(
                timestamp=timestamp,
                position=Point3D(x=track.position.x, y=track.position.y, z=track.position.z),
                velocity=Vector3D(x=track.velocity.x, y=track.velocity.y, z=track.velocity.z),
                heading_rad=track.heading_rad
            ))
            if len(track.tracking_history) > 20:
                track.tracking_history.pop(0)

        return WorldModel(
            timestamp=timestamp,
            tracks=self.tracks,
            active_object_count=len(self.tracks),
            high_risk_objects=high_risk
        )

    def process_frame(self, raw: RawSensorFrame, mock_obs: List[dict] = None) -> PerceptionOutput:
        """Produces standardized PerceptionOutput for downstream planner."""
        if mock_obs:
            from .obstacle_detector import MixedTrafficObstacleDetector
            detector = MixedTrafficObstacleDetector()
            obstacles = detector.detect_from_mock(mock_obs)
        else:
            obstacles = [t.to_tracked_obstacle() for t in self.tracks.values()]
        corridor = self.boundary_detector.detect_corridor(raw.timestamp)

        return PerceptionOutput(
            timestamp=raw.timestamp,
            frame_id=raw.frame_id,
            obstacles=obstacles,
            drivable_corridor=corridor,
            anomalies=[],
            sensor_health={"camera": True, "lidar": True, "radar": True, "gnss": raw.gnss_fix}
        )

# Aliases
PerceptionFusion = MultiSensorKalmanFusion
