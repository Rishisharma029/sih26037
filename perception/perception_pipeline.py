"""
Unified Perception Subsystem Pipeline.
Architectural single entry point for camera frames, neural IDD detection,
boundary detection, multi-object tracking, and perception output generation.
"""
from enum import Enum
import math
from typing import List, Dict, Any, Optional
import torch

from interfaces import (
    PerceptionOutput, FreeSpaceCorridor, TrackedObstacle,
    RoadAnomaly, RawSensorFrame, EgoVehicleState, ObstacleClass
)
from .boundary_detector import FreeSpaceBoundaryDetector
from .idd_detector import IndianTrafficDetector, DetectedObject2D
from .tracker import MultiObjectTracker

class PerceptionMode(str, Enum):
    NEURAL_IDD = "NEURAL_IDD"         # Neural IDD detector + MOT Kalman tracking
    FUSED_TRACKER = "FUSED_TRACKER"   # Camera + LiDAR + Radar sensor fusion
    GROUND_TRUTH = "GROUND_TRUTH"     # Direct simulation ground truth (fallback)

class UnifiedPerceptionPipeline:
    """Standardized Autonomous Perception Engine for Unstructured Indian Roads."""

    def __init__(self, mode: PerceptionMode = PerceptionMode.NEURAL_IDD, default_road_width_m: float = 4.3):
        self.mode = mode
        self.detector = IndianTrafficDetector()
        self.tracker = MultiObjectTracker()
        self.boundary_detector = FreeSpaceBoundaryDetector(default_width_m=default_road_width_m)
        self.frame_count = 0

    def process_frame(
        self,
        timestamp: float,
        current_s: float,
        geometry: Any,
        actors: List[Any],
        ego_state: EgoVehicleState,
        anomalies: Optional[List[RoadAnomaly]] = None,
        camera_image: Optional[torch.Tensor] = None,
        raw_sensor: Optional[RawSensorFrame] = None,
        dt: float = 0.05
    ) -> PerceptionOutput:
        """Executes perception pipeline: Detection -> 3D Bounding Box -> MOT Track ID -> Velocity -> Corridor."""
        self.frame_count += 1

        # 1. Drivable Free-Space Corridor Detection
        corridor = self.boundary_detector.detect_corridor(
            timestamp=timestamp,
            lookahead_m=45.0,
            step_m=3.0,
            current_s=current_s,
            geometry=geometry
        )

        # 2. Object Perception via IDD Engine
        tracked_obstacles: List[TrackedObstacle] = []

        if self.mode == PerceptionMode.NEURAL_IDD:
            if camera_image is not None and isinstance(camera_image, torch.Tensor):
                # Neural forward pass
                raw_dets = self.detector.detect_from_image_tensor(camera_image)
            else:
                # Simulated camera detection bridging
                from simulation.sensor_sim import SyntheticSensorSuite
                from coordinates import world_to_ego_2d
                sim_suite = SyntheticSensorSuite(geometry if hasattr(geometry, "actors") else type("Env", (), {"actors": actors})())
                cam_dets, _, _ = sim_suite.capture_sensor_measurements(ego_state)
                raw_dets = self.detector.detect_from_synthetic_camera(cam_dets, actors, ego_state)

            # 3. Multi-Object Tracking & Velocity Estimation
            tracked_obstacles = self.tracker.update(raw_dets, timestamp=timestamp, dt=dt)

        else:
            # Fallback Ground-Truth Perception
            from coordinates import transform_actor_to_ego_tracked_obstacle
            for a in actors:
                obs = transform_actor_to_ego_tracked_obstacle(
                    actor_id=a.id,
                    obstacle_class=a.obstacle_class,
                    x_world=a.x,
                    y_world=a.y,
                    z_world=a.z,
                    length_m=a.length_m,
                    width_m=a.width_m,
                    height_m=a.height_m,
                    yaw_world_rad=a.yaw_rad,
                    speed_mps=a.speed_mps,
                    is_static=a.is_static,
                    ego_pose=ego_state.pose,
                    ego_twist=ego_state.twist,
                    confidence=0.98
                )
                tracked_obstacles.append(obs)

        return PerceptionOutput(
            timestamp=timestamp,
            frame_id=self.frame_count,
            obstacles=tracked_obstacles,
            drivable_corridor=corridor,
            anomalies=anomalies if anomalies is not None else [],
            sensor_health={"camera": True, "lidar": True, "radar": True, "gnss": True}
        )
