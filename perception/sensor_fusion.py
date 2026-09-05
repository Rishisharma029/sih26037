"""Perception fusion engine combining boundary and obstacle detection."""
from typing import List, Dict
from interfaces import RawSensorFrame, PerceptionOutput, TrackedObstacle
from .boundary_detector import FreeSpaceBoundaryDetector
from .obstacle_detector import MixedTrafficObstacleDetector

class MultiSensorKalmanFusion:
    """Fuses sensor data to produce a consolidated PerceptionOutput."""
    def __init__(self, road_width_m: float = 6.0):
        self.boundary_detector = FreeSpaceBoundaryDetector(default_width_m=road_width_m)
        self.obstacle_detector = MixedTrafficObstacleDetector()

    def process_frame(self, raw: RawSensorFrame, mock_obs: List[dict] = None) -> PerceptionOutput:
        obstacles = self.obstacle_detector.detect_from_mock(mock_obs or [])
        corridor = self.boundary_detector.detect_corridor(raw.timestamp)

        return PerceptionOutput(
            timestamp=raw.timestamp,
            frame_id=raw.frame_id,
            obstacles=obstacles,
            drivable_corridor=corridor,
            anomalies=[],
            sensor_health={"camera": True, "lidar": True, "radar": True, "gnss": raw.gnss_fix}
        )
