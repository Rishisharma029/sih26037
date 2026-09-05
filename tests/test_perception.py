"""Unit tests for perception subsystem."""
from perception.boundary_detector import FreeSpaceBoundaryDetector
from perception.obstacle_detector import MixedTrafficObstacleDetector
from perception.sensor_fusion import MultiSensorKalmanFusion
from interfaces import RawSensorFrame

def test_boundary_detection():
    detector = FreeSpaceBoundaryDetector(default_width_m=7.0)
    corridor = detector.detect_corridor(timestamp=0.0, lookahead_m=20.0)
    assert corridor.average_width_m == 7.0
    assert len(corridor.boundary_points) == 5

def test_fusion_pipeline():
    fusion = MultiSensorKalmanFusion(road_width_m=6.0)
    raw = RawSensorFrame(timestamp=0.1, frame_id=1)
    mock_obs = [{"id": "rickshaw_1", "class": "AUTO_RICKSHAW", "x": 12.0, "y": 1.0, "vx": 5.0, "vy": 0.0}]
    output = fusion.process_frame(raw, mock_obs)
    assert len(output.obstacles) == 1
    assert output.obstacles[0].obstacle_class.value == "AUTO_RICKSHAW"
