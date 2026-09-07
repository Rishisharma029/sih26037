"""
Comprehensive Unit Tests for Real Perception (IDD Detection, 3D Bounding Box Regression, MOT Tracking & Velocity Estimation).
"""
import pytest
import torch
import math
from interfaces import ObstacleClass, Point3D, Vector3D, EgoVehicleState, Pose3D, Twist3D
from simulation.environment import RoadEnvironment, VillageRoadGeometry
from simulation.actors import SimulationActor
from perception.idd_detector import IndianTrafficDetector, DetectedObject2D
from perception.tracker import MultiObjectTracker
from perception.perception_pipeline import UnifiedPerceptionPipeline, PerceptionMode

def test_idd_detector_forward_pass():
    """Verify IndianTrafficDetector processes image tensors and outputs structured detections."""
    detector = IndianTrafficDetector(confidence_threshold=0.50)
    # Synthetic RGB image batch (3, 256, 256)
    img = torch.rand((3, 256, 256), dtype=torch.float32)
    dets = detector.detect_from_image_tensor(img)

    # Must return a list of DetectedObject2D (even if empty or detected)
    assert isinstance(dets, list)
    if len(dets) > 0:
        d = dets[0]
        assert isinstance(d.class_name, ObstacleClass)
        assert 0.0 <= d.confidence <= 1.0
        assert d.estimated_depth_m > 0.0
        bbox_3d = d.to_3d_bbox()
        assert bbox_3d.center.x > 0.0
        assert bbox_3d.size.x > 0.0

def test_mot_tracker_persistence():
    """Verify MultiObjectTracker maintains persistent track IDs across frames."""
    tracker = MultiObjectTracker(max_missed_steps=3, match_distance_threshold_m=3.0)

    # Frame 1: Tractor detected at x=30m, y=0.5m
    det_f1 = DetectedObject2D(
        class_name=ObstacleClass.TRUCK,
        confidence=0.92,
        bbox_2d=(100, 100, 200, 200),
        estimated_depth_m=30.0,
        lateral_offset_m=0.5,
        estimated_size_m=Vector3D(x=4.2, y=2.0, z=2.2)
    )
    obs_f1 = tracker.update([det_f1], timestamp=0.0, dt=0.05)
    assert len(obs_f1) == 1
    t_id_1 = obs_f1[0].id
    assert "trk" in t_id_1

    # Frame 2: Tractor advances to x=28.5m, y=0.5m (closing in)
    det_f2 = DetectedObject2D(
        class_name=ObstacleClass.TRUCK,
        confidence=0.94,
        bbox_2d=(100, 100, 200, 200),
        estimated_depth_m=28.5,
        lateral_offset_m=0.5,
        estimated_size_m=Vector3D(x=4.2, y=2.0, z=2.2)
    )
    obs_f2 = tracker.update([det_f2], timestamp=0.5, dt=0.5)
    assert len(obs_f2) == 1
    # Track ID must remain persistent across frames
    assert obs_f2[0].id == t_id_1
    assert obs_f2[0].bbox.center.x == pytest.approx(28.95, abs=0.5)

def test_mot_tracker_velocity_estimation():
    """Verify Kalman filter accurately estimates velocity from successive detections."""
    tracker = MultiObjectTracker(max_missed_steps=3)

    # Object moving southwards (y decreases by 0.5m each 0.5s -> vy = -1.0 m/s)
    for step in range(5):
        t = step * 0.5
        y_pos = 2.0 - step * 0.5
        det = DetectedObject2D(
            class_name=ObstacleClass.PEDESTRIAN,
            confidence=0.95,
            bbox_2d=(100, 100, 150, 200),
            estimated_depth_m=15.0,
            lateral_offset_m=y_pos,
            estimated_size_m=Vector3D(x=0.5, y=0.5, z=1.7)
        )
        obs = tracker.update([det], timestamp=t, dt=0.5)

    assert len(obs) == 1
    ped = obs[0]
    # Velocity vy should converge towards -1.0 m/s
    assert ped.velocity.y < -0.3
    assert not ped.is_static

def test_unified_perception_pipeline_e2e():
    """Verify end-to-end perception output generation with full contract compliance."""
    pipeline = UnifiedPerceptionPipeline(mode=PerceptionMode.NEURAL_IDD)
    geom = VillageRoadGeometry(length_m=200.0)
    tractor = SimulationActor(
        id="oncoming_tractor",
        obstacle_class=ObstacleClass.TRUCK,
        x=35.0, y=0.8, speed_mps=-3.5, yaw_rad=math.pi,
        length_m=4.2, width_m=2.0
    )
    ego = EgoVehicleState(
        timestamp=1.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
        twist=Twist3D(speed_mps=6.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )

    perc = pipeline.process_frame(
        timestamp=1.0,
        current_s=0.0,
        geometry=geom,
        actors=[tractor],
        ego_state=ego,
        anomalies=[],
        dt=0.05
    )

    assert perc.frame_id == 1
    assert len(perc.obstacles) >= 1
    obs = perc.obstacles[0]
    assert obs.obstacle_class == ObstacleClass.TRUCK
    assert obs.bbox.size.x == 4.2
    assert obs.distance_m > 0.0
    assert perc.drivable_corridor.average_width_m > 3.0
    assert perc.sensor_health["camera"] is True

def test_perception_mode_swappability():
    """Verify swappability between neural IDD perception and simulated ground truth."""
    pipe_neural = UnifiedPerceptionPipeline(mode=PerceptionMode.NEURAL_IDD)
    pipe_sim = UnifiedPerceptionPipeline(mode=PerceptionMode.GROUND_TRUTH)

    geom = VillageRoadGeometry(length_m=200.0)
    auto = SimulationActor(
        id="auto_rickshaw",
        obstacle_class=ObstacleClass.AUTO_RICKSHAW,
        x=20.0, y=1.2, speed_mps=0.0, yaw_rad=0.0,
        length_m=2.6, width_m=1.3, is_static=True
    )
    ego = EgoVehicleState(
        timestamp=0.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
        twist=Twist3D(speed_mps=5.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )

    out_neural = pipe_neural.process_frame(0.0, 0.0, geom, [auto], ego)
    out_sim = pipe_sim.process_frame(0.0, 0.0, geom, [auto], ego)

    assert len(out_neural.obstacles) >= 1
    assert len(out_sim.obstacles) == 1
    assert out_neural.obstacles[0].obstacle_class == ObstacleClass.AUTO_RICKSHAW
    assert out_sim.obstacles[0].obstacle_class == ObstacleClass.AUTO_RICKSHAW
