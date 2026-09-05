"""Comprehensive Unit & Regression Tests for Phase 5 Motion Prediction Subsystem."""
import math
import pytest
import torch

from interfaces import (
    TrackedObstacle, ObstacleClass, MotionIntent, BoundingBox3D,
    Point3D, Vector3D, PerceptionOutput, FreeSpaceCorridor
)
from prediction.intent_classifier import IntentClassifier
from prediction.kinematic_predictor import KinematicPredictor
from prediction.learned_predictor import LearnedPredictor, MultiModalTrajectoryNet
from prediction.trajectory_predictor import TrajectoryPredictor


def _create_mock_obstacle(
    id: str,
    cls: ObstacleClass,
    x: float,
    y: float,
    vx: float,
    vy: float,
    ax: float = 0.0,
    ay: float = 0.0,
    yaw: float = 0.0,
    is_static: bool = False
) -> TrackedObstacle:
    return TrackedObstacle(
        id=id,
        obstacle_class=cls,
        confidence=0.92,
        bbox=BoundingBox3D(center=Point3D(x=x, y=y, z=0.5), size=Vector3D(x=2.0, y=1.0, z=1.5), yaw_rad=yaw),
        velocity=Vector3D(x=vx, y=vy, z=0.0),
        acceleration=Vector3D(x=ax, y=ay, z=0.0),
        distance_m=math.hypot(x, y),
        is_static=is_static
    )


def test_cutin_intent_classification():
    """Verify cut-in intention detection for lateral moves towards ego center."""
    classifier = IntentClassifier()
    obs = _create_mock_obstacle("rick_1", ObstacleClass.AUTO_RICKSHAW, x=12.0, y=2.2, vx=5.0, vy=-0.85)
    intent = classifier.classify_intent(obs)
    assert intent == MotionIntent.CUTTING_IN


def test_pedestrian_crossing_intent():
    """Verify crossing intention detection for pedestrians."""
    classifier = IntentClassifier()
    ped = _create_mock_obstacle("ped_1", ObstacleClass.PEDESTRIAN, x=15.0, y=3.0, vx=0.2, vy=-1.2)
    intent = classifier.classify_intent(ped)
    assert intent == MotionIntent.CROSSING_PATH


def test_cattle_crossing_and_hesitation_intent():
    """Verify cattle wandering / crossing path."""
    classifier = IntentClassifier()
    cow = _create_mock_obstacle("cow_1", ObstacleClass.CATTLE_ANIMAL, x=18.0, y=-2.5, vx=0.4, vy=0.6)
    intent = classifier.classify_intent(cow)
    assert intent == MotionIntent.CROSSING_PATH


def test_motorcycle_multi_modal_branches():
    """Verify motorcycle generates 3 distinct modes with normalized probability and expanding sigma."""
    kin_pred = KinematicPredictor(horizon_seconds=3.0, dt=0.5)
    moto = _create_mock_obstacle("bike_1", ObstacleClass.MOTORCYCLE, x=15.0, y=1.5, vx=8.0, vy=0.0)
    modes = kin_pred.predict_modes(moto, MotionIntent.CRUISING, current_time=0.0)

    assert len(modes) == 3
    # Check probability sum
    total_p = sum(m.probability for m in modes)
    assert pytest.approx(total_p, abs=1e-2) == 1.0

    # Mode names: continuation, nudge_obstacle, cut_in_merge
    mode_names = [m.mode_name for m in modes]
    assert "continuation" in mode_names
    assert "nudge_obstacle" in mode_names
    assert "cut_in_merge" in mode_names

    # Check uncertainty expansion over time: sigma(T=3s) > sigma(T=0.5s)
    traj = modes[0]
    assert traj.waypoints[-1].sigma_x > traj.waypoints[0].sigma_x
    assert traj.waypoints[-1].sigma_y > traj.waypoints[0].sigma_y


def test_pedestrian_multi_modal_reversal_and_halt():
    """Verify pedestrian multi-modal branching includes hesitate/halt and reverse options."""
    kin_pred = KinematicPredictor(horizon_seconds=3.0, dt=0.5)
    ped = _create_mock_obstacle("ped_cross", ObstacleClass.PEDESTRIAN, x=10.0, y=3.0, vx=0.0, vy=-1.0)
    modes = kin_pred.predict_modes(ped, MotionIntent.CROSSING_PATH, current_time=0.0)

    assert len(modes) == 3
    halt_mode = next(m for m in modes if m.mode_name == "hesitate_halt")
    # Hesitate halt mode should have diminishing speed
    assert halt_mode.waypoints[-1].velocity.y < 0.1


def test_cattle_road_freeze_mode():
    """Verify cattle multi-modal branches handle sudden road freezing/blocking."""
    kin_pred = KinematicPredictor(horizon_seconds=3.0, dt=0.5)
    cow = _create_mock_obstacle("cow_block", ObstacleClass.CATTLE_ANIMAL, x=12.0, y=0.5, vx=0.5, vy=0.0)
    modes = kin_pred.predict_modes(cow, MotionIntent.CRUISING, current_time=0.0)

    freeze_mode = next(m for m in modes if m.mode_name == "road_freeze")
    assert freeze_mode.probability >= 0.40
    assert freeze_mode.waypoints[-1].velocity.x < 0.05


def test_learned_predictor_pytorch_forward():
    """Verify PyTorch neural network forward pass and trajectory formatting."""
    learned_pred = LearnedPredictor(horizon_seconds=3.0, dt=0.5)
    auto = _create_mock_obstacle("auto_1", ObstacleClass.AUTO_RICKSHAW, x=14.0, y=1.2, vx=4.0, vy=-0.3)
    modes = learned_pred.predict_modes(auto, MotionIntent.CUTTING_IN, current_time=0.0)

    assert len(modes) == 3
    assert all(len(m.waypoints) == 6 for m in modes)
    total_p = sum(m.probability for m in modes)
    assert pytest.approx(total_p, abs=1e-2) == 1.0


def test_unified_trajectory_predictor_pipeline():
    """Verify complete perception-to-prediction end-to-end pipeline with high risk detection."""
    predictor = TrajectoryPredictor(horizon_seconds=3.0, dt=0.5, mode="ensemble")

    bike = _create_mock_obstacle("bike_fast", ObstacleClass.MOTORCYCLE, x=8.0, y=1.0, vx=6.0, vy=-0.9)
    cow_static = _create_mock_obstacle("cow_stat", ObstacleClass.CATTLE_ANIMAL, x=25.0, y=3.0, vx=0.0, vy=0.0, is_static=True)

    perception = PerceptionOutput(
        timestamp=10.0,
        frame_id=100,
        obstacles=[bike, cow_static],
        drivable_corridor=FreeSpaceCorridor(timestamp=10.0, boundary_points=[], average_width_m=6.5)
    )

    pred_out = predictor.predict(perception, ego_speed=6.0)

    assert len(pred_out.agents) == 2
    assert pred_out.horizon_seconds == 3.0
    # The fast cutting-in bike at 8m should be flagged high risk
    assert "bike_fast" in pred_out.high_risk_agent_ids
