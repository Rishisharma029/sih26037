"""Unit tests for prediction subsystem."""
from prediction.intent_classifier import IntentClassifier
from prediction.trajectory_predictor import TrajectoryPredictor
from interfaces import TrackedObstacle, ObstacleClass, BoundingBox3D, Point3D, Vector3D, PerceptionOutput, FreeSpaceCorridor

def test_cutin_intent_classification():
    classifier = IntentClassifier()
    obs = TrackedObstacle(
        id="rick_1",
        obstacle_class=ObstacleClass.AUTO_RICKSHAW,
        confidence=0.9,
        bbox=BoundingBox3D(center=Point3D(x=10.0, y=2.0, z=0.5), size=Vector3D(x=2.5, y=1.2, z=1.6)),
        velocity=Vector3D(x=5.0, y=-1.0, z=0.0),
        distance_m=10.2,
        is_static=False
    )
    intent = classifier.classify_intent(obs)
    assert intent.value == "CUTTING_IN"

def test_trajectory_forecasting():
    predictor = TrajectoryPredictor(horizon_seconds=2.0, dt=0.5)
    obs = TrackedObstacle(
        id="cow_1",
        obstacle_class=ObstacleClass.CATTLE_ANIMAL,
        confidence=0.95,
        bbox=BoundingBox3D(center=Point3D(x=15.0, y=0.0, z=0.7), size=Vector3D(x=2.0, y=0.8, z=1.4)),
        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        distance_m=15.0,
        is_static=True
    )
    p_out = PerceptionOutput(
        timestamp=0.0,
        frame_id=1,
        obstacles=[obs],
        drivable_corridor=FreeSpaceCorridor(timestamp=0.0, boundary_points=[], average_width_m=6.0)
    )
    pred_out = predictor.predict(p_out)
    assert len(pred_out.agents) == 1
    assert len(pred_out.agents[0].trajectories[0].waypoints) == 4
