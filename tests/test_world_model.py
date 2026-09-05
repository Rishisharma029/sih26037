"""
Tests for WorldModel Track Representation and State Serialization.
"""
import pytest
from interfaces import ObstacleClass, Point3D, Vector3D
from perception.world_model import WorldModel, WorldModelTrack, TrackHistoryPoint

def test_world_model_track_instantiation():
    track = WorldModelTrack(
        object_id="trk_auto_01",
        obstacle_type=ObstacleClass.AUTO_RICKSHAW,
        position=Point3D(x=15.0, y=1.2, z=0.8),
        velocity=Vector3D(x=6.5, y=-0.2, z=0.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
        heading_rad=0.05,
        size=Vector3D(x=2.6, y=1.3, z=1.7),
        confidence=0.95,
        risk_score=0.75
    )

    assert track.object_id == "trk_auto_01"
    assert track.obstacle_type == ObstacleClass.AUTO_RICKSHAW
    assert track.risk_score == 0.75

    # Test conversion to standard TrackedObstacle
    obs = track.to_tracked_obstacle()
    assert obs.id == "trk_auto_01"
    assert abs(obs.distance_m - 15.04) < 0.1
