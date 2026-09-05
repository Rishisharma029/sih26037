"""Unit tests for collision avoidance subsystem."""
from collision_avoidance.ttc_calculator import TTCCalculator
from collision_avoidance.emergency_brake import EmergencyBrakeSupervisory
from interfaces import (
    TrackedObstacle, ObstacleClass, BoundingBox3D, Point3D, Vector3D,
    EgoVehicleState, Pose3D, Twist3D, PlannedTrajectory, TrajectoryPoint,
    BehaviorMode, PerceptionOutput, FreeSpaceCorridor, PredictionOutput
)

def test_ttc_calculation():
    calc = TTCCalculator()
    ego = EgoVehicleState(
        timestamp=0.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0)),
        twist=Twist3D(speed_mps=10.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )
    obs = TrackedObstacle(
        id="lead_car",
        obstacle_class=ObstacleClass.CAR,
        confidence=0.9,
        bbox=BoundingBox3D(center=Point3D(x=10.0, y=0.0, z=0.5), size=Vector3D(x=4.0, y=1.8, z=1.5)),
        velocity=Vector3D(x=5.0, y=0.0, z=0.0),
        distance_m=10.0,
        is_static=False
    )
    risk = calc.compute_ttc(ego, [obs])
    # Relative dx=10m, relative vx=5m/s -> TTC=2.0s
    assert abs(risk.min_ttc_seconds - 2.0) < 0.1

def test_aeb_trigger():
    supervisory = EmergencyBrakeSupervisory(aeb_ttc_threshold_s=1.0)
    ego = EgoVehicleState(
        timestamp=0.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0)),
        twist=Twist3D(speed_mps=10.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )
    # Obstacle 4m ahead, rel speed 10m/s -> TTC = 0.4s (Critical)
    obs = TrackedObstacle(
        id="darting_ped",
        obstacle_class=ObstacleClass.PEDESTRIAN,
        confidence=0.95,
        bbox=BoundingBox3D(center=Point3D(x=4.0, y=0.0, z=0.8), size=Vector3D(x=0.5, y=0.5, z=1.7)),
        velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        distance_m=4.0,
        is_static=True
    )
    planned = PlannedTrajectory(
        trajectory_id="p1",
        timestamp=0.0,
        behavior_mode=BehaviorMode.CRUISE,
        waypoints=[TrajectoryPoint(timestamp=0.1, x=1.0, y=0.0, speed_mps=10.0)],
        target_speed_mps=10.0
    )
    perc = PerceptionOutput(
        timestamp=0.0, frame_id=1, obstacles=[obs],
        drivable_corridor=FreeSpaceCorridor(timestamp=0.0, boundary_points=[], average_width_m=6.0)
    )
    pred = PredictionOutput(timestamp=0.0, horizon_seconds=2.0, agents=[])
    safe = supervisory.supervise(planned, ego, perc, pred)

    assert safe.is_emergency_stop is True
    assert safe.waypoints[0].speed_mps == 0.0
