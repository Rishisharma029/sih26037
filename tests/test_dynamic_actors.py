"""Comprehensive Unit Tests for Behaviorally Intelligent Dynamic Road Actors."""
import math
import pytest

from interfaces import (
    EgoVehicleState, Pose3D, Point3D, Twist3D, Vector3D, ObstacleClass
)
from simulation.actors import (
    SimulationActor, TractorActor, PedestrianActor, MotorcycleActor,
    AutoRickshawActor, CattleActor, ActorBehaviorState
)


def _create_mock_ego(x: float = 0.0, y: float = 0.0, speed: float = 6.0) -> EgoVehicleState:
    return EgoVehicleState(
        timestamp=10.0,
        pose=Pose3D(position=Point3D(x=x, y=y, z=0.0), heading_rad=0.0),
        twist=Twist3D(speed_mps=speed),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
    )


def test_tractor_center_drift_behavior():
    """Verify that oncoming tractor drifts laterally toward centerline to simulate avoiding shoulder erosion."""
    tractor = TractorActor(
        id="tractor_test",
        obstacle_class=ObstacleClass.TRUCK,
        x=50.0, y=1.2, speed_mps=3.0,
        base_lateral_y=1.2,
        drift_period_s=4.0,
        drift_amplitude_m=0.8
    )
    ego = _create_mock_ego(x=0.0, y=0.0, speed=6.0)

    initial_x = tractor.x
    initial_y = tractor.y

    # Step for 1 second (1/4 period -> maximum lateral drift toward center)
    for _ in range(20):
        tractor.step(0.05, ego)

    # Must move oncoming (x decreases)
    assert tractor.x < initial_x
    # Must have drifted closer to centerline (y decreases toward 0.0)
    assert tractor.y < initial_y
    assert tractor.behavior_state in [ActorBehaviorState.CENTER_DRIFT, ActorBehaviorState.NOMINAL]

    # Convert to tracked obstacle
    obs = tractor.to_tracked_obstacle(ego.pose.position.x, ego.pose.position.y)
    assert obs.obstacle_class == ObstacleClass.TRUCK
    assert obs.is_static is False
    assert obs.velocity.x < 0.0


def test_motorcycle_cut_in_maneuver():
    """Verify that oncoming motorcycle triggers aggressive cut-in into ego lane when approaching."""
    bike = MotorcycleActor(
        id="moto_test",
        obstacle_class=ObstacleClass.MOTORCYCLE,
        x=40.0, y=1.5, speed_mps=5.0,
        cut_in_target_y=-0.2
    )
    ego = _create_mock_ego(x=0.0, y=0.0, speed=6.0)

    # Initial state: far from ego (40m > 26m trigger)
    bike.step(0.05, ego)
    assert bike.cut_in_triggered is False
    assert bike.behavior_state == ActorBehaviorState.NOMINAL

    # Move bike within 24m
    bike.x = 22.0
    bike.step(0.05, ego)
    assert bike.cut_in_triggered is True
    assert bike.behavior_state == ActorBehaviorState.CUTTING_IN

    # Step through cut-in execution
    for _ in range(30):
        bike.step(0.05, ego)

    # Bike lateral position must have shifted across toward target_y (-0.2m)
    assert bike.y < 0.5


def test_auto_rickshaw_roadside_pull_out():
    """Verify that roadside parked auto-rickshaw detects oncoming vehicle, accelerates and merges into lane."""
    auto = AutoRickshawActor(
        id="auto_pull_out",
        obstacle_class=ObstacleClass.AUTO_RICKSHAW,
        x=20.0, y=1.8, speed_mps=0.0,
        target_lane_y=0.6,
        pull_out_dist_trigger_m=22.0
    )
    ego = _create_mock_ego(x=0.0, y=0.0, speed=6.0)

    assert auto.speed_mps == 0.0
    assert auto.behavior_state == ActorBehaviorState.PARKED_ROADSIDE

    # Ego at x=0, auto at x=20m (< 22m trigger)
    auto.step(0.05, ego)
    assert auto.pull_out_triggered is True
    assert auto.behavior_state == ActorBehaviorState.PULLING_OUT

    # Step through pull-out acceleration
    for _ in range(40):
        auto.step(0.05, ego)

    # Must accelerate from 0.0
    assert auto.speed_mps > 1.5
    # Must merge closer to target lane (y ~ 0.6)
    assert auto.y < 1.4
    assert auto.x > 20.0


def test_pedestrian_interactive_crossing_and_darting():
    """Verify that pedestrian initiates crossing or hesitation state when vehicle approaches."""
    ped = PedestrianActor(
        id="villager_test",
        obstacle_class=ObstacleClass.PEDESTRIAN,
        x=18.0, y=-2.0, speed_mps=1.4,
        crossing_target_y=2.0
    )
    ego = _create_mock_ego(x=0.0, y=0.0, speed=6.0)

    # Distance to ego is 18m (< 22m trigger)
    ped.step(0.05, ego)
    assert ped.decision_made is True
    assert ped.behavior_state in [
        ActorBehaviorState.CROSSING_NORMAL,
        ActorBehaviorState.HESITATION_STOP,
        ActorBehaviorState.SUDDEN_DART
    ]

    # Step pedestrian
    for _ in range(30):
        ped.step(0.05, ego)

    # Pedestrian moves across road (y increases from -2.0)
    assert ped.y >= -2.0


def test_cattle_crossing_and_freeze():
    """Verify cattle wanders across road and can enter road freeze mode in center corridor."""
    cow = CattleActor(
        id="cow_test",
        obstacle_class=ObstacleClass.CATTLE_ANIMAL,
        x=15.0, y=-0.2, speed_mps=0.6,
        crossing_dir=1.0,
        freeze_prob=1.0 # Guarantee freeze for test determinism
    )
    ego = _create_mock_ego(x=0.0, y=0.0, speed=6.0)

    # Cow is already at center (y = -0.2m) and ego is within 20m
    cow.step(0.05, ego)
    assert cow.has_frozen is True
    assert cow.behavior_state == ActorBehaviorState.ROAD_FREEZE
    assert cow.speed_mps == 0.0

    # Stay frozen for 2 seconds
    for _ in range(40):
        cow.step(0.05, ego)
    assert cow.behavior_state == ActorBehaviorState.ROAD_FREEZE

    # Resume after freeze timer (> 3.0s)
    for _ in range(30):
        cow.step(0.05, ego)
    assert cow.behavior_state == ActorBehaviorState.CROSSING_NORMAL
    assert cow.speed_mps > 0.0
