"""
Comprehensive unit and integration tests for Phase 1 Simulation.
"""
import pytest
import math
from simulation.actors import SimulationActor
from simulation.environment import RoadEnvironment, VillageRoadGeometry
from simulation.vehicle_model import KinematicBicycleModel
from simulation.simulator import ClosedLoopSimulator
from scenarios.scenario_unmarked_village import UnmarkedVillageRoadScenario, build_unmarked_village_environment
from scenarios.run_village_baseline import run_village_road_baseline
from interfaces import ControlCommand, ObstacleClass, EgoVehicleState, Pose3D, Point3D, Twist3D, Vector3D

def test_actor_stepping():
    actor = SimulationActor(
        id="ped_01",
        obstacle_class=ObstacleClass.PEDESTRIAN,
        x=10.0,
        y=2.0,
        yaw_rad=-1.5708, # Moving south
        speed_mps=1.0,
        is_static=False
    )
    actor.step(dt=1.0)
    assert abs(actor.x - 10.0) < 1e-4
    assert abs(actor.y - 1.0) < 0.05

def test_road_geometry_and_irregular_corridor():
    geom = VillageRoadGeometry(length_m=200.0)
    x, y, yaw = geom.get_centerline_point(s=50.0)
    assert x == 50.0
    assert y == 0.0

    d_left, d_right = geom.get_corridor_widths(s=50.0)
    total_width = d_left - d_right
    assert 3.6 <= total_width <= 4.8  # Unmarked road width check

def test_vehicle_steering_rate_limiting():
    model = KinematicBicycleModel()
    ego = EgoVehicleState(
        timestamp=0.0,
        pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
        twist=Twist3D(speed_mps=5.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
        steer_angle_rad=0.0
    )
    # Demand max steering jump of 0.785 rad in 0.05s
    cmd = ControlCommand(timestamp=0.0, steering_angle_rad=0.785, throttle_pct=20.0, brake_pct=0.0)
    next_state = model.step(ego, cmd, dt=0.05)
    # Max rate is 0.6 rad/s -> max step is 0.03 rad
    assert next_state.steer_angle_rad <= 0.035
    assert next_state.steer_angle_rad > 0.0

def test_unmarked_village_scene_population():
    env = build_unmarked_village_environment()
    actor_ids = [a.id for a in env.actors]
    assert "roadside_boulder" in actor_ids
    assert "parked_auto_rickshaw" in actor_ids
    assert "oncoming_tractor" in actor_ids
    assert "crossing_pedestrian" in actor_ids
    assert len(env.anomalies) >= 1

def test_village_road_motion_baseline():
    result = run_village_road_baseline(target_speed_mps=5.0, total_time_s=10.0)
    assert result["final_x"] > 35.0  # Traveled at least 35m
    assert result["min_corridor_margin"] > 0.5  # Stayed inside the unmarked road boundaries
    assert result["avg_speed"] > 3.0
