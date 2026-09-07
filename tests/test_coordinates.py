"""Unit tests for Coordinate Frame Transformation Engine."""
import math
import pytest
from interfaces import Point3D, Vector3D, Pose3D, Twist3D, BoundingBox3D, ObstacleClass
from coordinates import (
    world_to_ego_2d,
    ego_to_world_2d,
    world_to_ego_velocity,
    ego_to_world_velocity,
    world_to_ego_heading,
    ego_to_world_heading,
    transform_point_to_ego,
    transform_point_to_world,
    transform_actor_to_ego_tracked_obstacle
)


def test_pure_forward_translation():
    """Object directly ahead of ego with zero yaw."""
    x_ego, y_ego = world_to_ego_2d(
        x_world=25.0, y_world=2.0,
        ego_x=10.0, ego_y=2.0,
        ego_heading_rad=0.0
    )
    assert abs(x_ego - 15.0) < 1e-6
    assert abs(y_ego - 0.0) < 1e-6


def test_pure_lateral_offset():
    """Object to the left and right of ego vehicle with zero yaw."""
    # Left (+Y)
    x_ego_left, y_ego_left = world_to_ego_2d(
        x_world=10.0, y_world=4.5,
        ego_x=10.0, ego_y=2.0,
        ego_heading_rad=0.0
    )
    assert abs(x_ego_left - 0.0) < 1e-6
    assert abs(y_ego_left - 2.5) < 1e-6

    # Right (-Y)
    x_ego_right, y_ego_right = world_to_ego_2d(
        x_world=10.0, y_world=-1.0,
        ego_x=10.0, ego_y=2.0,
        ego_heading_rad=0.0
    )
    assert abs(x_ego_right - 0.0) < 1e-6
    assert abs(y_ego_right - (-3.0)) < 1e-6


def test_heading_rotation_90_deg():
    """Ego vehicle pointing North (pi/2 rad).
    World +Y is now ego Forward (+X_ego).
    World -X is now ego Left (+Y_ego).
    """
    # Object at (0, 10) in world, ego at (0, 0) facing North
    x_ego, y_ego = world_to_ego_2d(
        x_world=0.0, y_world=10.0,
        ego_x=0.0, ego_y=0.0,
        ego_heading_rad=math.pi / 2.0
    )
    assert abs(x_ego - 10.0) < 1e-5 # Forward ahead
    assert abs(y_ego - 0.0) < 1e-5

    # Object at (-5, 0) in world (West) -> Should be Left (+Y_ego = 5)
    x_ego_w, y_ego_w = world_to_ego_2d(
        x_world=-5.0, y_world=0.0,
        ego_x=0.0, ego_y=0.0,
        ego_heading_rad=math.pi / 2.0
    )
    assert abs(x_ego_w - 0.0) < 1e-5
    assert abs(y_ego_w - 5.0) < 1e-5


def test_round_trip_invariance():
    """Ego -> World -> Ego round-trip should be identity within numerical precision."""
    ego_x, ego_y, ego_yaw = 42.3, -15.7, 0.785398 # 45 degrees
    orig_world_x, orig_world_y = 65.4, 8.2

    x_ego, y_ego = world_to_ego_2d(orig_world_x, orig_world_y, ego_x, ego_y, ego_yaw)
    rec_world_x, rec_world_y = ego_to_world_2d(x_ego, y_ego, ego_x, ego_y, ego_yaw)

    assert abs(orig_world_x - rec_world_x) < 1e-5
    assert abs(orig_world_y - rec_world_y) < 1e-5


def test_relative_velocity_transformation():
    """Vehicle moving East at 10 m/s, Oncoming truck moving West at 15 m/s.
    Relative closing velocity in ego frame should be -25 m/s.
    """
    ego_yaw = 0.0
    vx_ego, vy_ego = world_to_ego_velocity(
        vx_world=-15.0, vy_world=0.0,
        ego_heading_rad=ego_yaw,
        ego_vx_world=10.0, ego_vy_world=0.0
    )
    assert abs(vx_ego - (-25.0)) < 1e-5
    assert abs(vy_ego - 0.0) < 1e-5


def test_transform_actor_to_ego_tracked_obstacle():
    """Verify complete actor to TrackedObstacle conversion in body coordinates."""
    ego_pose = Pose3D(position=Point3D(x=20.0, y=1.0, z=0.0), heading_rad=0.0)
    ego_twist = Twist3D(speed_mps=6.0)

    obs = transform_actor_to_ego_tracked_obstacle(
        actor_id="cow_01",
        obstacle_class=ObstacleClass.CATTLE_ANIMAL,
        x_world=32.0,
        y_world=2.5,
        z_world=0.0,
        length_m=2.0,
        width_m=0.8,
        height_m=1.4,
        yaw_world_rad=0.0,
        speed_mps=1.0,
        is_static=False,
        ego_pose=ego_pose,
        ego_twist=ego_twist
    )

    assert obs.id == "cow_01"
    assert obs.obstacle_class == ObstacleClass.CATTLE_ANIMAL
    assert abs(obs.bbox.center.x - 12.0) < 1e-3 # 32 - 20 = 12m ahead
    assert abs(obs.bbox.center.y - 1.5) < 1e-3  # 2.5 - 1.0 = 1.5m to the left
    assert abs(obs.distance_m - math.hypot(12.0, 1.5)) < 1e-3
    assert abs(obs.velocity.x - (-5.0)) < 1e-3 # 1 m/s - 6 m/s = -5 m/s closing
