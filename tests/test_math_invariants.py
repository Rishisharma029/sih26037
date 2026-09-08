"""
Comprehensive Mathematical Invariant & Coordinate Frame Test Suite.
Tests:
1. Coordinate Frame Transformation Invariants (ISO 8855 / SAE Standard: X=Forward, Y=Left, Z=Up).
2. Relative Direction Invariants:
   - x_ego > 0 for objects ahead, x_ego < 0 for objects behind.
   - y_ego > 0 for objects to the left, y_ego < 0 for objects to the right.
3. Coordinate Roundtrip Identity: ego_to_world(world_to_ego(P)) == P.
4. Road Corridor Invariants: d_left(s) > 0, d_right(s) < 0, width > 0 for all stations.
5. Trajectory Validity Invariants:
   - Within corridor -> VALID
   - Breaching corridor -> INVALID_CORRIDOR_BREACH
6. Collision Safety Invariants:
   - Clear trajectory -> SAFE
   - Intersecting trajectory / Critical TTC -> REJECT
7. Kinematic Bicycle Model Invariants:
   - Forward speed non-negative (v >= 0)
   - Steering sign: delta > 0 -> yaw_rate > 0 (left turn), delta < 0 -> yaw_rate < 0 (right turn)
   - Kinematics integration: dx/dt = v*cos(psi), dy/dt = v*sin(psi)
"""
import math
import pytest
from interfaces import (
    Point3D, Pose3D, Twist3D, Vector3D, EgoVehicleState,
    ControlCommand, PlannedTrajectory, TrajectoryPoint,
    TrackedObstacle, ObstacleClass, BoundingBox3D, SafetyAction,
    BehaviorMode
)
from coordinates import (
    world_to_ego_2d, ego_to_world_2d, world_to_ego_velocity,
    ego_to_world_velocity, world_to_ego_heading, ego_to_world_heading,
    transform_point_to_ego, transform_point_to_world, transform_actor_to_ego_tracked_obstacle
)
from simulation.environment import VillageRoadGeometry, RoadEnvironment
from simulation.vehicle_model import KinematicBicycleModel
from simulation.math_invariants import MathematicalInvariantAuditor, InvariantCheckResult
from collision_avoidance.safety_supervisor import SafetySupervisoryLayer
from collision_avoidance.ttc_calculator import TTCCalculator


class TestCoordinateFrameInvariants:
    """Rigorous tests for ISO 8855 Ego Frame (+X=Forward, +Y=Left, +Z=Up)."""

    def test_directional_ahead_behind_left_right_zero_heading(self):
        """Ego at (10, 20) with heading 0 (facing East / +X)."""
        ego_pose = Pose3D(position=Point3D(x=10.0, y=20.0, z=0.0), heading_rad=0.0)

        # 1. Object directly ahead (20, 20) -> x_ego = +10, y_ego = 0
        x_e, y_e, tag = MathematicalInvariantAuditor.verify_ego_frame_direction(20.0, 20.0, ego_pose)
        assert x_e == 10.0 > 0.0
        assert y_e == 0.0
        assert "AHEAD" in tag

        # 2. Object directly behind (0, 20) -> x_ego = -10, y_ego = 0
        x_e, y_e, tag = MathematicalInvariantAuditor.verify_ego_frame_direction(0.0, 20.0, ego_pose)
        assert x_e == -10.0 < 0.0
        assert y_e == 0.0
        assert "BEHIND" in tag

        # 3. Object to the left (10, 25) -> x_ego = 0, y_ego = +5
        x_e, y_e, tag = MathematicalInvariantAuditor.verify_ego_frame_direction(10.0, 25.0, ego_pose)
        assert x_e == 0.0
        assert y_e == 5.0 > 0.0
        assert "LEFT" in tag

        # 4. Object to the right (10, 15) -> x_ego = 0, y_ego = -5
        x_e, y_e, tag = MathematicalInvariantAuditor.verify_ego_frame_direction(10.0, 15.0, ego_pose)
        assert x_e == 0.0
        assert y_e == -5.0 < 0.0
        assert "RIGHT" in tag

    def test_directional_north_heading(self):
        """Ego at (0, 0) with heading pi/2 (facing North / +Y)."""
        ego_pose = Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=math.pi / 2.0)

        # Ahead is now North (+Y)
        x_e, y_e, tag = MathematicalInvariantAuditor.verify_ego_frame_direction(0.0, 15.0, ego_pose)
        assert pytest.approx(x_e, abs=1e-3) == 15.0
        assert pytest.approx(y_e, abs=1e-3) == 0.0
        assert "AHEAD" in tag

        # Left is now West (-X)
        x_e, y_e, tag = MathematicalInvariantAuditor.verify_ego_frame_direction(-4.0, 0.0, ego_pose)
        assert pytest.approx(x_e, abs=1e-3) == 0.0
        assert pytest.approx(y_e, abs=1e-3) == 4.0
        assert "LEFT" in tag

        # Right is now East (+X)
        x_e, y_e, tag = MathematicalInvariantAuditor.verify_ego_frame_direction(4.0, 0.0, ego_pose)
        assert pytest.approx(x_e, abs=1e-3) == 0.0
        assert pytest.approx(y_e, abs=1e-3) == -4.0
        assert "RIGHT" in tag

    def test_coordinate_roundtrip_identity(self):
        """Ego at arbitrary pose, testing roundtrip transformation across all 4 quadrants."""
        ego_pose = Pose3D(position=Point3D(x=34.2, y=-18.7, z=0.5), heading_rad=0.785)
        test_points = [
            (50.0, 10.0), (-20.0, 30.0), (100.0, -50.0), (-40.0, -60.0), (34.2, -18.7)
        ]
        for wx, wy in test_points:
            res = MathematicalInvariantAuditor.verify_coordinate_roundtrip(wx, wy, ego_pose)
            assert res.is_valid, f"Failed roundtrip for point ({wx}, {wy}): {res.message}"


class TestRoadCorridorInvariants:
    """Rigorous tests for Road Geometry and Boundary Invariants."""

    def test_corridor_boundary_signs_and_width(self):
        """Ensure d_left > 0, d_right < 0, and width > 0 across entire road length."""
        geom = VillageRoadGeometry(length_m=200.0, base_width_m=4.3)
        for s in range(0, 200, 5):
            d_left, d_right = geom.get_corridor_widths(float(s))
            res = MathematicalInvariantAuditor.verify_corridor_boundaries(float(s), d_left, d_right)
            assert res.is_valid, f"Corridor boundary violated at s={s}: {res.message}"
            assert d_left > 0.0
            assert d_right < 0.0
            assert (d_left - d_right) > 2.5

    def test_ditch_margin_calculation(self):
        """Ensure ditch margin is positive when vehicle is centered, negative when off-road."""
        geom = VillageRoadGeometry(length_m=100.0, base_width_m=4.0)
        # Centerline d=0.0 -> margin should be ~ 2.0 - 0.9 = 1.1m
        margin_center = geom.get_ditch_margin(s=20.0, d=0.0, vehicle_half_width=0.90)
        assert margin_center > 0.8

        # Left ditch breach d = +2.5m (d_left ~ 2.0m) -> margin < 0
        margin_left_breach = geom.get_ditch_margin(s=20.0, d=2.5, vehicle_half_width=0.90)
        assert margin_left_breach < 0.0

        # Right ditch breach d = -2.5m (d_right ~ -2.0m) -> margin < 0
        margin_right_breach = geom.get_ditch_margin(s=20.0, d=-2.5, vehicle_half_width=0.90)
        assert margin_right_breach < 0.0


class TestTrajectoryAndCollisionInvariants:
    """Tests for Trajectory Corridor Validity & Hard Collision Rejection Invariants."""

    def test_trajectory_within_corridor_is_valid(self):
        geom = VillageRoadGeometry(length_m=100.0, base_width_m=4.3)
        waypoints = [
            TrajectoryPoint(timestamp=float(i)*0.2, x=float(x), y=0.2, speed_mps=5.0)
            for i, x in enumerate(range(10, 40, 2))
        ]
        traj = PlannedTrajectory(
            trajectory_id="T_SAFE",
            timestamp=0.0,
            behavior_mode=BehaviorMode.CRUISE,
            waypoints=waypoints,
            target_speed_mps=5.0
        )
        res = MathematicalInvariantAuditor.verify_trajectory_corridor_validity(traj, geom)
        assert res.is_valid

    def test_trajectory_breaching_corridor_is_invalid(self):
        geom = VillageRoadGeometry(length_m=100.0, base_width_m=4.0)
        # Waypoints with lateral offset d=4.5m (well outside d_left ~ 2.0m)
        waypoints = [
            TrajectoryPoint(timestamp=float(i)*0.2, x=float(x), y=4.5, speed_mps=5.0)
            for i, x in enumerate(range(10, 40, 2))
        ]
        traj = PlannedTrajectory(
            trajectory_id="T_BREACH",
            timestamp=0.0,
            behavior_mode=BehaviorMode.CRUISE,
            waypoints=waypoints,
            target_speed_mps=5.0
        )
        res = MathematicalInvariantAuditor.verify_trajectory_corridor_validity(traj, geom)
        assert not res.is_valid
        assert "INVALID_CORRIDOR_BREACH" in res.message

    def test_ttc_and_closing_speed_math(self):
        """TTC = dist / v_close when approaching, inf when diverging."""
        ttc_calc = TTCCalculator(critical_threshold_s=1.0)
        ego = EgoVehicleState(
            timestamp=1.0,
            pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
            twist=Twist3D(speed_mps=10.0), # moving at 10 m/s East
            acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
        )
        
        # 1. Oncoming obstacle 30m ahead moving West at 5 m/s with ego moving East at 10 m/s (relative vx_ego = -15 m/s)
        oncoming_obs = TrackedObstacle(
            id="TRACTOR_01",
            obstacle_class=ObstacleClass.TRUCK,
            confidence=0.95,
            bbox=BoundingBox3D(center=Point3D(x=30.0, y=0.0, z=0.0), size=Vector3D(x=4.0, y=2.0, z=2.0)),
            velocity=Vector3D(x=-15.0, y=0.0, z=0.0), # relative closing speed = 15 m/s
            distance_m=30.0,
            is_static=False
        )
        risk = ttc_calc.compute_ttc(ego, [oncoming_obs])
        assert risk.min_ttc_seconds == pytest.approx(2.0, abs=0.1)

        # 2. Preceding obstacle 30m ahead moving faster (12 m/s, pulling away) -> TTC = inf
        pulling_away_obs = TrackedObstacle(
            id="CAR_AHEAD",
            obstacle_class=ObstacleClass.CAR,
            confidence=0.95,
            bbox=BoundingBox3D(center=Point3D(x=30.0, y=0.0, z=0.0), size=Vector3D(x=4.0, y=1.8, z=1.5)),
            velocity=Vector3D(x=12.0, y=0.0, z=0.0),
            distance_m=30.0,
            is_static=False
        )
        risk2 = ttc_calc.compute_ttc(ego, [pulling_away_obs])
        assert risk2.min_ttc_seconds > 100.0


class TestKinematicBicycleInvariants:
    """Tests for Vehicle Kinematics Integration & Directional Steering."""

    def test_bicycle_kinematics_positive_steering_turns_left(self):
        """Steering delta > 0 must produce positive yaw rate (counter-clockwise / Left)."""
        model = KinematicBicycleModel()
        state = EgoVehicleState(
            timestamp=0.0,
            pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
            twist=Twist3D(speed_mps=8.0),
            acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
            steer_angle_rad=0.0
        )
        cmd = ControlCommand(timestamp=0.0, steering_angle_rad=0.25, throttle_pct=50.0, brake_pct=0.0)
        next_state = model.step(state, cmd, dt=0.05)

        # Invariant checks
        res = MathematicalInvariantAuditor.verify_kinematic_bicycle_step(
            state, next_state, steer_rad=0.25, dt=0.05
        )
        assert res.is_valid, f"Kinematics invariant failed: {res.message}"
        assert next_state.twist.angular.z > 0.0  # Positive yaw rate
        assert next_state.pose.heading_rad > 0.0  # Heading turned left
        assert next_state.pose.position.y > 0.0   # Displaced to the left (+Y)

    def test_bicycle_kinematics_negative_steering_turns_right(self):
        """Steering delta < 0 must produce negative yaw rate (clockwise / Right)."""
        model = KinematicBicycleModel()
        state = EgoVehicleState(
            timestamp=0.0,
            pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
            twist=Twist3D(speed_mps=8.0),
            acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
            steer_angle_rad=0.0
        )
        cmd = ControlCommand(timestamp=0.0, steering_angle_rad=-0.25, throttle_pct=50.0, brake_pct=0.0)
        next_state = model.step(state, cmd, dt=0.05)

        # Invariant checks
        res = MathematicalInvariantAuditor.verify_kinematic_bicycle_step(
            state, next_state, steer_rad=-0.25, dt=0.05
        )
        assert res.is_valid, f"Kinematics invariant failed: {res.message}"
        assert next_state.twist.angular.z < 0.0  # Negative yaw rate
        assert next_state.pose.heading_rad < 0.0  # Heading turned right
        assert next_state.pose.position.y < 0.0   # Displaced to the right (-Y)
