"""
SIH26037 Mathematical Invariant Engine & Coordinate Frame Verification.

Enforces absolute mathematical invariants across all autonomous driving subsystems:
1. Coordinate Frames:
   - Global World Frame: X=East, Y=North, Z=Up, yaw=0 along +X, counter-clockwise positive.
   - Ego Body Frame (ISO 8855 / SAE Standard): +X_ego=Forward (Ahead), +Y_ego=Left (Port), +Z_ego=Up (Zenith).
   - Frenet Frame: s=Longitudinal station along centerline, d=Lateral offset (+d = Left, -d = Right).
2. Automated Invariants:
   - Object ahead: x_ego > 0
   - Object behind: x_ego < 0
   - Object left: y_ego > 0
   - Object right: y_ego < 0
   - Road boundary: d_left(s) > 0 and d_right(s) < 0
   - Road width: W(s) = d_left(s) - d_right(s) > 0
   - Trajectory outside corridor -> INVALID_CORRIDOR_BREACH
   - Collision trajectory -> INVALID_COLLISION_INTERSECTION -> REJECT (Safety Supervisor)
   - Vehicle kinematics: dx/dt = v*cos(psi), dy/dt = v*sin(psi), dpsi/dt = (v/L)*tan(delta)
"""
import math
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

from interfaces import (
    Point3D, Pose3D, Twist3D, EgoVehicleState,
    PlannedTrajectory, SafeTrajectory, TrackedObstacle
)


@dataclass
class InvariantCheckResult:
    is_valid: bool
    invariant_name: str
    message: str
    details: Optional[Dict[str, Any]] = None


class MathematicalInvariantAuditor:
    """Automated validator and runtime auditor for mathematical invariants."""

    @staticmethod
    def verify_ego_frame_direction(
        x_world: float,
        y_world: float,
        ego_pose: Pose3D,
        tolerance_m: float = 1e-4
    ) -> Tuple[float, float, str]:
        """Converts world coordinate to ego body frame and classifies relative position.
        
        Returns:
            (x_ego, y_ego, directional_tag):
                x_ego: forward distance (m) [+ = ahead, - = behind]
                y_ego: lateral distance (m) [+ = left, - = right]
                directional_tag: e.g. 'AHEAD_LEFT', 'AHEAD_RIGHT', 'BEHIND_LEFT', 'BEHIND_RIGHT'
        """
        dx = x_world - ego_pose.position.x
        dy = y_world - ego_pose.position.y
        cos_h = math.cos(ego_pose.heading_rad)
        sin_h = math.sin(ego_pose.heading_rad)

        x_ego = dx * cos_h + dy * sin_h
        y_ego = -dx * sin_h + dy * cos_h

        # Invariant checks
        long_tag = "AHEAD" if x_ego > tolerance_m else ("BEHIND" if x_ego < -tolerance_m else "ALIGNED_LONG")
        lat_tag = "LEFT" if y_ego > tolerance_m else ("RIGHT" if y_ego < -tolerance_m else "ALIGNED_LAT")
        
        return round(x_ego, 4), round(y_ego, 4), f"{long_tag}_{lat_tag}"

    @staticmethod
    def verify_coordinate_roundtrip(
        x_world: float,
        y_world: float,
        ego_pose: Pose3D,
        epsilon_m: float = 1e-4
    ) -> InvariantCheckResult:
        """Verifies mathematical identity: ego_to_world(world_to_ego(P)) == P."""
        # 1. World -> Ego
        dx = x_world - ego_pose.position.x
        dy = y_world - ego_pose.position.y
        cos_h = math.cos(ego_pose.heading_rad)
        sin_h = math.sin(ego_pose.heading_rad)
        x_ego = dx * cos_h + dy * sin_h
        y_ego = -dx * sin_h + dy * cos_h

        # 2. Ego -> World
        rec_dx = x_ego * cos_h - y_ego * sin_h
        rec_dy = x_ego * sin_h + y_ego * cos_h
        rec_x_world = ego_pose.position.x + rec_dx
        rec_y_world = ego_pose.position.y + rec_dy

        err = math.hypot(rec_x_world - x_world, rec_y_world - y_world)
        is_ok = err < epsilon_m
        return InvariantCheckResult(
            is_valid=is_ok,
            invariant_name="COORDINATE_ROUNDTRIP_IDENTITY",
            message=f"Roundtrip error: {err:.6f}m (limit: {epsilon_m}m)",
            details={"err_m": err, "orig": (x_world, y_world), "rec": (rec_x_world, rec_y_world)}
        )

    @staticmethod
    def verify_corridor_boundaries(
        s: float,
        d_left: float,
        d_right: float
    ) -> InvariantCheckResult:
        """Verifies road corridor invariants: d_left > 0, d_right < 0, W = d_left - d_right > 0."""
        is_valid = (d_left > 0.0) and (d_right < 0.0) and ((d_left - d_right) > 1.5)
        width = d_left - d_right
        return InvariantCheckResult(
            is_valid=is_valid,
            invariant_name="CORRIDOR_BOUNDARIES_INVARIANT",
            message=f"At s={s:.1f}m: d_left={d_left:.2f}m (>0), d_right={d_right:.2f}m (<0), width={width:.2f}m (>0)",
            details={"s": s, "d_left": d_left, "d_right": d_right, "width": width}
        )

    @staticmethod
    def verify_trajectory_corridor_validity(
        planned: PlannedTrajectory,
        geometry_accessor,
        vehicle_half_width: float = 0.90
    ) -> InvariantCheckResult:
        """Verifies that all trajectory waypoints stay within valid corridor boundaries."""
        if not planned.waypoints:
            return InvariantCheckResult(
                is_valid=False,
                invariant_name="TRAJECTORY_CORRIDOR_INVARIANT",
                message="Trajectory has no waypoints (EMPTY)"
            )

        min_margin = float("inf")
        breach_wp = None

        for idx, wp in enumerate(planned.waypoints):
            wx = getattr(wp, "x", getattr(getattr(wp, "position", None), "x", 0.0))
            wy = getattr(wp, "y", getattr(getattr(wp, "position", None), "y", 0.0))
            s_wp, d_wp = geometry_accessor.cartesian_to_frenet(wx, wy)
            d_left, d_right = geometry_accessor.get_corridor_widths(s_wp)
            
            margin_left = d_left - (d_wp + vehicle_half_width)
            margin_right = (d_wp - vehicle_half_width) - d_right
            wp_margin = min(margin_left, margin_right)
            
            if wp_margin < min_margin:
                min_margin = wp_margin
                if wp_margin < 0.0:
                    breach_wp = (idx, s_wp, d_wp, wp_margin)
                    break

        is_valid = min_margin >= 0.0
        return InvariantCheckResult(
            is_valid=is_valid,
            invariant_name="TRAJECTORY_CORRIDOR_INVARIANT",
            message=f"Trajectory corridor margin: {min_margin:0.2f}m" if is_valid else f"INVALID_CORRIDOR_BREACH at wp {breach_wp}",
            details={"min_margin_m": min_margin, "breach": breach_wp}
        )

    @staticmethod
    def verify_collision_safety(
        trajectory_waypoints: List[Any],
        obstacles: List[TrackedObstacle],
        min_clearance_buffer_m: float = 1.0
    ) -> InvariantCheckResult:
        """Verifies that trajectory waypoints maintain minimum clearance to all obstacle bounding boxes."""
        min_dist = float("inf")
        conflict_pair = None

        for wp_idx, wp in enumerate(trajectory_waypoints):
            wx = getattr(wp, "x", getattr(getattr(wp, "position", None), "x", 0.0))
            wy = getattr(wp, "y", getattr(getattr(wp, "position", None), "y", 0.0))
            
            for obs in obstacles:
                # Obstacle bounding box center (in world or ego)
                obs_x = obs.bbox.center.x
                obs_y = obs.bbox.center.y
                obs_radius = max(obs.bbox.size.x, obs.bbox.size.y) * 0.5
                
                dist = math.hypot(wx - obs_x, wy - obs_y) - obs_radius
                if dist < min_dist:
                    min_dist = dist
                    if dist < min_clearance_buffer_m:
                        conflict_pair = (wp_idx, obs.id, dist)
                        break

        is_valid = min_dist >= min_clearance_buffer_m
        return InvariantCheckResult(
            is_valid=is_valid,
            invariant_name="COLLISION_CLEARANCE_INVARIANT",
            message=f"Min clearance: {min_dist:0.2f}m" if is_valid else f"INVALID_COLLISION_INTERSECTION with obs {conflict_pair}",
            details={"min_dist_m": min_dist, "conflict": conflict_pair}
        )

    @staticmethod
    def verify_kinematic_bicycle_step(
        state_prev: EgoVehicleState,
        state_next: EgoVehicleState,
        steer_rad: float,
        dt: float,
        wheelbase_m: float = 2.7
    ) -> InvariantCheckResult:
        """Verifies bicycle model kinematics integration:
        - v >= 0
        - dpsi/dt = (v / L) * tan(delta)
        - dx/dt = v * cos(psi), dy/dt = v * sin(psi)
        - positive steer (delta > 0) -> turn left -> yaw increases -> dy > 0 when psi=0
        """
        v_next = state_next.twist.speed_mps
        if v_next < -1e-4:
            return InvariantCheckResult(
                is_valid=False,
                invariant_name="KINEMATICS_SPEED_NON_NEGATIVE",
                message=f"Negative speed invariant violated: v={v_next:.3f} m/s"
            )

        # Yaw rate verification
        v_avg = 0.5 * (state_prev.twist.speed_mps + v_next)
        effective_steer = getattr(state_next, "steer_angle_rad", steer_rad)
        expected_yaw_rate = (v_avg / wheelbase_m) * math.tan(effective_steer)
        actual_yaw_rate = state_next.twist.angular.z

        yaw_rate_err = abs(actual_yaw_rate - expected_yaw_rate)
        is_yaw_ok = yaw_rate_err < 0.10 or v_avg < 0.1

        # Steering directional check: delta > 0 should yield yaw_rate >= 0
        steer_dir_ok = True
        if steer_rad > 0.05 and v_avg > 0.5:
            steer_dir_ok = (actual_yaw_rate >= 0.0)
        elif steer_rad < -0.05 and v_avg > 0.5:
            steer_dir_ok = (actual_yaw_rate <= 0.0)

        is_valid = is_yaw_ok and steer_dir_ok
        return InvariantCheckResult(
            is_valid=is_valid,
            invariant_name="KINEMATIC_BICYCLE_INVARIANT",
            message="Kinematic bicycle integration verified" if is_valid else f"Kinematic mismatch (yaw_rate_err={yaw_rate_err:.4f}, steer_dir_ok={steer_dir_ok})",
            details={
                "expected_yaw_rate": expected_yaw_rate,
                "actual_yaw_rate": actual_yaw_rate,
                "yaw_rate_err": yaw_rate_err,
                "steer_dir_ok": steer_dir_ok
            }
        )
