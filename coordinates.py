"""
SIH26037 Coordinate Frame Transformation Engine.
Single Source of Truth for converting between:
- World Frame (Global Cartesian X=East, Y=North, Z=Up, yaw=0 along +X)
- Ego Vehicle Body Frame (+X_ego=Forward, +Y_ego=Left, +Z_ego=Up, yaw=0 along vehicle heading)

Math:
    dx = x_world - ego_x
    dy = y_world - ego_y
    x_ego =  dx * cos(psi_ego) + dy * sin(psi_ego)   # Forward distance ahead
    y_ego = -dx * sin(psi_ego) + dy * cos(psi_ego)   # Lateral distance left (+) / right (-)
"""
import math
from typing import Tuple, Optional, List, Union
from interfaces import Point3D, Vector3D, Pose3D, Twist3D, BoundingBox3D, TrackedObstacle, ObstacleClass


def world_to_ego_2d(
    x_world: float,
    y_world: float,
    ego_x: float,
    ego_y: float,
    ego_heading_rad: float
) -> Tuple[float, float]:
    """Transforms a 2D point from global world coordinates to ego vehicle body frame.
    
    Args:
        x_world: Global X coordinate (m)
        y_world: Global Y coordinate (m)
        ego_x: Global X position of ego vehicle (m)
        ego_y: Global Y position of ego vehicle (m)
        ego_heading_rad: Global heading yaw of ego vehicle (radians, 0 = +X axis)
        
    Returns:
        (x_ego, y_ego):
            x_ego: Forward distance along vehicle heading (m) [positive = in front]
            y_ego: Lateral distance perpendicular to heading (m) [positive = left, negative = right]
    """
    dx = x_world - ego_x
    dy = y_world - ego_y
    cos_h = math.cos(ego_heading_rad)
    sin_h = math.sin(ego_heading_rad)
    
    x_ego = dx * cos_h + dy * sin_h
    y_ego = -dx * sin_h + dy * cos_h
    return x_ego, y_ego


def ego_to_world_2d(
    x_ego: float,
    y_ego: float,
    ego_x: float,
    ego_y: float,
    ego_heading_rad: float
) -> Tuple[float, float]:
    """Transforms a 2D point from ego vehicle body frame back to global world coordinates."""
    cos_h = math.cos(ego_heading_rad)
    sin_h = math.sin(ego_heading_rad)
    
    dx = x_ego * cos_h - y_ego * sin_h
    dy = x_ego * sin_h + y_ego * cos_h
    
    x_world = ego_x + dx
    y_world = ego_y + dy
    return x_world, y_world


def world_to_ego_velocity(
    vx_world: float,
    vy_world: float,
    ego_heading_rad: float,
    ego_vx_world: float = 0.0,
    ego_vy_world: float = 0.0
) -> Tuple[float, float]:
    """Transforms a 2D velocity vector to ego vehicle relative coordinates.
    
    Returns:
        (vx_ego, vy_ego):
            vx_ego: Forward relative speed (m/s) [negative = closing in, positive = pulling away]
            vy_ego: Lateral relative speed (m/s) [positive = moving left, negative = moving right]
    """
    dvx = vx_world - ego_vx_world
    dvy = vy_world - ego_vy_world
    cos_h = math.cos(ego_heading_rad)
    sin_h = math.sin(ego_heading_rad)
    
    vx_ego = dvx * cos_h + dvy * sin_h
    vy_ego = -dvx * sin_h + dvy * cos_h
    return vx_ego, vy_ego


def ego_to_world_velocity(
    vx_ego: float,
    vy_ego: float,
    ego_heading_rad: float,
    ego_vx_world: float = 0.0,
    ego_vy_world: float = 0.0
) -> Tuple[float, float]:
    """Transforms ego relative velocity back into world coordinates."""
    cos_h = math.cos(ego_heading_rad)
    sin_h = math.sin(ego_heading_rad)
    
    dvx = vx_ego * cos_h - vy_ego * sin_h
    dvy = vx_ego * sin_h + vy_ego * cos_h
    
    vx_world = ego_vx_world + dvx
    vy_world = ego_vy_world + dvy
    return vx_world, vy_world


def world_to_ego_heading(
    heading_world_rad: float,
    ego_heading_rad: float
) -> float:
    """Computes heading of an object relative to ego vehicle in range [-pi, pi]."""
    rel_h = heading_world_rad - ego_heading_rad
    return (rel_h + math.pi) % (2.0 * math.pi) - math.pi


def ego_to_world_heading(
    heading_ego_rad: float,
    ego_heading_rad: float
) -> float:
    """Computes global world heading from ego-relative heading in range [-pi, pi]."""
    world_h = heading_ego_rad + ego_heading_rad
    return (world_h + math.pi) % (2.0 * math.pi) - math.pi


def transform_point_to_ego(
    point_world: Point3D,
    ego_pose: Pose3D
) -> Point3D:
    """Converts a 3D point from world coordinates to ego frame."""
    x_ego, y_ego = world_to_ego_2d(
        point_world.x,
        point_world.y,
        ego_pose.position.x,
        ego_pose.position.y,
        ego_pose.heading_rad
    )
    z_ego = point_world.z - ego_pose.position.z
    return Point3D(x=round(x_ego, 4), y=round(y_ego, 4), z=round(z_ego, 4))


def transform_point_to_world(
    point_ego: Point3D,
    ego_pose: Pose3D
) -> Point3D:
    """Converts a 3D point from ego frame to world coordinates."""
    x_w, y_w = ego_to_world_2d(
        point_ego.x,
        point_ego.y,
        ego_pose.position.x,
        ego_pose.position.y,
        ego_pose.heading_rad
    )
    z_w = point_ego.z + ego_pose.position.z
    return Point3D(x=round(x_w, 4), y=round(y_w, 4), z=round(z_w, 4))


def transform_bbox_to_ego(
    bbox_world: BoundingBox3D,
    ego_pose: Pose3D
) -> BoundingBox3D:
    """Transforms a 3D bounding box to ego coordinates."""
    ego_center = transform_point_to_ego(bbox_world.center, ego_pose)
    ego_yaw = world_to_ego_heading(bbox_world.yaw_rad, ego_pose.heading_rad)
    return BoundingBox3D(
        center=ego_center,
        size=bbox_world.size,
        yaw_rad=round(ego_yaw, 4)
    )


def transform_actor_to_ego_tracked_obstacle(
    actor_id: str,
    obstacle_class: ObstacleClass,
    x_world: float,
    y_world: float,
    z_world: float,
    length_m: float,
    width_m: float,
    height_m: float,
    yaw_world_rad: float,
    speed_mps: float,
    is_static: bool,
    ego_pose: Pose3D,
    ego_twist: Optional[Twist3D] = None,
    confidence: float = 0.95
) -> TrackedObstacle:
    """Single canonical function to transform any world-frame simulation actor or perception track
    into a standardized TrackedObstacle expressed strictly in vehicle body coordinates.
    """
    # Position in ego frame
    x_ego, y_ego = world_to_ego_2d(
        x_world, y_world,
        ego_pose.position.x, ego_pose.position.y,
        ego_pose.heading_rad
    )
    z_ego = z_world - ego_pose.position.z + height_m / 2.0
    
    # Heading in ego frame
    heading_ego = world_to_ego_heading(yaw_world_rad, ego_pose.heading_rad)
    
    # Velocity in world and ego frame
    vx_w = speed_mps * math.cos(yaw_world_rad) if not is_static else 0.0
    vy_w = speed_mps * math.sin(yaw_world_rad) if not is_static else 0.0
    
    ego_vx_w = (ego_twist.speed_mps * math.cos(ego_pose.heading_rad)) if ego_twist else 0.0
    ego_vy_w = (ego_twist.speed_mps * math.sin(ego_pose.heading_rad)) if ego_twist else 0.0
    
    vx_ego, vy_ego = world_to_ego_velocity(
        vx_w, vy_w,
        ego_pose.heading_rad,
        ego_vx_w, ego_vy_w
    )
    
    distance_m = math.hypot(x_ego, y_ego)
    
    return TrackedObstacle(
        id=actor_id,
        obstacle_class=obstacle_class,
        confidence=confidence,
        bbox=BoundingBox3D(
            center=Point3D(x=round(x_ego, 4), y=round(y_ego, 4), z=round(z_ego, 4)),
            size=Vector3D(x=length_m, y=width_m, z=height_m),
            yaw_rad=round(heading_ego, 4)
        ),
        velocity=Vector3D(x=round(vx_ego, 4), y=round(vy_ego, 4), z=0.0),
        acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
        distance_m=round(distance_m, 4),
        is_static=is_static
    )
