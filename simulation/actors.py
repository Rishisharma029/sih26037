"""
Dynamic and Static actor definitions for simulation.
Models non-ego road users (Tractors, Auto-rickshaws, Pedestrians, Boulders).
"""
import math
from typing import List, Optional
from dataclasses import dataclass, field
from interfaces import ObstacleClass, Point3D, Vector3D, TrackedObstacle, BoundingBox3D

@dataclass
class SimulationActor:
    """Represents a simulated dynamic or static actor in the scene."""
    id: str
    obstacle_class: ObstacleClass
    x: float
    y: float
    z: float = 0.0
    yaw_rad: float = 0.0
    speed_mps: float = 0.0
    length_m: float = 2.5
    width_m: float = 1.4
    height_m: float = 1.6
    is_static: bool = False
    trajectory_waypoints: List[tuple[float, float]] = field(default_factory=list)
    current_wp_idx: int = 0

    def step(self, dt: float):
        """Update actor position based on velocity and waypoint sequence."""
        if self.is_static:
            return

        if self.trajectory_waypoints and self.current_wp_idx < len(self.trajectory_waypoints):
            target_x, target_y = self.trajectory_waypoints[self.current_wp_idx]
            dx = target_x - self.x
            dy = target_y - self.y
            dist = math.hypot(dx, dy)

            if dist < 0.8:
                self.current_wp_idx += 1
                if self.current_wp_idx >= len(self.trajectory_waypoints):
                    self.speed_mps = 0.0
                    return
                target_x, target_y = self.trajectory_waypoints[self.current_wp_idx]
                dx = target_x - self.x
                dy = target_y - self.y
                dist = math.hypot(dx, dy)

            self.yaw_rad = math.atan2(dy, dx)
            step_dist = min(dist, self.speed_mps * dt)
            self.x += step_dist * math.cos(self.yaw_rad)
            self.y += step_dist * math.sin(self.yaw_rad)
        else:
            # Constant velocity motion along yaw
            self.x += self.speed_mps * math.cos(self.yaw_rad) * dt
            self.y += self.speed_mps * math.sin(self.yaw_rad) * dt

    def to_tracked_obstacle(self, ego_x: float, ego_y: float) -> TrackedObstacle:
        """Convert simulation actor to Perception TrackedObstacle."""
        rel_x = (self.x - ego_x)
        rel_y = (self.y - ego_y)
        dist = math.hypot(rel_x, rel_y)
        vx = self.speed_mps * math.cos(self.yaw_rad)
        vy = self.speed_mps * math.sin(self.yaw_rad)

        return TrackedObstacle(
            id=self.id,
            obstacle_class=self.obstacle_class,
            confidence=0.98,
            bbox=BoundingBox3D(
                center=Point3D(x=self.x, y=self.y, z=self.z + self.height_m / 2.0),
                size=Vector3D(x=self.length_m, y=self.width_m, z=self.height_m),
                yaw_rad=self.yaw_rad
            ),
            velocity=Vector3D(x=vx, y=vy, z=0.0),
            acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
            distance_m=dist,
            is_static=self.is_static
        )
