"""Artificial Potential Field (APF) repulsive obstacle forces."""
import math
from typing import List, Tuple
from interfaces import TrackedObstacle, Point3D

class ArtificialPotentialField:
    """Applies repulsive potential fields from obstacles and road ditches."""
    def __init__(self, k_rep: float = 15.0, d_influence_m: float = 8.0):
        self.k_rep = k_rep
        self.d_influence_m = d_influence_m

    def compute_repulsive_force(self, ego_pos: Point3D, obstacles: List[TrackedObstacle]) -> tuple[float, float]:
        f_rx = 0.0
        f_ry = 0.0
        for obs in obstacles:
            dx = ego_pos.x - obs.bbox.center.x
            dy = ego_pos.y - obs.bbox.center.y
            dist = math.hypot(dx, dy)
            if 0.1 < dist < self.d_influence_m:
                force = self.k_rep * (1.0 / dist - 1.0 / self.d_influence_m) / (dist ** 2)
                f_rx += force * (dx / dist)
                f_ry += force * (dy / dist)
        return f_rx, f_ry
