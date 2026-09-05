"""Road environment, free-space boundary generation, and dynamic entities."""
from dataclasses import dataclass
from typing import List
from interfaces import Point3D, ObstacleClass, Vector3D

@dataclass
class ObstacleDefinition:
    id: str
    obstacle_class: ObstacleClass
    position: Point3D
    size: Vector3D
    speed_mps: float = 0.0
    heading_rad: float = 0.0
    is_static: bool = False

class RoadEnvironment:
    """Generates unstructured road boundaries, ditches, potholes, and obstacles."""
    def __init__(self, road_width_m: float = 6.0, length_m: float = 200.0):
        self.road_width_m = road_width_m
        self.length_m = length_m
        self.obstacles: List[ObstacleDefinition] = []

    def add_obstacle(self, obs: ObstacleDefinition):
        self.obstacles.append(obs)

    def get_corridor_boundaries_at(self, s: float) -> tuple[float, float]:
        """Returns (d_left, d_right) at distance s."""
        half_w = self.road_width_m / 2.0
        return half_w, -half_w
