"""Benchmark 5: Stationary Stray Cattle Roadblock on Blind Curvature."""
from interfaces import Point3D, Vector3D, ObstacleClass
from simulation.environment import ObstacleDefinition
from .scenario_base import BaseScenario

class CattleCrossingScenario(BaseScenario):
    def __init__(self):
        super().__init__(name="Stray Cattle Roadblock", duration_seconds=10.0)

    def setup_environment(self):
        self.env.road_width_m = 6.0
        # Two cows resting in middle of path
        self.env.add_obstacle(ObstacleDefinition(
            id="cattle_01",
            obstacle_class=ObstacleClass.CATTLE_ANIMAL,
            position=Point3D(x=22.0, y=0.2, z=0.7),
            size=Vector3D(x=2.0, y=0.9, z=1.4),
            speed_mps=0.0,
            is_static=True
        ))
