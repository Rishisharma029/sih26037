"""Benchmark 1: Unmarked Narrow Village Road with Lateral Drop-offs."""
from interfaces import Point3D, Vector3D, ObstacleClass
from simulation.environment import ObstacleDefinition
from .scenario_base import BaseScenario

class UnmarkedVillageRoadScenario(BaseScenario):
    def __init__(self):
        super().__init__(name="Unmarked Village Road", duration_seconds=12.0)

    def setup_environment(self):
        self.env.road_width_m = 4.5
        # Add oncoming slow tractor on edge
        self.env.add_obstacle(ObstacleDefinition(
            id="tractor_01",
            obstacle_class=ObstacleClass.TRUCK,
            position=Point3D(x=35.0, y=1.2, z=1.2),
            size=Vector3D(x=4.0, y=2.0, z=2.2),
            speed_mps=-4.0,
            is_static=False
        ))
