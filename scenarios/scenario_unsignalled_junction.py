"""Benchmark 2: Unsignalled Chaotic Urban Multi-Way Intersection."""
from interfaces import Point3D, Vector3D, ObstacleClass
from simulation.environment import ObstacleDefinition
from .scenario_base import BaseScenario

class UnsignalledJunctionScenario(BaseScenario):
    def __init__(self):
        super().__init__(name="Unsignalled Chaotic Junction", duration_seconds=10.0)

    def setup_environment(self):
        self.env.road_width_m = 8.0
        # Auto-rickshaw crossing perpendicular
        self.env.add_obstacle(ObstacleDefinition(
            id="rickshaw_crossing",
            obstacle_class=ObstacleClass.AUTO_RICKSHAW,
            position=Point3D(x=20.0, y=6.0, z=0.8),
            size=Vector3D(x=2.6, y=1.3, z=1.7),
            speed_mps=3.5,
            is_static=False
        ))
