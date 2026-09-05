"""Benchmark 3: Aggressive Auto-Rickshaw Cut-In & Highway Merge."""
from interfaces import Point3D, Vector3D, ObstacleClass
from simulation.environment import ObstacleDefinition
from .scenario_base import BaseScenario

class HighwayCutInScenario(BaseScenario):
    def __init__(self):
        super().__init__(name="Aggressive Cut-In", duration_seconds=8.0)

    def setup_environment(self):
        self.env.road_width_m = 7.0
        # Auto cutting in rapidly from left shoulder
        self.env.add_obstacle(ObstacleDefinition(
            id="cutin_rickshaw",
            obstacle_class=ObstacleClass.AUTO_RICKSHAW,
            position=Point3D(x=15.0, y=2.2, z=0.8),
            size=Vector3D(x=2.6, y=1.3, z=1.7),
            speed_mps=6.0,
            is_static=False
        ))
