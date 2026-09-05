"""Benchmark 3: Aggressive Auto-Rickshaw Cut-In & Highway Merge."""
from interfaces import Point3D, Vector3D, ObstacleClass
from simulation.actors import SimulationActor
from .scenario_base import BaseScenario

class HighwayCutInScenario(BaseScenario):
    def __init__(self):
        super().__init__(name="Aggressive Cut-In", duration_seconds=8.0)

    def setup_environment(self):
        self.env.geometry.length_m = 150.0
        # Auto cutting in rapidly from left shoulder
        self.env.add_actor(SimulationActor(
            id="cutin_rickshaw",
            obstacle_class=ObstacleClass.AUTO_RICKSHAW,
            x=15.0,
            y=2.2,
            z=0.0,
            length_m=2.6,
            width_m=1.3,
            height_m=1.7,
            speed_mps=6.0,
            is_static=False
        ))
