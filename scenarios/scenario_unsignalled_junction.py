"""Benchmark 2: Unsignalled Chaotic Urban Multi-Way Intersection."""
from interfaces import Point3D, Vector3D, ObstacleClass
from simulation.actors import SimulationActor
from .scenario_base import BaseScenario

class UnsignalledJunctionScenario(BaseScenario):
    def __init__(self):
        super().__init__(name="Unsignalled Chaotic Junction", duration_seconds=10.0)

    def setup_environment(self):
        self.env.geometry.length_m = 120.0
        # Auto-rickshaw crossing perpendicular
        self.env.add_actor(SimulationActor(
            id="rickshaw_crossing",
            obstacle_class=ObstacleClass.AUTO_RICKSHAW,
            x=20.0,
            y=6.0,
            z=0.0,
            length_m=2.6,
            width_m=1.3,
            height_m=1.7,
            speed_mps=3.5,
            is_static=False
        ))
