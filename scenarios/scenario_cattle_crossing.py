"""Benchmark 5: Stationary Stray Cattle Roadblock on Blind Curvature."""
from interfaces import Point3D, Vector3D, ObstacleClass
from simulation.actors import SimulationActor
from .scenario_base import BaseScenario

class CattleCrossingScenario(BaseScenario):
    def __init__(self):
        super().__init__(name="Stray Cattle Roadblock", duration_seconds=10.0)

    def setup_environment(self):
        self.env.geometry.length_m = 100.0
        # Two cows resting in middle of path
        self.env.add_actor(SimulationActor(
            id="cattle_01",
            obstacle_class=ObstacleClass.CATTLE_ANIMAL,
            x=22.0,
            y=0.2,
            z=0.0,
            length_m=2.0,
            width_m=0.9,
            height_m=1.4,
            speed_mps=0.0,
            is_static=True
        ))
