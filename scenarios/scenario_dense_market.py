"""Benchmark 4: High-Density Pedestrian & Pushcart Market Street."""
from interfaces import Point3D, Vector3D, ObstacleClass
from simulation.actors import SimulationActor
from .scenario_base import BaseScenario

class DenseMarketScenario(BaseScenario):
    def __init__(self):
        super().__init__(name="Dense Market Street", duration_seconds=15.0)

    def setup_environment(self):
        self.env.geometry.length_m = 100.0
        # Vegetable pushcart stationary in corridor
        self.env.add_actor(SimulationActor(
            id="vendor_cart",
            obstacle_class=ObstacleClass.PUSHCART,
            x=18.0,
            y=-1.0,
            z=0.0,
            length_m=1.8,
            width_m=1.1,
            height_m=1.2,
            speed_mps=0.0,
            is_static=True
        ))
