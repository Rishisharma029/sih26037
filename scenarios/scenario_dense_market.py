"""Benchmark 4: High-Density Pedestrian & Pushcart Market Street."""
from interfaces import Point3D, Vector3D, ObstacleClass
from simulation.environment import ObstacleDefinition
from .scenario_base import BaseScenario

class DenseMarketScenario(BaseScenario):
    def __init__(self):
        super().__init__(name="Dense Market Street", duration_seconds=15.0)

    def setup_environment(self):
        self.env.road_width_m = 5.0
        # Vegetable pushcart stationary in corridor
        self.env.add_obstacle(ObstacleDefinition(
            id="vendor_cart",
            obstacle_class=ObstacleClass.PUSHCART,
            position=Point3D(x=18.0, y=-1.0, z=0.5),
            size=Vector3D(x=1.8, y=1.1, z=1.2),
            speed_mps=0.0,
            is_static=True
        ))
