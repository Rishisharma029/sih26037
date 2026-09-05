"""
Benchmark 4: Dense Market Road (Chandni Chowk, Old Delhi).
"""
from interfaces import ObstacleClass
from simulation.environment import RoadEnvironment
from simulation.actors import SimulationActor
from .scenario_base import BaseScenario
from .geospatial.geo_registry import INDIAN_SCENARIO_REGISTRY

class DenseMarketScenario(BaseScenario):
    """Scenario 4: High-density pedestrian swarms and stationary vendor carts."""
    def __init__(self, duration_seconds: float = 15.0, dt: float = 0.05):
        self.geo_spec = INDIAN_SCENARIO_REGISTRY["SCENARIO_4_MARKET"]
        super().__init__(name=f"Scenario 4: {self.geo_spec.name} ({self.geo_spec.location_name})", duration_seconds=duration_seconds, dt=dt)

    def setup_environment(self):
        self.env = RoadEnvironment(length_m=140.0)
        self.env.geometry.length_m = 140.0

        # Stationary vegetable pushcart on left corridor (x=22m, y=1.2m)
        self.env.add_actor(SimulationActor(
            id="vendor_pushcart_01",
            obstacle_class=ObstacleClass.PUSHCART,
            x=22.0,
            y=1.1,
            length_m=1.8,
            width_m=1.2,
            height_m=1.3,
            is_static=True
        ))

        # Pedestrian 1 crossing casually
        self.env.add_actor(SimulationActor(
            id="market_shopper_01",
            obstacle_class=ObstacleClass.PEDESTRIAN,
            x=32.0,
            y=-1.8,
            yaw_rad=1.5708, # Heading north across road
            speed_mps=0.9,
            length_m=0.5,
            width_m=0.5,
            height_m=1.7,
            is_static=False,
            trajectory_waypoints=[(32.0, -1.8), (32.0, 2.0)]
        ))

        # Pedestrian 2 walking along right shoulder
        self.env.add_actor(SimulationActor(
            id="market_shopper_02",
            obstacle_class=ObstacleClass.PEDESTRIAN,
            x=45.0,
            y=-1.4,
            yaw_rad=0.0,
            speed_mps=1.1,
            length_m=0.5,
            width_m=0.5,
            height_m=1.7,
            is_static=False
        ))
