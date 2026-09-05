"""
Benchmark 5: Sudden Cattle Roadblock on Blind Curve (Varanasi Highway Bend, UP).
"""
from interfaces import ObstacleClass
from simulation.environment import RoadEnvironment
from simulation.actors import SimulationActor
from .scenario_base import BaseScenario
from .geospatial.geo_registry import INDIAN_SCENARIO_REGISTRY

class CattleCrossingScenario(BaseScenario):
    """Scenario 5: Resting and wandering stray cows on blind curve."""
    def __init__(self, duration_seconds: float = 12.0, dt: float = 0.05):
        self.geo_spec = INDIAN_SCENARIO_REGISTRY["SCENARIO_5_CATTLE"]
        super().__init__(name=f"Scenario 5: {self.geo_spec.name} ({self.geo_spec.location_name})", duration_seconds=duration_seconds, dt=dt)

    def setup_environment(self):
        self.env = RoadEnvironment(length_m=180.0)
        self.env.geometry.length_m = 180.0

        # Resting holy cow directly in lane center on the curve (x=45m, y=0.1m)
        self.env.add_actor(SimulationActor(
            id="resting_cow_01",
            obstacle_class=ObstacleClass.CATTLE_ANIMAL,
            x=45.0,
            y=0.1,
            length_m=2.1,
            width_m=0.9,
            height_m=1.3,
            speed_mps=0.0,
            is_static=True
        ))

        # Second stray calf moving slowly on left verge (x=55m, y=1.4m)
        self.env.add_actor(SimulationActor(
            id="stray_calf_02",
            obstacle_class=ObstacleClass.CATTLE_ANIMAL,
            x=55.0,
            y=1.4,
            yaw_rad=-0.4,
            speed_mps=0.6,
            length_m=1.4,
            width_m=0.6,
            height_m=1.0,
            is_static=False
        ))
