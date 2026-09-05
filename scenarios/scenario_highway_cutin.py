"""
Benchmark 3: Highway Merge with Slow Vehicle (NH-48 Jaipur, Rajasthan).
"""
from interfaces import ObstacleClass
from simulation.environment import RoadEnvironment
from simulation.actors import SimulationActor
from .scenario_base import BaseScenario
from .geospatial.geo_registry import INDIAN_SCENARIO_REGISTRY

class HighwayCutInScenario(BaseScenario):
    """Scenario 3: Low-speed three-wheeler cutting in aggressively on highway."""
    def __init__(self, duration_seconds: float = 10.0, dt: float = 0.05):
        self.geo_spec = INDIAN_SCENARIO_REGISTRY["SCENARIO_3_HIGHWAY"]
        super().__init__(name=f"Scenario 3: {self.geo_spec.name} ({self.geo_spec.location_name})", duration_seconds=duration_seconds, dt=dt)

    def setup_environment(self):
        self.env = RoadEnvironment(length_m=220.0)
        self.env.geometry.length_m = 220.0

        # Slow commercial 3-wheeler cutting in from right shoulder into ego lane
        self.env.add_actor(SimulationActor(
            id="cutin_tempo_01",
            obstacle_class=ObstacleClass.AUTO_RICKSHAW,
            x=28.0,
            y=-2.4, # On shoulder
            yaw_rad=0.35, # Angling into center lane
            speed_mps=6.5, # 23.4 km/h (Slow relative to highway)
            length_m=3.0,
            width_m=1.4,
            height_m=1.8,
            is_static=False,
            trajectory_waypoints=[(28.0, -2.4), (55.0, 0.0), (120.0, 0.0)]
        ))
