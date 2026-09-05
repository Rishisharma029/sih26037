"""
Benchmark 2: Unsignalled Chaotic Urban Multi-Way Intersection (Silk Board, Bengaluru).
"""
from interfaces import ObstacleClass
from simulation.environment import RoadEnvironment
from simulation.actors import SimulationActor
from .scenario_base import BaseScenario
from .geospatial.geo_registry import INDIAN_SCENARIO_REGISTRY

class UnsignalledJunctionScenario(BaseScenario):
    """Scenario 2: Unsignalled junction with perpendicular crossing and gap-filtering."""
    def __init__(self, duration_seconds: float = 12.0, dt: float = 0.05):
        self.geo_spec = INDIAN_SCENARIO_REGISTRY["SCENARIO_2_JUNCTION"]
        super().__init__(name=f"Scenario 2: {self.geo_spec.name} ({self.geo_spec.location_name})", duration_seconds=duration_seconds, dt=dt)

    def setup_environment(self):
        self.env = RoadEnvironment(length_m=160.0)
        self.env.geometry.length_m = 160.0

        # Perpendicular crossing auto-rickshaw (x=38m, crossing from left y=7.0 to y=-6.0 at 3.8 m/s)
        self.env.add_actor(SimulationActor(
            id="crossing_auto_rickshaw",
            obstacle_class=ObstacleClass.AUTO_RICKSHAW,
            x=38.0,
            y=6.5,
            yaw_rad=-1.5708, # Heading south
            speed_mps=3.8,
            length_m=2.6,
            width_m=1.3,
            height_m=1.7,
            is_static=False,
            trajectory_waypoints=[(38.0, 6.5), (38.0, -6.5)]
        ))

        # Filtering motorcycle swerving across gap (x=50m, traveling at 6.5 m/s)
        self.env.add_actor(SimulationActor(
            id="filtering_motorcycle",
            obstacle_class=ObstacleClass.MOTORCYCLE,
            x=50.0,
            y=1.8,
            yaw_rad=3.0, # Slight opposing angle
            speed_mps=6.2,
            length_m=1.9,
            width_m=0.8,
            height_m=1.2,
            is_static=False
        ))
