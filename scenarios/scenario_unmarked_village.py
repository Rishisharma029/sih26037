"""
Phase 1 Benchmark Scene: Unmarked Indian Village Road.
Sets up the complete scene:
1. Road without lane markings & irregular edges (3.8m - 4.6m width).
2. Roadside static boulder / barrier.
3. Parked static auto-rickshaw on shoulder.
4. Roadway deep pothole anomaly.
5. Oncoming slow tractor.
6. Crossing village pedestrian.
"""
from interfaces import ObstacleClass, Point3D, Vector3D, RoadAnomaly
from simulation.environment import RoadEnvironment
from simulation.actors import SimulationActor
from scenarios.scenario_base import BaseScenario

def build_unmarked_village_environment() -> RoadEnvironment:
    env = RoadEnvironment(length_m=200.0)

    # 1. Roadside Static Boulder (x=45m, y=2.6m on left verge)
    env.add_actor(SimulationActor(
        id="roadside_boulder",
        obstacle_class=ObstacleClass.STATIC_DEBRIS,
        x=45.0,
        y=2.6,
        length_m=1.2,
        width_m=0.9,
        height_m=0.7,
        is_static=True
    ))

    # 2. Parked Auto-Rickshaw on right shoulder (x=65m, y=-2.2m)
    env.add_actor(SimulationActor(
        id="parked_auto_rickshaw",
        obstacle_class=ObstacleClass.AUTO_RICKSHAW,
        x=65.0,
        y=-2.2,
        length_m=2.6,
        width_m=1.3,
        height_m=1.7,
        is_static=True
    ))

    # 3. Deep Road Pothole Anomaly (x=88m, y=0.3m)
    env.add_anomaly(RoadAnomaly(
        id="village_pothole_01",
        anomaly_type="POTHOLE",
        position=Point3D(x=88.0, y=0.3, z=-0.12),
        radius_m=0.45,
        depth_or_height_m=-0.12
    ))

    # 4. Oncoming Slow Tractor (starts at x=135m, traveling towards ego at 3.5 m/s on its left lane)
    env.add_actor(SimulationActor(
        id="oncoming_tractor",
        obstacle_class=ObstacleClass.TRUCK,
        x=135.0,
        y=1.1,
        yaw_rad=3.14159,
        speed_mps=3.5,
        length_m=4.2,
        width_m=2.0,
        height_m=2.3,
        is_static=False
    ))

    # 5. Crossing Pedestrian (starts at x=52m, y=3.2m, crossing towards right shoulder at 1.2 m/s)
    env.add_actor(SimulationActor(
        id="crossing_pedestrian",
        obstacle_class=ObstacleClass.PEDESTRIAN,
        x=52.0,
        y=3.2,
        yaw_rad=-1.5708, # heading towards negative y
        speed_mps=1.1,
        length_m=0.5,
        width_m=0.5,
        height_m=1.7,
        is_static=False,
        trajectory_waypoints=[(52.0, 3.2), (52.0, -2.8)]
    ))

    return env

class UnmarkedVillageRoadScenario(BaseScenario):
    """Complete Phase 1 Benchmark Scenario instance."""
    def __init__(self, duration_seconds: float = 20.0, dt: float = 0.05):
        super().__init__(name="Unmarked Indian Village Road (Phase 1)", duration_seconds=duration_seconds, dt=dt)

    def setup_environment(self):
        self.env = build_unmarked_village_environment()
