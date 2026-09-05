"""Closed-loop simulation coordinator."""
from interfaces import EgoVehicleState, ControlCommand, Pose3D, Point3D, Twist3D, Vector3D
from .vehicle_model import KinematicBicycleModel
from .environment import RoadEnvironment
from .sensor_sim import SyntheticSensorSuite

class ClosedLoopSimulator:
    """Runs synchronous time-stepped simulation for AV testing."""
    def __init__(self, env: RoadEnvironment = None, dt: float = 0.05):
        self.env = env or RoadEnvironment()
        self.dt = dt
        self.vehicle_model = KinematicBicycleModel()
        self.sensors = SyntheticSensorSuite(self.env)
        self.state = EgoVehicleState(
            timestamp=0.0,
            pose=Pose3D(position=Point3D(x=0.0, y=0.0, z=0.0), heading_rad=0.0),
            twist=Twist3D(speed_mps=0.0),
            acceleration=Vector3D(x=0.0, y=0.0, z=0.0)
        )

    def step(self, command: ControlCommand):
        self.state = self.vehicle_model.step(self.state, command, self.dt)
        raw_sensor = self.sensors.capture(self.state)
        return self.state, raw_sensor
