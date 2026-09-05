"""Abstract base class for benchmark scenarios."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from interfaces import EgoVehicleState, ControlCommand, SafeTrajectory
from simulation.simulator import ClosedLoopSimulator
from simulation.environment import RoadEnvironment

class BaseScenario(ABC):
    """Base harness executing a closed-loop scenario episode."""
    def __init__(self, name: str, duration_seconds: float = 10.0, dt: float = 0.05):
        self.name = name
        self.duration_seconds = duration_seconds
        self.dt = dt
        self.env = RoadEnvironment()
        self.setup_environment()
        self.simulator = ClosedLoopSimulator(env=self.env, dt=self.dt)
        self.history_states: List[EgoVehicleState] = []
        self.history_commands: List[ControlCommand] = []

    @abstractmethod
    def setup_environment(self):
        """Populate obstacles and road properties."""
        pass

    def run_step(self, command: ControlCommand):
        state, raw_sensor = self.simulator.step(command)
        self.history_states.append(state)
        self.history_commands.append(command)
        return state, raw_sensor
