"""Simulation-specific data types."""
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class VehicleParameters:
    wheelbase_m: float = 2.7
    track_width_m: float = 1.6
    mass_kg: float = 1500.0
    moment_of_inertia_z: float = 2500.0
    cornering_stiffness_front: float = 80000.0
    cornering_stiffness_rear: float = 85000.0
    max_steer_rad: float = 0.785
    max_accel_mps2: float = 3.5
    max_decel_mps2: float = 7.0
