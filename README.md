# SIH26037 — Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads

[![Phase](https://img.shields.io/badge/SIH2026-Phase%200%20Frozen-blue.svg)](docs/ARCHITECTURE.md)
[![Status](https://img.shields.io/badge/Interface%20Contract-Strictly%20Typed-success.svg)](docs/INTERFACE_SPECIFICATION.md)
[![Safety](https://img.shields.io/badge/Safety%20Case-ISO%2026262%20%2F%20SOTIF-orange.svg)](docs/SAFETY_CASE.md)

## 1. Executive Summary

Operating an autonomous vehicle (AV) in India represents one of the most demanding challenges in modern robotics. Unlike structured Western operational design domains (ODDs), Indian road environments exhibit:
- **Absence of Lane Markings**: Faded, non-existent, or abruptly terminating road paint, requiring free-space boundary estimation rather than lane-centering.
- **Heterogeneous & Non-Lane-Respecting Traffic**: Auto-rickshaws, motorcycles, pushcarts, cyclists, tractors, buses, and pedestrians sharing narrow corridors without lane discipline.
- **Erratic Dynamic Cut-Ins**: Sudden swerving, aggressive nudges, and opposing traffic driving on wrong sides.
- **Unstructured Road Hazards**: Stray cattle/dogs resting on the carriageway, open ditches, irregular shoulders, unmarked speed breakers, and deep potholes.

**SIH26037** is an end-to-end, high-assurance autonomous mobility stack engineered specifically for unstructured Indian road environments. It provides closed-loop perception fusion, stochastic intent prediction, adaptive Frenet-frame lattice planning, Control Barrier Function (CBF) collision avoidance, drive-by-wire actuation, and seamless integration with the **CampusOS Genova Platform**.

---

## 2. Directory Architecture

```
projects/sih26037/
├── README.md                      # Project documentation and quickstart guide
├── pyproject.toml                 # Package dependencies and configuration
├── interfaces.py                  # Core typed data contracts (Pydantic V2)
├── docs/
│   ├── ARCHITECTURE.md            # Deep system architecture & engineering specification
│   ├── INTERFACE_SPECIFICATION.md # Formal Interface Control Document (ICD)
│   ├── SCENARIOS.md               # 5 Hallmark Indian Road Benchmark Scenarios
│   └── SAFETY_CASE.md             # SOTIF & Fail-Operational Safety Architecture
├── simulation/
│   ├── simulator.py               # Closed-loop discrete-time multi-agent simulation engine
│   ├── vehicle_model.py           # Kinematic and dynamic bicycle models
│   ├── sensor_sim.py              # Synthetic LiDAR, Radar, Camera, and GNSS/IMU simulators
│   └── environment.py             # Road corridor, boundary geometry, and obstacle generator
├── perception/
│   ├── sensor_fusion.py           # Multi-sensor Extended Kalman Filter & Bayesian fusion
│   ├── boundary_detector.py       # Drivable free-space corridor & road boundary estimation
│   └── obstacle_detector.py       # Mixed-class 3D bounding box detection & tracking
├── prediction/
│   ├── intent_classifier.py       # Cut-in, crossing, swerving, and yielding classifier
│   └── trajectory_predictor.py    # CTRA & multi-modal probabilistic trajectory forecaster
├── planning/
│   ├── global_router.py           # Topological waypoint router without lane constraints
│   ├── behavior_planner.py        # Finite State Machine (Cruise, Nudge, Follow, Overtake, Yield, Emergency)
│   └── local_planner.py           # Adaptive Frenet-frame polynomial trajectory generator
├── collision_avoidance/
│   ├── ttc_calculator.py          # Vector Time-to-Collision & Distance-at-Closest-Point (DCPA)
│   ├── artificial_potential_field.py # Repulsive obstacle forces & drivable corridor potentials
│   ├── control_barrier_functions.py  # Discrete-time CBF safety filter guaranteeing invariance
│   └── emergency_brake.py         # Autonomous Emergency Braking (AEB) supervisory interlock
├── vehicle_control/
│   ├── lateral_controller.py      # Stanley Controller with yaw-damping & Pure Pursuit fallback
│   ├── longitudinal_controller.py # Feedforward + PID acceleration and velocity tracker
│   └── drive_by_wire_bridge.py    # CAN bus serialization & ROS2 AckermannDrive adapter
├── scenarios/
│   ├── scenario_base.py           # Benchmark harness base class
│   ├── scenario_unmarked_village.py    # Benchmark 1: Unmarked narrow village road with ditches
│   ├── scenario_unsignalled_junction.py # Benchmark 2: Unsignalled chaotic 4-way intersection
│   ├── scenario_highway_cutin.py       # Benchmark 3: Auto-rickshaw cut-in & highway merge
│   ├── scenario_dense_market.py        # Benchmark 4: Dense pedestrian & pushcart swarming
│   └── scenario_cattle_crossing.py     # Benchmark 5: Stray cattle roadblock on blind bend
├── datasets/
│   ├── schemas.py                 # Indian Driving Dataset (IDD) inspired schemas
│   └── loaders.py                 # Telemetry log replay & synthetic scenario loader
├── evaluation/
│   ├── metrics.py                 # Safety (Min TTC, collisions), comfort (jerk), progress
│   ├── evaluator.py               # Automated test harness & benchmark scorecard runner
│   └── reporter.py                # Formatted markdown/JSON scorecard generator
├── dashboard/
│   ├── server.py                  # Standalone FastAPI + WebSocket server (Port 5002)
│   └── bridge.py                  # ROS2 WebSocket bridge client (Port 9090)
└── tests/
    ├── test_interfaces.py         # Strict type & interface contract validation suite
    ├── test_perception.py         # Perception fusion and boundary detection tests
    ├── test_prediction.py         # Motion intent & trajectory prediction tests
    ├── test_planning.py           # Behavioral and lattice trajectory planner tests
    ├── test_collision_avoidance.py# TTC, APF, CBF, and AEB safety envelope tests
    ├── test_control.py            # Stanley & PID tracking stability tests
    └── test_scenarios.py          # End-to-end execution of all 5 benchmark scenarios
```

---

## 3. Quickstart & Verification

```bash
# Run interface and subsystem test suite
python -m pytest tests/ -v

# Launch the standalone telemetry service on port 5002
python -m dashboard.server
```
