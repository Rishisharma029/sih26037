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
# 1. Run the entire 66-test verification suite
python -m pytest tests/ -v

# 2. Run the 20-Episode Closed-Loop Benchmark Matrix
python -m scenarios.benchmark_suite

# 3. Train/Evaluate the IDD Perception Detector and Segmenter
python -m perception.train_detector --epochs 5
python -m perception.train_segmenter --epochs 5

# 4. Launch the standalone AV Telemetry & Simulation Server
python -m dashboard.server
```

---

## 4. Benchmark Performance Matrix (20 Closed-Loop Episodes)

All 5 hallmark Indian scenarios evaluated across `EASY`, `MEDIUM`, `HARD`, and `EXTREME` difficulties:

| Benchmark Scenario | Difficulty | Safety Pass | Min TTC (s) | Min Distance (m) | Max Lat Accel (m/s²) | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `01_village_road` | EASY | 100% | > 10.0 | 2.50 | 0.00 | **PASS** |
| `01_village_road` | MEDIUM | 100% | > 10.0 | 1.85 | 0.42 | **PASS** |
| `01_village_road` | HARD | 100% | 4.82 | 1.60 | 0.88 | **PASS** |
| `01_village_road` | EXTREME | 100% | 3.12 | 1.35 | 1.45 | **PASS** |
| `02_uncontrolled_intersection` | EASY | 100% | > 10.0 | 3.20 | 0.15 | **PASS** |
| `02_uncontrolled_intersection` | MEDIUM | 100% | 5.10 | 2.10 | 0.65 | **PASS** |
| `02_uncontrolled_intersection` | HARD | 100% | 2.95 | 1.40 | 1.20 | **PASS** |
| `02_uncontrolled_intersection` | EXTREME | 100% | 1.85 | 1.25 | 1.82 | **PASS** |
| `03_highway_merge` | EASY | 100% | > 10.0 | 4.50 | 0.10 | **PASS** |
| `03_highway_merge` | MEDIUM | 100% | 6.20 | 2.80 | 0.55 | **PASS** |
| `03_highway_merge` | HARD | 100% | 2.40 | 1.50 | 1.30 | **PASS** |
| `03_highway_merge` | EXTREME | 100% | 1.65 | 1.20 | 1.95 | **PASS** |
| `04_dense_market` | EASY | 100% | 8.50 | 2.40 | 0.20 | **PASS** |
| `04_dense_market` | MEDIUM | 100% | 4.10 | 1.80 | 0.75 | **PASS** |
| `04_dense_market` | HARD | 100% | 2.20 | 1.30 | 1.40 | **PASS** |
| `04_dense_market` | EXTREME | 100% | 1.45 | 1.15 | 1.90 | **PASS** |
| `05_cattle_crossing` | EASY | 100% | > 10.0 | 3.80 | 0.10 | **PASS** |
| `05_cattle_crossing` | MEDIUM | 100% | 5.80 | 2.20 | 0.60 | **PASS** |
| `05_cattle_crossing` | HARD | 100% | 2.60 | 1.45 | 1.10 | **PASS** |
| `05_cattle_crossing` | EXTREME | 100% | 1.50 | 1.20 | 1.65 | **PASS** |

- **Zero Collisions** across all 20 episodes.
- Full details documented in [`docs/BENCHMARK_SCORECARD.md`](docs/BENCHMARK_SCORECARD.md).

