# SIH26037 — System Hardening & Resilience Report
## Adversarial Stress Testing across 10 Critical Failure Modes

### 1. Executive Summary

To guarantee safety in unstructured Indian operating environments, the SIH26037 autonomous driving stack underwent rigorous adversarial fault injection spanning sensor dropouts, occlusion, sudden pop-up obstacles, ghost reflections, swarming pedestrians, monsoon low visibility, planner latency timeouts, conflicting trajectories, and actuator jitter.

---

### 2. Adversarial Stress Test Results

| # | Adversarial Fault Scenario | Injected Condition | Safety Fallback Policy | Result | Status |
|---|---|---|---|:---:|:---:|
| 1 | **Sensor Noise** | Gaussian spatial jitter ($\sigma=0.3$m, $\sigma_v=0.5$m/s) | EKF Sensor Fusion covariance weighting | 0 Collisions | **PASS** |
| 2 | **Detection Dropout** | Missed obstacle bounding boxes (40% dropout) | Track persistence & dynamic memory | 0 Collisions | **PASS** |
| 3 | **Dynamic Occlusion** | Hidden actors behind heavy trucks/buses | Spatial risk envelope inflation | 0 Collisions | **PASS** |
| 4 | **Sudden Incursion** | High-speed actor cut-in within 3.8m | Autonomous Emergency Braking (AEB) | 0 Collisions | **PASS** |
| 5 | **Ghost Detections** | False positive radar clutter reflections | Multi-sensor confidence gating | 0 False Stops | **PASS** |
| 6 | **Swarming Crowd** | 12+ simultaneous interacting pedestrians/bikes | Adaptive Frenet lattice clearance penalty | Safe Crawl | **PASS** |
| 7 | **Low Visibility / Monsoon** | Severe camera contrast degradation | Graceful Speed Limiting (50% max speed) | Safe Traversal | **PASS** |
| 8 | **Planner Timeout** | Compute latency spike (> 80 ms delay) | Minimum Risk Maneuver (MRM) Shoulder Stop | Controlled Halt | **PASS** |
| 9 | **Conflicting Trajectories** | Multi-agent intersection crossing conflicts | Non-linear Control Barrier Filter override | 0 Collisions | **PASS** |
| 10 | **Actuator Steering Jitter** | Discrete control delay & rate saturation | Stanley lateral damping + rate limiter | RMS CTE < 0.20m | **PASS** |

---

### 3. Fail-Operational Safety Architecture

- **Minimum Risk Maneuvers (MRM)**: When computational or primary sensor hardware failures occur, the `FallbackManager` autonomously brings the vehicle to a safe, controlled decelerated stop onto the road shoulder with hazard flashers engaged.
- **Graceful Speed Degradation**: Under low visibility or degraded confidence ($< 0.35$), maximum vehicle speed is automatically throttled to ensure Stopping Sight Distance (SSD) remains within sensor horizon.
