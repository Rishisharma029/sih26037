# SIH26037 — System Architecture Document
## Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads

---

## 1. System Vision & Problem Statement

Operating an autonomous vehicle (AV) in India requires fundamentally rethinking standard Western assumptions (HD maps, painted lane lines, homogeneous car-following, polite intersection yielding).

### Hallmark Indian Road Conditions:
1. **Absence of Lane Markings**: Faded, non-existent, or abruptly terminating road paint. The system must compute a **continuous free-space drivable corridor** dynamically.
2. **Extreme Heterogeneity**: Traffic consists of auto-rickshaws, motorcycles, pushcarts, cycles, heavy trucks, pedestrians, and stray animals sharing the corridor.
3. **Non-Lane-Respecting Actor Dynamics**: Vehicles swerve across imaginary lines, squeeze through 1.5m gaps, overtake from both sides, and negotiate intersections without traffic lights.
4. **Physical Surface Hazards**: Deep potholes, unpaved shoulders, open ditches, and unmarked speed humps.

---

## 2. Information Flow & Top-Level Architecture

```
[ SENSORS ] (Camera, LiDAR, Radar, RTK-GNSS/IMU)
     │
     ▼
[ PERCEPTION ] ──> FreeSpaceCorridor, TrackedObstacles (Auto/Bike/Cattle/Ped), RoadAnomalies
     │
     ▼
[ PREDICTION ] ──> Intent (Cut-in, Swerve, Cross, Stationary) + Multi-Modal Trajectories
     │
     ▼
[ PLANNING ] ──> Behavior FSM (Cruise/Nudge/Follow/Yield) + Frenet Lattice Trajectory
     │
     ▼
[ COLLISION AVOIDANCE ] ──> Vector TTC, Artificial Potential Field (APF), Control Barrier Filter (CBF), AEB
     │
     ▼
[ VEHICLE CONTROL ] ──> Stanley Lateral Steering, Longitudinal PID, Drive-By-Wire CAN
     │
     ▼
[ ACTUATION & SIMULATION ]
     │
     ▼
[ DASHBOARD & CAMPUSOS ] ──> WebSocket 5002, ROS2 Bridge 9090 (/vehicle/odom, /vehicle/battery_state)
```

---

## 3. Detailed Subsystem Specifications

### 3.1 Simulation (`simulation/`)
- Kinematic and 4-DOF dynamic bicycle models.
- Multi-agent traffic generator with stochastic behavior.
- Synthetic sensor suite producing LiDAR point clouds, radar Doppler targets, and camera 3D bounding boxes.

### 3.2 Perception (`perception/`)
- Drivable corridor extraction without lane lines: calculates lateral boundaries $(d_{left}(s), d_{right}(s))$.
- Multi-sensor Kalman filter fusion combining radar velocity, LiDAR spatial centroids, and camera semantics.
- Detection of 12 distinct Indian road classes (Auto-Rickshaw, Cattle, Pedestrian, Pushcart, Pothole, etc.).

### 3.3 Prediction (`prediction/`)
- Intention classifier evaluating lateral swerves, cut-ins, and crossing trajectories.
- Constant Turn Rate & Acceleration (CTRA) probabilistic trajectory forecasting over a 3.0s horizon.

### 3.4 Planning (`planning/`)
- Behavioral finite state machine: `CRUISE`, `FOLLOW`, `NUDGE_LEFT`, `NUDGE_RIGHT`, `YIELD`, `OVERTAKE`, `EMERGENCY_STOP`.
- Adaptive Frenet-frame lattice planner generating jerk-optimal quintic polynomial trajectories.

### 3.5 Collision Avoidance (`collision_avoidance/`)
- Vector Time-to-Collision (TTC) & Distance-at-Closest-Point-of-Approach (DCPA).
- Artificial Potential Fields (APF) exerting repulsive forces from obstacles and boundary ditches.
- Control Barrier Functions (CBF) guaranteeing forward invariance ($h(x) >= 0$).
- Autonomous Emergency Braking (AEB) supervisory interlock triggering if $TTC < 0.85$s.

### 3.6 Vehicle Control (`vehicle_control/`)
- Stanley lateral controller compensating for front-axle cross-track error.
- Longitudinal PID controller tracking target velocity and acceleration profiles.
- Drive-by-wire gateway encoding commands to CAN frames and ROS2 `ackermann_msgs/AckermannDrive`.

---

## 4. Latency Budget

- Perception & Fusion: 25 Hz (40 ms)
- Prediction: 20 Hz (50 ms)
- Planning: 15 Hz (66 ms)
- Collision Avoidance & CBF: 50 Hz (20 ms)
- Control & Actuation: 100 Hz (10 ms)
- End-to-End Reaction Time: < 80 ms
