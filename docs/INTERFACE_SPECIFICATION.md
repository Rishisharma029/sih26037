# SIH26037 — Interface Control Document (ICD)

All interfaces are strongly-typed Pydantic V2 models defined in `interfaces.py`.

| Subsystem | Produces | Consumes | Update Rate |
|---|---|---|---|
| `simulation` | `RawSensorFrame`, `EgoVehicleState` | `ControlCommand` | 50-100 Hz |
| `perception` | `PerceptionOutput` | `RawSensorFrame` | 25 Hz |
| `prediction` | `PredictionOutput` | `PerceptionOutput`, `EgoVehicleState` | 20 Hz |
| `planning` | `PlannedTrajectory` | `PredictionOutput`, `DrivableCorridor` | 15 Hz |
| `collision_avoidance` | `SafeTrajectory` | `PlannedTrajectory`, `PerceptionOutput` | 50 Hz |
| `vehicle_control` | `ControlCommand` | `SafeTrajectory`, `EgoVehicleState` | 100 Hz |
| `dashboard` | `VehicleTelemetry` | `EgoVehicleState`, `SafeTrajectory` | 20 Hz |

### Core Units Standard
- Length / Distance: meters (m)
- Time: seconds (s)
- Speed: meters per second (m/s)
- Angles: radians (-pi to +pi)
- Acceleration: meters per second squared (m/s^2)
- Percentage: 0.0 to 100.0%
