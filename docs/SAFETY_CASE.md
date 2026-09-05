# SIH26037 — Safety Case & SOTIF Architecture

### 1. Multi-Layer Safety Envelope
- **Layer 1: Behavioral Safety**: Speed capping, conservative headway, courtesy yielding.
- **Layer 2: Vector TTC Monitoring**: Time-to-Collision computed continuously across all tracked actors.
- **Layer 3: Control Barrier Functions (CBF)**: Mathematical guarantee ensuring vehicle state stays within the safe invariant set h(x) >= 0.
- **Layer 4: Hardware Emergency Brake (AEB)**: Supervisory watchdog interlock overrides vehicle actuators to 100% braking if TTC < 0.85s.

### 2. Fail-Safe Degraded Modes
- **Sensor Glare/Drop**: Conservative mode speed capped at 20 km/h.
- **Corridor Occlusion**: Automatic safe controlled stop (SAFE_STOP).
- **CampusOS Emergency SOS**: Immediate mechanical brake lockup.
