# SIH26037 — Autonomous Decision Audit Log
## Hallmark Scenario 5: Extreme Cattle Crossing

This audit record captures real-time explainable decisions made by the autonomous driving stack.

---

### EVENT #1 (t = 0.00s | Speed = 0.0 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
999.00 s (Distance: 12.0 m, Side: FRONT)

Current path:
SAFE

Candidates Evaluated:
56

Selected:
traj_1_cand_16_d+0.0_v6.0 [CRUISE]

Safety Action:
NONE

Reason:
Optimal nominal cruise velocity + minimal cross-track error + zero obstacle conflict
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #2 (t = 0.05s | Speed = 0.6 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
67.34 s (Distance: 12.0 m, Side: FRONT)

Current path:
SAFE

Candidates Evaluated:
56

Selected:
traj_1_cand_16_d+0.0_v6.0 [CRUISE]

Safety Action:
NONE

Reason:
Optimal nominal cruise velocity + minimal cross-track error + zero obstacle conflict
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #3 (t = 0.10s | Speed = 1.2 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
34.78 s (Distance: 12.0 m, Side: FRONT)

Current path:
SAFE

Candidates Evaluated:
56

Selected:
traj_1_cand_16_d+0.0_v6.0 [CRUISE]

Safety Action:
NONE

Reason:
Optimal nominal cruise velocity + minimal cross-track error + zero obstacle conflict
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #4 (t = 0.15s | Speed = 1.8 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
24.07 s (Distance: 12.0 m, Side: FRONT)

Current path:
SAFE

Candidates Evaluated:
56

Selected:
traj_1_cand_16_d+0.0_v6.0 [CRUISE]

Safety Action:
NONE

Reason:
Optimal nominal cruise velocity + minimal cross-track error + zero obstacle conflict
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #5 (t = 0.20s | Speed = 2.3 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
18.72 s (Distance: 11.9 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_2_cand_43_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #6 (t = 0.25s | Speed = 2.6 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
16.73 s (Distance: 11.9 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_2_cand_43_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #7 (t = 0.30s | Speed = 2.8 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
15.18 s (Distance: 11.9 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_2_cand_43_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #8 (t = 0.35s | Speed = 3.0 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
13.98 s (Distance: 11.8 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_2_cand_43_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #9 (t = 0.40s | Speed = 3.2 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
13.02 s (Distance: 11.8 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_3_cand_15_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #10 (t = 0.45s | Speed = 3.8 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
10.99 s (Distance: 11.7 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_3_cand_15_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #11 (t = 0.50s | Speed = 4.1 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
10.25 s (Distance: 11.7 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_3_cand_15_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #12 (t = 0.55s | Speed = 4.4 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
9.53 s (Distance: 11.6 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_3_cand_15_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #13 (t = 0.60s | Speed = 4.6 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
8.95 s (Distance: 11.6 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_4_cand_15_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #14 (t = 0.65s | Speed = 5.1 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
8.03 s (Distance: 11.5 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_4_cand_15_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #15 (t = 0.70s | Speed = 5.4 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
7.63 s (Distance: 11.4 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_4_cand_15_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #16 (t = 0.75s | Speed = 5.6 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
7.24 s (Distance: 11.3 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_4_cand_15_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #17 (t = 0.80s | Speed = 5.9 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
6.91 s (Distance: 11.3 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_5_cand_15_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #18 (t = 0.85s | Speed = 6.3 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
6.37 s (Distance: 11.2 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_5_cand_15_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #19 (t = 0.90s | Speed = 6.5 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
LOW

TTC:
6.12 s (Distance: 11.1 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_5_cand_15_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #20 (t = 0.95s | Speed = 6.7 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
MEDIUM

TTC:
5.87 s (Distance: 11.0 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_5_cand_15_d+0.0_v4.2 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #21 (t = 1.00s | Speed = 6.9 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
MEDIUM

TTC:
5.65 s (Distance: 10.9 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_6_cand_14_d+0.0_v2.4 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #22 (t = 1.05s | Speed = 7.0 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
MEDIUM

TTC:
5.57 s (Distance: 10.8 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_6_cand_14_d+0.0_v2.4 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #23 (t = 1.10s | Speed = 7.0 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
MEDIUM

TTC:
5.48 s (Distance: 10.7 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_6_cand_14_d+0.0_v2.4 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #24 (t = 1.15s | Speed = 7.1 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
MEDIUM

TTC:
5.40 s (Distance: 10.6 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_6_cand_14_d+0.0_v2.4 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #25 (t = 1.20s | Speed = 7.1 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
MEDIUM

TTC:
5.31 s (Distance: 10.5 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_7_cand_14_d+0.0_v2.4 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #26 (t = 1.25s | Speed = 7.2 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
MEDIUM

TTC:
5.21 s (Distance: 10.4 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_7_cand_14_d+0.0_v2.4 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #27 (t = 1.30s | Speed = 7.2 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
MEDIUM

TTC:
5.14 s (Distance: 10.3 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_7_cand_14_d+0.0_v2.4 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #28 (t = 1.35s | Speed = 7.3 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
MEDIUM

TTC:
5.06 s (Distance: 10.2 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_7_cand_14_d+0.0_v2.4 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #29 (t = 1.40s | Speed = 7.3 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
MEDIUM

TTC:
4.98 s (Distance: 10.1 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_8_cand_14_d+0.0_v2.4 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
### EVENT #30 (t = 1.45s | Speed = 7.4 km/h)

```
Hazard:
Stray cattle stationary / crossing road

Risk:
MEDIUM

TTC:
4.89 s (Distance: 10.0 m, Side: FRONT)

Current path:
OBSTRUCTED_FOLLOWING

Candidates Evaluated:
56

Selected:
traj_8_cand_14_d+0.0_v2.4 [FOLLOW]

Safety Action:
ADAPTIVE_CRUISE_SLOWDOWN

Reason:
Maintains safe headway behind leading CATTLE_ANIMAL with zero risk
```

- **Safety Margin**: 2.50 m
- **Curvature Rating**: ACCEPTABLE | **Lateral Deviation**: LOW

---
