"""Frenet-frame Lattice Trajectory Generator using Quintic Polynomials."""
import math
from typing import List, Tuple
from interfaces import TrajectoryPoint, PlannedTrajectory, BehaviorMode, EgoVehicleState


class QuinticPolynomial:
    """1D quintic polynomial boundary value solver:

    s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    """

    def __init__(self, xs: float, vxs: float, axs: float, xe: float, vxe: float, axe: float, T: float):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = 0.5 * axs

        # Matrix formulation for [a3, a4, a5]
        # A * [a3, a4, a5]^T = B
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        b0 = xe - self.a0 - self.a1 * T - self.a2 * T2
        b1 = vxe - self.a1 - 2.0 * self.a2 * T
        b2 = axe - 2.0 * self.a2

        # Invert 3x3 matrix explicitly for high efficiency
        # [[T^3, T^4, T^5], [3T^2, 4T^3, 5T^4], [6T, 12T^2, 20T^3]]
        det = 2.0 * T5
        if abs(det) < 1e-6:
            self.a3 = 0.0
            self.a4 = 0.0
            self.a5 = 0.0
            return

        inv_00 = 10.0 / T3
        inv_01 = -4.0 / T2
        inv_02 = 0.5 / T

        inv_10 = -15.0 / T4
        inv_11 = 7.0 / T3
        inv_12 = -1.0 / T2

        inv_20 = 6.0 / T5
        inv_21 = -3.0 / T4
        inv_22 = 0.5 / T3

        self.a3 = inv_00 * b0 + inv_01 * b1 + inv_02 * b2
        self.a4 = inv_10 * b0 + inv_11 * b1 + inv_12 * b2
        self.a5 = inv_20 * b0 + inv_21 * b1 + inv_22 * b2

    def calc_point(self, t: float) -> float:
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        return self.a0 + self.a1 * t + self.a2 * t2 + self.a3 * t3 + self.a4 * t4 + self.a5 * t5

    def calc_first_derivative(self, t: float) -> float:
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        return self.a1 + 2.0 * self.a2 * t + 3.0 * self.a3 * t2 + 4.0 * self.a4 * t3 + 5.0 * self.a5 * t4

    def calc_second_derivative(self, t: float) -> float:
        t2 = t * t
        t3 = t2 * t
        return 2.0 * self.a2 + 6.0 * self.a3 * t + 12.0 * self.a4 * t2 + 20.0 * self.a5 * t3

    def calc_third_derivative(self, t: float) -> float:
        t2 = t * t
        return 6.0 * self.a3 + 24.0 * self.a4 * t + 60.0 * self.a5 * t2


class CandidateTrajectory:
    """Represents a generated candidate trajectory with kinematic metrics."""

    def __init__(
        self,
        candidate_id: str,
        target_d: float,
        target_v: float,
        horizon_t: float,
        waypoints: List[TrajectoryPoint]
    ):
        self.candidate_id = candidate_id
        self.target_d = target_d
        self.target_v = target_v
        self.horizon_t = horizon_t
        self.waypoints = waypoints

        # Pre-calculated features
        self.max_curvature = max((abs(p.curvature) for p in waypoints), default=0.0)
        self.max_lateral_accel = 0.0
        self.total_jerk = 0.0
        self.path_length_m = 0.0

        for i, p in enumerate(waypoints):
            lat_acc = (p.speed_mps ** 2) * abs(p.curvature)
            self.max_lateral_accel = max(self.max_lateral_accel, lat_acc)
            self.total_jerk += abs(p.jerk_mps3)
            if i > 0:
                prev = waypoints[i - 1]
                self.path_length_m += math.hypot(p.x - prev.x, p.y - prev.y)


class FrenetLatticeGenerator:
    """Samples a bundle of smooth spatio-temporal trajectories across lateral offsets and speeds."""

    def __init__(self, dt: float = 0.2):
        self.dt = dt
        # Lateral sample offsets (relative to road center or current ego lateral position)
        self.lateral_offsets = [-1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8]
        # Speed scaling factors relative to cruise speed
        self.speed_ratios = [0.0, 0.40, 0.70, 1.00]
        # Planning horizons (seconds)
        self.horizons = [2.5, 3.0]

    def sample_candidates(
        self,
        ego_state: EgoVehicleState,
        target_cruise_speed_mps: float
    ) -> List[CandidateTrajectory]:
        """Generate candidate trajectory lattice using quintic polynomial generation."""
        candidates: List[CandidateTrajectory] = []

        x0 = ego_state.pose.position.x
        y0 = ego_state.pose.position.y
        yaw0 = ego_state.pose.heading_rad
        v0 = ego_state.twist.speed_mps
        # Longitudinal velocity component approx equal to speed
        vx0 = max(0.1, v0 * math.cos(yaw0))
        vy0 = v0 * math.sin(yaw0)
        ax0 = 0.0
        ay0 = 0.0

        cand_idx = 0
        for T in self.horizons:
            steps = max(5, int(T / self.dt))
            for d_target in self.lateral_offsets:
                # Lateral polynomial y(t): from y0 -> d_target with 0 terminal lateral velocity/acc
                poly_lat = QuinticPolynomial(
                    xs=y0, vxs=vy0, axs=ay0,
                    xe=d_target, vxe=0.0, axe=0.0,
                    T=T
                )

                for spd_ratio in self.speed_ratios:
                    cand_idx += 1
                    v_end = target_cruise_speed_mps * spd_ratio

                    # Longitudinal polynomial x(t): target end speed v_end
                    # Approximate target distance over time T
                    avg_v = max(0.1, 0.5 * (vx0 + v_end))
                    x_end = x0 + avg_v * T

                    poly_lon = QuinticPolynomial(
                        xs=x0, vxs=vx0, axs=ax0,
                        xe=x_end, vxe=v_end, axe=0.0,
                        T=T
                    )

                    waypoints = []
                    for i in range(1, steps + 1):
                        t = i * self.dt
                        wx = poly_lon.calc_point(t)
                        wy = poly_lat.calc_point(t)

                        vx = poly_lon.calc_first_derivative(t)
                        vy = poly_lat.calc_first_derivative(t)
                        ax = poly_lon.calc_second_derivative(t)
                        ay = poly_lat.calc_second_derivative(t)
                        jx = poly_lon.calc_third_derivative(t)
                        jy = poly_lat.calc_third_derivative(t)

                        spd = math.hypot(vx, vy)
                        yaw = math.atan2(vy, vx) if spd > 0.05 else yaw0

                        # Path curvature kappa = (vx * ay - vy * ax) / (vx^2 + vy^2)^(1.5)
                        spd_sq = vx * vx + vy * vy
                        if spd_sq > 0.01:
                            kappa = (vx * ay - vy * ax) / (spd_sq ** 1.5)
                        else:
                            kappa = 0.0

                        total_acc = math.hypot(ax, ay)
                        total_jerk = math.hypot(jx, jy)

                        waypoints.append(TrajectoryPoint(
                            timestamp=ego_state.timestamp + t,
                            x=wx,
                            y=wy,
                            yaw_rad=yaw,
                            curvature=kappa,
                            speed_mps=max(0.0, spd),
                            acceleration_mps2=total_acc,
                            jerk_mps3=total_jerk
                        ))

                    cand = CandidateTrajectory(
                        candidate_id=f"cand_{cand_idx}_d{d_target:+0.1f}_v{v_end:0.1f}",
                        target_d=d_target,
                        target_v=v_end,
                        horizon_t=T,
                        waypoints=waypoints
                    )
                    candidates.append(cand)

        return candidates
