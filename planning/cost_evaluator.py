"""Multi-Objective Cost Evaluator for Candidate Trajectories on Unstructured Roads."""
import math
from typing import Dict, List, Optional, Tuple
from interfaces import (
    PerceptionOutput, PredictionOutput, FreeSpaceCorridor,
    ObstacleClass, RoadAnomaly
)
from .frenet_lattice import CandidateTrajectory


class TrajectoryCostScore:
    """Detailed score breakdown for a candidate trajectory."""

    def __init__(
        self,
        candidate_id: str,
        total_cost: float,
        is_feasible: bool,
        cost_breakdown: Dict[str, float],
        min_clearance_m: float,
        min_boundary_margin_m: float
    ):
        self.candidate_id = candidate_id
        self.total_cost = total_cost
        self.is_feasible = is_feasible
        self.cost_breakdown = cost_breakdown
        self.min_clearance_m = min_clearance_m
        self.min_boundary_margin_m = min_boundary_margin_m


class TrajectoryCostEvaluator:
    """Evaluates candidate trajectories against safety, clearance, traversability,

    progress, curvature, comfort, and dynamic uncertainty.
    """

    def __init__(
        self,
        w_safety: float = 300.0,
        w_clearance: float = 80.0,
        w_traversability: float = 350.0,
        w_progress: float = 45.0,
        w_path_length: float = 5.0,
        w_curvature: float = 20.0,
        w_comfort: float = 15.0,
        w_uncertainty: float = 90.0,
        max_lateral_accel: float = 3.2,
        max_curvature: float = 0.35,
        safety_bubble_m: float = 2.0
    ):
        self.w_safety = w_safety
        self.w_clearance = w_clearance
        self.w_traversability = w_traversability
        self.w_progress = w_progress
        self.w_path_length = w_path_length
        self.w_curvature = w_curvature
        self.w_comfort = w_comfort
        self.w_uncertainty = w_uncertainty

        self.max_lateral_accel = max_lateral_accel
        self.max_curvature = max_curvature
        self.safety_bubble_m = safety_bubble_m

    def evaluate(
        self,
        candidate: CandidateTrajectory,
        perception: PerceptionOutput,
        prediction: PredictionOutput,
        target_cruise_speed_mps: float
    ) -> TrajectoryCostScore:
        """Computes comprehensive multi-objective cost score for a single candidate."""
        default_breakdown = {
            "safety": 0.0,
            "clearance": 0.0,
            "traversability": 0.0,
            "progress": 0.0,
            "path_length": 0.0,
            "curvature": 0.0,
            "comfort": 0.0,
            "uncertainty": 0.0,
        }

        # 1. Feasibility check on dynamic constraints
        if candidate.max_curvature > self.max_curvature:
            default_breakdown["curvature"] = 99999.0
            return TrajectoryCostScore(
                candidate_id=candidate.candidate_id,
                total_cost=99999.0,
                is_feasible=False,
                cost_breakdown=default_breakdown,
                min_clearance_m=0.0,
                min_boundary_margin_m=0.0
            )

        if candidate.max_lateral_accel > self.max_lateral_accel:
            default_breakdown["comfort"] = 99999.0
            return TrajectoryCostScore(
                candidate_id=candidate.candidate_id,
                total_cost=99999.0,
                is_feasible=False,
                cost_breakdown=default_breakdown,
                min_clearance_m=0.0,
                min_boundary_margin_m=0.0
            )

        # 2. Road Traversability & Corridor Boundaries
        cost_traversability = 0.0
        min_boundary_margin = 999.0

        corridor_half_w = max(2.5, perception.drivable_corridor.average_width_m * 0.5)

        for pt in candidate.waypoints:
            dist_to_left = corridor_half_w - pt.y
            dist_to_right = pt.y - (-corridor_half_w)
            margin = min(dist_to_left, dist_to_right)
            if margin < min_boundary_margin:
                min_boundary_margin = margin

            if margin < 0.30:
                cost_traversability += (0.30 - margin) ** 2 * 100.0

        if min_boundary_margin < 0.0:
            default_breakdown["traversability"] = 88888.0
            return TrajectoryCostScore(
                candidate_id=candidate.candidate_id,
                total_cost=88888.0,
                is_feasible=False,
                cost_breakdown=default_breakdown,
                min_clearance_m=0.0,
                min_boundary_margin_m=min_boundary_margin
            )

        # 3. Dynamic Safety & Collision Risk against Multi-Modal Predictions
        cost_safety = 0.0
        cost_uncertainty = 0.0
        min_clearance = 999.0

        for agent in prediction.agents:
            for traj_mode in agent.trajectories:
                p_mode = traj_mode.probability
                for step_idx, ego_pt in enumerate(candidate.waypoints):
                    # Check against matching timestep
                    step_to_check = min(step_idx, len(traj_mode.waypoints) - 1)
                    agent_pt = traj_mode.waypoints[step_to_check]

                    dx = ego_pt.x - agent_pt.position.x
                    dy = ego_pt.y - agent_pt.position.y
                    dist = math.hypot(dx, dy)

                    if dist < min_clearance:
                        min_clearance = dist

                    # Safety bubble intrusion cost
                    if dist < self.safety_bubble_m:
                        intrusion = self.safety_bubble_m - dist
                        cost_safety += p_mode * (intrusion ** 2) * 60.0

                    # Proximity exponential hazard
                    if dist < 4.5:
                        cost_safety += p_mode * math.exp(-dist / 1.5) * 6.0

                    # Spatial uncertainty overlap (sigma_x, sigma_y)
                    sig_x = max(0.1, agent_pt.sigma_x)
                    sig_y = max(0.1, agent_pt.sigma_y)
                    norm_sq = (dx / sig_x) ** 2 + (dy / sig_y) ** 2
                    if norm_sq < 9.0:
                        cost_uncertainty += p_mode * math.exp(-0.5 * norm_sq) * 10.0

            # Direct line-of-sight blockage check against static obstacles
            # If candidate trajectory terminates directly behind a static obstacle at low clearance
            if agent.primary_intent.value == "STATIONARY":
                for ego_pt in candidate.waypoints:
                    if ego_pt.x < agent.trajectories[0].waypoints[0].position.x:
                        lat_d = abs(ego_pt.y - agent.trajectories[0].waypoints[0].position.y)
                        long_d = agent.trajectories[0].waypoints[0].position.x - ego_pt.x
                        if lat_d < 1.0 and long_d < 6.0:
                            cost_safety += 35.0 * (1.0 - long_d / 6.0)

        # 4. Obstacle & Road Anomaly Clearance
        cost_clearance = 0.0
        for anom in perception.anomalies:
            for ego_pt in candidate.waypoints:
                adx = ego_pt.x - anom.position.x
                ady = ego_pt.y - anom.position.y
                adist = math.hypot(adx, ady)
                if adist < (anom.radius_m + 0.6):
                    cost_clearance += 20.0 * (1.0 - adist / (anom.radius_m + 0.6))

        if min_clearance < 4.0:
            cost_clearance += (4.0 - min_clearance) ** 2 * 3.0

        # Hard collision filter: if trajectory gets closer than 0.85m to any obstacle
        if min_clearance < 0.85:
            breakdown_collision = {
                "safety": 77777.0,
                "clearance": round(self.w_clearance * cost_clearance, 2),
                "traversability": round(self.w_traversability * cost_traversability, 2),
                "progress": 0.0,
                "path_length": 0.0,
                "curvature": round(self.w_curvature * candidate.max_curvature * 10.0, 2),
                "comfort": 0.0,
                "uncertainty": round(self.w_uncertainty * cost_uncertainty, 2),
            }
            return TrajectoryCostScore(
                candidate_id=candidate.candidate_id,
                total_cost=77777.0,
                is_feasible=False,
                cost_breakdown=breakdown_collision,
                min_clearance_m=min_clearance,
                min_boundary_margin_m=min_boundary_margin
            )

        # 5. Progress & Target Speed Efficiency
        end_speed = candidate.target_v
        speed_dev = (target_cruise_speed_mps - end_speed) / max(1.0, target_cruise_speed_mps)
        cost_progress = (speed_dev ** 2) * 20.0

        # Reward forward distance gained
        dx_progress = candidate.waypoints[-1].x - candidate.waypoints[0].x
        cost_progress -= dx_progress * 0.25

        # 6. Path Length & Curvature Smoothness
        cost_path_length = abs(candidate.path_length_m - dx_progress) * 1.5
        cost_curvature = candidate.max_curvature * 10.0

        # 7. Passenger Comfort (Jerk & Lateral Acceleration)
        cost_comfort = candidate.max_lateral_accel * 3.0 + (candidate.total_jerk / len(candidate.waypoints)) * 0.3

        # Weighted Total Score
        total_score = (
            self.w_safety * cost_safety +
            self.w_clearance * cost_clearance +
            self.w_traversability * cost_traversability +
            self.w_progress * cost_progress +
            self.w_path_length * cost_path_length +
            self.w_curvature * cost_curvature +
            self.w_comfort * cost_comfort +
            self.w_uncertainty * cost_uncertainty
        )

        breakdown = {
            "safety": round(self.w_safety * cost_safety, 2),
            "clearance": round(self.w_clearance * cost_clearance, 2),
            "traversability": round(self.w_traversability * cost_traversability, 2),
            "progress": round(self.w_progress * cost_progress, 2),
            "path_length": round(self.w_path_length * cost_path_length, 2),
            "curvature": round(self.w_curvature * cost_curvature, 2),
            "comfort": round(self.w_comfort * cost_comfort, 2),
            "uncertainty": round(self.w_uncertainty * cost_uncertainty, 2),
        }

        return TrajectoryCostScore(
            candidate_id=candidate.candidate_id,
            total_cost=round(total_score, 2),
            is_feasible=True,
            cost_breakdown=breakdown,
            min_clearance_m=round(min_clearance, 2),
            min_boundary_margin_m=round(min_boundary_margin, 2)
        )
