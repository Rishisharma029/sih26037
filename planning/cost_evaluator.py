"""Multi-Objective Cost Evaluator for Candidate Trajectories on Unstructured Roads.

Evaluates candidates against the 7 core objectives:
  - 40% Collision Safety
  - 20% Obstacle Clearance
  - 15% Road Traversability (Hard boundary invariant)
  - 10% Progress & Target Speed
  - 5%  Path Smoothness
  - 5%  Vehicle Dynamics & Comfort
  - 5%  Dynamic Uncertainty
"""
import math
from typing import Dict, List, Optional, Tuple
from interfaces import (
    PerceptionOutput, PredictionOutput, FreeSpaceCorridor,
    ObstacleClass, RoadAnomaly
)
from .frenet_lattice import CandidateTrajectory


class TrajectoryCostScore:
    """Detailed score breakdown and explainable decision tagging for a candidate trajectory."""

    def __init__(
        self,
        candidate_id: str,
        total_cost: float,
        is_feasible: bool,
        cost_breakdown: Dict[str, float],
        min_clearance_m: float = 0.0,
        min_boundary_margin_m: float = 0.0,
        label: str = "",
        status_tag: str = "SAFE",
        explanation: str = ""
    ):
        self.candidate_id = candidate_id
        self.total_cost = total_cost
        self.is_feasible = is_feasible
        self.cost_breakdown = cost_breakdown
        self.min_clearance_m = min_clearance_m
        self.min_boundary_margin_m = min_boundary_margin_m
        self.label = label or candidate_id
        self.status_tag = status_tag
        self.explanation = explanation

    def to_dict(self) -> Dict:
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "total_cost": round(self.total_cost, 2),
            "is_feasible": self.is_feasible,
            "status_tag": self.status_tag,
            "cost_breakdown": {k: round(v, 2) for k, v in self.cost_breakdown.items()},
            "min_clearance_m": round(self.min_clearance_m, 2),
            "min_boundary_margin_m": round(self.min_boundary_margin_m, 2),
            "explanation": self.explanation
        }


class TrajectoryCostEvaluator:
    """Evaluates candidate trajectories against 7 normalized objective functions
    with hard feasibility pruning and natural-language explainability.
    """

    def __init__(
        self,
        w_collision: float = 0.40,
        w_clearance: float = 0.20,
        w_traversability: float = 0.15,
        w_progress: float = 0.10,
        w_smoothness: float = 0.05,
        w_dynamics: float = 0.05,
        w_uncertainty: float = 0.05,
        max_lateral_accel: float = 3.2,
        max_curvature: float = 0.35,
        safety_bubble_m: float = 2.0
    ):
        # Weights normalized to sum = 1.00
        total_w = (
            w_collision + w_clearance + w_traversability +
            w_progress + w_smoothness + w_dynamics + w_uncertainty
        )
        self.w_collision = w_collision / total_w
        self.w_clearance = w_clearance / total_w
        self.w_traversability = w_traversability / total_w
        self.w_progress = w_progress / total_w
        self.w_smoothness = w_smoothness / total_w
        self.w_dynamics = w_dynamics / total_w
        self.w_uncertainty = w_uncertainty / total_w

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
        """Computes comprehensive 7-objective cost score for a single candidate."""
        default_breakdown = {
            "collision_safety": 0.0,
            "obstacle_clearance": 0.0,
            "road_traversability": 0.0,
            "traversability": 0.0,
            "progress": 0.0,
            "path_smoothness": 0.0,
            "vehicle_dynamics": 0.0,
            "curvature": 0.0,
            "comfort": 0.0,
            "uncertainty": 0.0,
        }

        # 1. Hard Dynamic Feasibility Check: Curvature
        if candidate.max_curvature > self.max_curvature:
            default_breakdown["vehicle_dynamics"] = 100.0
            default_breakdown["curvature"] = 99999.0
            return TrajectoryCostScore(
                candidate_id=candidate.candidate_id,
                label=candidate.label,
                total_cost=99999.0,
                is_feasible=False,
                status_tag="EXCESSIVE_CURVATURE",
                cost_breakdown=default_breakdown,
                min_clearance_m=0.0,
                min_boundary_margin_m=0.0,
                explanation=f"Curvature {candidate.max_curvature:.2f} rad/m exceeds steering limit {self.max_curvature:.2f} rad/m"
            )

        # 2. Hard Dynamic Feasibility Check: Lateral Acceleration / Passenger Comfort
        if candidate.max_lateral_accel > self.max_lateral_accel:
            default_breakdown["vehicle_dynamics"] = 100.0
            default_breakdown["comfort"] = 99999.0
            return TrajectoryCostScore(
                candidate_id=candidate.candidate_id,
                label=candidate.label,
                total_cost=99999.0,
                is_feasible=False,
                status_tag="COMFORT_VIOLATION",
                cost_breakdown=default_breakdown,
                min_clearance_m=0.0,
                min_boundary_margin_m=0.0,
                explanation=f"Lateral acceleration {candidate.max_lateral_accel:.2f} m/s² exceeds passenger comfort threshold {self.max_lateral_accel:.2f} m/s²"
            )

        # 3. Road Traversability & Surface Anomaly Hard Invariant
        cost_traversability = 0.0
        min_boundary_margin = 999.0
        vehicle_half_width = 0.90
        safety_buffer = 0.15  # Hard safety margin to road shoulder / ditch

        corridor_pts = perception.drivable_corridor.boundary_points
        fallback_half_w = max(2.1, perception.drivable_corridor.average_width_m * 0.5)

        for pt in candidate.waypoints:
            d_left = fallback_half_w
            d_right = -fallback_half_w

            if corridor_pts:
                # Find closest boundary station
                ref_x = candidate.waypoints[0].x
                closest_bp = min(corridor_pts, key=lambda bp: abs(bp.s - (pt.x - ref_x)))
                d_left = closest_bp.d_left
                d_right = closest_bp.d_right

            # Physical clearance from outer vehicle contour to left and right road edges
            margin_left = d_left - (pt.y + vehicle_half_width)
            margin_right = (pt.y - vehicle_half_width) - d_right
            margin = min(margin_left, margin_right)

            if margin < min_boundary_margin:
                min_boundary_margin = margin

            # Traversability penalty if running close to road edge (< 0.6m margin)
            if margin < 0.60:
                cost_traversability += max(0.0, (0.60 - margin) / 0.60) * 15.0

        # Hard safety invariant: Breaching road ditch boundary
        if min_boundary_margin < safety_buffer:
            default_breakdown["road_traversability"] = 99999.0
            default_breakdown["traversability"] = 99999.0
            return TrajectoryCostScore(
                candidate_id=candidate.candidate_id,
                label=candidate.label,
                total_cost=99999.0,
                is_feasible=False,
                status_tag="DITCH_BREACH",
                cost_breakdown=default_breakdown,
                min_clearance_m=0.0,
                min_boundary_margin_m=round(min_boundary_margin, 3),
                explanation=f"Ditch clearance {min_boundary_margin:.2f}m violates safety invariant margin ({safety_buffer:.2f}m)"
            )

        # 3b. Road Surface Anomalies (Potholes, Waterlogging, Gravel, Speed Bumps)
        for anom in perception.anomalies:
            is_passable = getattr(anom, "is_passable", True)
            max_safe_v = getattr(anom, "max_safe_speed_mps", 0.0 if not is_passable else 5.0)
            t_score = getattr(anom, "traversability_score", 0.5)
            anom_type = getattr(anom, "anomaly_type", "POTHOLE").upper()
            anom_desc = getattr(anom, "description", anom_type)

            for pt in candidate.waypoints:
                adx = pt.x - anom.position.x
                ady = pt.y - anom.position.y
                adist = math.hypot(adx, ady)
                collision_radius = anom.radius_m + 0.45  # Vehicle wheel track envelope

                if adist < collision_radius:
                    # Non-passable hazard (e.g. Deep Pothole, Flood/Waterlogging, Blocked debris)
                    if not is_passable and candidate.target_v > max_safe_v:
                        default_breakdown["road_traversability"] = 99999.0
                        default_breakdown["traversability"] = 99999.0
                        depth_txt = f", depth {abs(anom.depth_or_height_m)*100:.0f}cm" if anom.depth_or_height_m < 0 else ""
                        return TrajectoryCostScore(
                            candidate_id=candidate.candidate_id,
                            label=candidate.label,
                            total_cost=99999.0,
                            is_feasible=False,
                            status_tag="UNSAFE_TRAVERSABILITY",
                            cost_breakdown=default_breakdown,
                            min_clearance_m=round(adist, 2),
                            min_boundary_margin_m=round(min_boundary_margin, 3),
                            explanation=f"Technically open, but unsafe to drive through: {anom_desc}{depth_txt} (speed {candidate.target_v:.1f}m/s > {max_safe_v:.1f}m/s)"
                        )

                    # Passable with speed penalty (e.g. Speed Bump, Gravel, Degraded)
                    if candidate.target_v > max_safe_v:
                        overspeed = candidate.target_v - max_safe_v
                        cost_traversability += overspeed * 18.0
                    cost_traversability += (1.0 - t_score) * 30.0

        cost_traversability = min(100.0, cost_traversability)

        # 4. Spatio-temporal Collision Safety & Prediction Risk
        cost_collision = 0.0
        cost_uncertainty = 0.0
        min_clearance = 999.0
        collision_threat_agent = None

        for agent in prediction.agents:
            obs_cls_name = getattr(agent.obstacle_class, "value", str(agent.obstacle_class)) if hasattr(agent, "obstacle_class") else "OBSTACLE"
            for traj_mode in agent.trajectories:
                p_mode = traj_mode.probability
                for step_idx, ego_pt in enumerate(candidate.waypoints):
                    step_to_check = min(step_idx, len(traj_mode.waypoints) - 1)
                    agent_pt = traj_mode.waypoints[step_to_check]

                    dx = ego_pt.x - agent_pt.position.x
                    dy = ego_pt.y - agent_pt.position.y
                    dist = math.hypot(dx, dy)

                    if dist < min_clearance:
                        min_clearance = dist
                        if dist < 2.0:
                            collision_threat_agent = obs_cls_name

                    # Proximity penalty
                    if dist < self.safety_bubble_m:
                        intrusion = (self.safety_bubble_m - dist) / self.safety_bubble_m
                        cost_collision += p_mode * (intrusion ** 2) * 50.0

                    if dist < 4.0:
                        cost_collision += p_mode * math.exp(-dist / 1.5) * 15.0

                    # Dynamic uncertainty overlap against 2D Gaussian prediction distribution
                    sig_x = max(0.2, getattr(agent_pt, "sigma_x", 0.3))
                    sig_y = max(0.2, getattr(agent_pt, "sigma_y", 0.3))
                    norm_sq = (dx / sig_x) ** 2 + (dy / sig_y) ** 2
                    if norm_sq < 9.0:
                        cost_uncertainty += p_mode * math.exp(-0.5 * norm_sq) * 20.0

            # Static obstacle alignment check
            if agent.primary_intent.value == "STATIONARY":
                for ego_pt in candidate.waypoints:
                    if ego_pt.x < agent.trajectories[0].waypoints[0].position.x:
                        lat_d = abs(ego_pt.y - agent.trajectories[0].waypoints[0].position.y)
                        long_d = agent.trajectories[0].waypoints[0].position.x - ego_pt.x
                        if lat_d < 1.1 and long_d < 7.0:
                            cost_collision += 40.0 * (1.0 - long_d / 7.0)

        cost_collision = min(100.0, cost_collision)
        cost_uncertainty = min(100.0, cost_uncertainty)

        # 5. Obstacle & Road Anomaly Clearance
        cost_clearance = 0.0
        for anom in perception.anomalies:
            for ego_pt in candidate.waypoints:
                adx = ego_pt.x - anom.position.x
                ady = ego_pt.y - anom.position.y
                adist = math.hypot(adx, ady)
                if adist < (anom.radius_m + 0.8):
                    cost_clearance += 25.0 * (1.0 - adist / (anom.radius_m + 0.8))

        if min_clearance < 3.5:
            cost_clearance += min(100.0, ((3.5 - min_clearance) / 3.5) ** 2 * 40.0)

        cost_clearance = min(100.0, cost_clearance)

        # Hard collision rejection (< 0.85m minimum clearance)
        if min_clearance < 0.85:
            breakdown_collision = {
                "collision_safety": 100.0,
                "obstacle_clearance": round(cost_clearance, 2),
                "road_traversability": round(cost_traversability, 2),
                "traversability": round(cost_traversability, 2),
                "progress": 0.0,
                "path_smoothness": 0.0,
                "vehicle_dynamics": round(candidate.max_curvature * 30.0, 2),
                "curvature": round(candidate.max_curvature * 30.0, 2),
                "comfort": 0.0,
                "uncertainty": round(cost_uncertainty, 2),
            }
            threat_desc = f"with {collision_threat_agent}" if collision_threat_agent else "in vehicle envelope"
            return TrajectoryCostScore(
                candidate_id=candidate.candidate_id,
                label=candidate.label,
                total_cost=99999.0,
                is_feasible=False,
                status_tag="COLLISION",
                cost_breakdown=breakdown_collision,
                min_clearance_m=round(min_clearance, 2),
                min_boundary_margin_m=round(min_boundary_margin, 2),
                explanation=f"Collision risk {threat_desc} (min clearance {min_clearance:.2f}m < 0.85m)"
            )

        # Unsafe clearance warning (< 1.20m minimum clearance)
        if min_clearance < 1.20:
            status = "UNSAFE_CLEARANCE"
            explanation_str = f"Unsafe clearance to {collision_threat_agent or 'obstacle'} ({min_clearance:.2f}m < 1.20m buffer)"
        else:
            status = "SAFE"
            explanation_str = f"Feasible trajectory with {min_clearance:.2f}m clearance & {min_boundary_margin:.2f}m ditch margin"

        # 6. Progress & Speed Efficiency
        end_speed = candidate.target_v
        speed_dev = abs(target_cruise_speed_mps - end_speed) / max(1.0, target_cruise_speed_mps)
        cost_progress = speed_dev * 40.0

        # Reward forward travel progress
        dx_progress = candidate.waypoints[-1].x - candidate.waypoints[0].x if len(candidate.waypoints) > 1 else 0.0
        cost_progress += max(0.0, (20.0 - dx_progress) * 1.5)
        cost_progress = min(100.0, cost_progress)

        # 7. Path Smoothness & Length Detour
        cost_smoothness = abs(candidate.path_length_m - dx_progress) * 8.0 + abs(candidate.target_d) * 4.0
        cost_smoothness = min(100.0, cost_smoothness)

        # 8. Vehicle Dynamics & Comfort
        cost_dynamics = (
            (candidate.max_curvature / self.max_curvature) * 35.0 +
            (candidate.max_lateral_accel / self.max_lateral_accel) * 35.0 +
            min(30.0, (candidate.total_jerk / max(1, len(candidate.waypoints))) * 5.0)
        )
        cost_dynamics = min(100.0, cost_dynamics)

        # Composite Weighted Score (0 to 100)
        total_score = (
            self.w_collision * cost_collision +
            self.w_clearance * cost_clearance +
            self.w_traversability * cost_traversability +
            self.w_progress * cost_progress +
            self.w_smoothness * cost_smoothness +
            self.w_dynamics * cost_dynamics +
            self.w_uncertainty * cost_uncertainty
        )

        breakdown = {
            "collision_safety": round(cost_collision, 2),
            "obstacle_clearance": round(cost_clearance, 2),
            "road_traversability": round(cost_traversability, 2),
            "traversability": round(cost_traversability, 2),
            "progress": round(cost_progress, 2),
            "path_smoothness": round(cost_smoothness, 2),
            "vehicle_dynamics": round(cost_dynamics, 2),
            "curvature": round(candidate.max_curvature * 10.0, 2),
            "comfort": round(cost_dynamics, 2),
            "uncertainty": round(cost_uncertainty, 2),
        }

        return TrajectoryCostScore(
            candidate_id=candidate.candidate_id,
            label=candidate.label,
            total_cost=round(total_score, 2),
            is_feasible=True,
            status_tag=status,
            cost_breakdown=breakdown,
            min_clearance_m=round(min_clearance, 2),
            min_boundary_margin_m=round(min_boundary_margin, 2),
            explanation=explanation_str
        )


MultiObjectiveCostEvaluator = TrajectoryCostEvaluator

