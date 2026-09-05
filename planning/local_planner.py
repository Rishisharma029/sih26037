"""Adaptive Frenet-frame lattice trajectory planner for Indian traffic."""
import math
from typing import List, Optional, Tuple
from interfaces import (
    PlannedTrajectory, TrajectoryPoint, BehaviorMode,
    EgoVehicleState, PerceptionOutput, PredictionOutput
)
from .behavior_planner import BehaviorPlanner
from .frenet_lattice import FrenetLatticeGenerator, CandidateTrajectory
from .cost_evaluator import TrajectoryCostEvaluator, TrajectoryCostScore


class AdaptiveLatticePlanner:
    """Adaptive Multi-Candidate Frenet-Lattice Trajectory Planner.

    Generates a rich bundle of candidate trajectories across lateral offsets and speeds,
    evaluates them against safety, clearance, boundary traversability, comfort, progress,
    and uncertainty penalties, and selects the safest feasible trajectory.
    """

    def __init__(
        self,
        horizon_seconds: float = 3.0,
        dt: float = 0.2,
        w_safety: float = 250.0,
        w_clearance: float = 60.0,
        w_traversability: float = 300.0,
        w_progress: float = 25.0,
        w_path_length: float = 10.0,
        w_curvature: float = 40.0,
        w_comfort: float = 30.0,
        w_uncertainty: float = 80.0
    ):
        self.horizon_seconds = horizon_seconds
        self.dt = dt
        self.plan_counter = 0
        self.behavior_planner = BehaviorPlanner()
        self.lattice_generator = FrenetLatticeGenerator(dt=dt)
        self.cost_evaluator = TrajectoryCostEvaluator(
            w_safety=w_safety,
            w_clearance=w_clearance,
            w_traversability=w_traversability,
            w_progress=w_progress,
            w_path_length=w_path_length,
            w_curvature=w_curvature,
            w_comfort=w_comfort,
            w_uncertainty=w_uncertainty
        )
        self.last_candidate_scores: List[TrajectoryCostScore] = []

    def plan(
        self,
        ego_state: EgoVehicleState,
        perception: PerceptionOutput,
        prediction: PredictionOutput,
        target_cruise_speed_mps: float = 8.0
    ) -> PlannedTrajectory:
        """Generate candidate trajectory bundle, evaluate multi-objective costs,

        and select the globally optimal feasible path.
        """
        self.plan_counter += 1

        # 1. Sample candidate trajectory lattice
        candidates = self.lattice_generator.sample_candidates(
            ego_state=ego_state,
            target_cruise_speed_mps=target_cruise_speed_mps
        )

        # 2. Evaluate multi-objective cost for each candidate
        scored_candidates: List[Tuple[CandidateTrajectory, TrajectoryCostScore]] = []
        for cand in candidates:
            score = self.cost_evaluator.evaluate(
                candidate=cand,
                perception=perception,
                prediction=prediction,
                target_cruise_speed_mps=target_cruise_speed_mps
            )
            scored_candidates.append((cand, score))

        self.last_candidate_scores = [s for _, s in scored_candidates]

        # 3. Filter feasible candidates
        feasible_candidates = [
            (cand, score) for cand, score in scored_candidates if score.is_feasible
        ]

        # 4. Select optimal trajectory
        if feasible_candidates:
            # Sort by total cost ascending
            feasible_candidates.sort(key=lambda item: item[1].total_cost)
            best_cand, best_score = feasible_candidates[0]

            # Classify behavior mode from selected trajectory
            behavior = self._infer_behavior_mode(best_cand, target_cruise_speed_mps)

            return PlannedTrajectory(
                trajectory_id=f"traj_{self.plan_counter}_{best_cand.candidate_id}",
                timestamp=ego_state.timestamp,
                behavior_mode=behavior,
                waypoints=best_cand.waypoints,
                target_speed_mps=best_cand.target_v,
                total_cost=best_score.total_cost,
                is_feasible=True
            )
        else:
            # Fallback: All candidate paths are blocked -> execute safe controlled stop
            return self._generate_fallback_stop(ego_state)

    def _infer_behavior_mode(self, cand: CandidateTrajectory, cruise_speed: float) -> BehaviorMode:
        """Map selected candidate parameters to standard BehaviorMode."""
        if cand.target_v < 0.5:
            return BehaviorMode.EMERGENCY_STOP
        elif cand.target_d > 0.5:
            return BehaviorMode.NUDGE_LEFT
        elif cand.target_d < -0.5:
            return BehaviorMode.NUDGE_RIGHT
        elif cand.target_v < cruise_speed * 0.75:
            return BehaviorMode.FOLLOW
        return BehaviorMode.CRUISE

    def _generate_fallback_stop(self, ego_state: EgoVehicleState) -> PlannedTrajectory:
        """Generate safe emergency braking trajectory when corridor is completely blocked."""
        steps = max(5, int(self.horizon_seconds / self.dt))
        curr_x = ego_state.pose.position.x
        curr_y = ego_state.pose.position.y
        curr_v = ego_state.twist.speed_mps

        waypoints = []
        for i in range(1, steps + 1):
            t = ego_state.timestamp + i * self.dt
            # Linear deceleration to 0
            v_t = max(0.0, curr_v * (1.0 - (i / steps)))
            dist = (curr_v + v_t) * 0.5 * (i * self.dt)
            waypoints.append(TrajectoryPoint(
                timestamp=t,
                x=curr_x + dist,
                y=curr_y,
                yaw_rad=ego_state.pose.heading_rad,
                curvature=0.0,
                speed_mps=v_t,
                acceleration_mps2=-3.5,
                jerk_mps3=0.0
            ))

        return PlannedTrajectory(
            trajectory_id=f"traj_{self.plan_counter}_fallback_stop",
            timestamp=ego_state.timestamp,
            behavior_mode=BehaviorMode.SAFE_STOP,
            waypoints=waypoints,
            target_speed_mps=0.0,
            total_cost=999.0,
            is_feasible=True
        )
