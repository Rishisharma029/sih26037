"""Unified multi-modal motion prediction engine for SIH26037."""
import math
from typing import List, Optional
from interfaces import (
    PerceptionOutput, PredictionOutput, PredictedAgent,
    PredictedTrajectory, MotionIntent, ObstacleClass
)
from .intent_classifier import IntentClassifier
from .kinematic_predictor import KinematicPredictor
from .learned_predictor import LearnedPredictor


class TrajectoryPredictor:
    """Unified Motion Predictor for unstructured Indian driving environments.

    Forecasting multiple candidate trajectories (continuation, evasive swerve, cut-in)
    with probabilistic confidence, expanding spatial covariance uncertainty, and dynamic
    corridor invasion assessment against the autonomous ego vehicle.
    """

    def __init__(
        self,
        horizon_seconds: float = 3.0,
        dt: float = 0.5,
        mode: str = "kinematic"
    ):
        self.horizon_seconds = horizon_seconds
        self.dt = dt
        self.mode = mode
        self.intent_classifier = IntentClassifier()
        self.kinematic_predictor = KinematicPredictor(horizon_seconds=horizon_seconds, dt=dt)
        self.learned_predictor = LearnedPredictor(horizon_seconds=horizon_seconds, dt=dt)

    def predict(self, perception: PerceptionOutput, ego_speed: float = 6.0) -> PredictionOutput:
        """Forecast future trajectories for all tracked obstacles in the perception frame."""
        predicted_agents: List[PredictedAgent] = []
        high_risk_ids: List[str] = []

        corridor_half_w = max(1.8, perception.drivable_corridor.average_width_m * 0.5)

        for obs in perception.obstacles:
            # 1. Infer dynamic intention
            intent = self.intent_classifier.classify_intent(obs)

            # 2. Generate multi-modal trajectory branches
            if self.mode == "learned":
                trajectories = self.learned_predictor.predict_modes(
                    obs, intent, perception.timestamp, ego_speed=ego_speed
                )
            elif self.mode == "ensemble":
                kin_trajs = self.kinematic_predictor.predict_modes(
                    obs, intent, perception.timestamp, ego_speed=ego_speed
                )
                lrn_trajs = self.learned_predictor.predict_modes(
                    obs, intent, perception.timestamp, ego_speed=ego_speed
                )
                trajectories = []
                for k in range(min(len(kin_trajs), len(lrn_trajs))):
                    avg_p = round(0.5 * (kin_trajs[k].probability + lrn_trajs[k].probability), 3)
                    avg_risk = max(kin_trajs[k].collision_risk, lrn_trajs[k].collision_risk)
                    trajectories.append(PredictedTrajectory(
                        probability=avg_p,
                        mode_name=kin_trajs[k].mode_name,
                        collision_risk=avg_risk,
                        waypoints=kin_trajs[k].waypoints
                    ))
            else: # "kinematic" (default baseline)
                trajectories = self.kinematic_predictor.predict_modes(
                    obs, intent, perception.timestamp, ego_speed=ego_speed
                )

            # 3. Assess Corridor Invasion Probability & Time-To-Conflict
            corridor_invasion_prob = 0.0
            earliest_conflict_time: Optional[float] = None

            for traj in trajectories:
                branch_invades = False
                for pt in traj.waypoints:
                    # Check if waypoint falls inside the drivable forward corridor
                    if 0.0 <= pt.position.x <= 45.0 and abs(pt.position.y) <= (corridor_half_w + 0.3):
                        branch_invades = True
                        t_delta = pt.timestamp - perception.timestamp
                        if earliest_conflict_time is None or t_delta < earliest_conflict_time:
                            earliest_conflict_time = t_delta

                if branch_invades:
                    corridor_invasion_prob += traj.probability

            corridor_invasion_prob = round(min(1.0, corridor_invasion_prob), 2)
            time_to_conflict_s = round(earliest_conflict_time, 2) if earliest_conflict_time is not None else None

            # 4. Generate Explainable Reasoning Summary
            cls_name = obs.obstacle_class.value.lower().replace("_", " ")
            inv_pct = int(round(corridor_invasion_prob * 100))

            if corridor_invasion_prob >= 0.15:
                if time_to_conflict_s is not None:
                    explanation = (
                        f"There is a {inv_pct}% probability this {cls_name} ({obs.id}) "
                        f"will cut into my corridor in {time_to_conflict_s:.1f}s."
                    )
                else:
                    explanation = (
                        f"There is a {inv_pct}% probability this {cls_name} ({obs.id}) "
                        f"will cut into my corridor."
                    )
            elif obs.is_static:
                explanation = f"Stationary {cls_name} ({obs.id}) resting at roadside."
            else:
                explanation = (
                    f"{cls_name.capitalize()} ({obs.id}) predicted to maintain path "
                    f"clear of active corridor (cut-in risk < 15%)."
                )

            # 5. Determine if actor poses high risk to ego path
            max_risk = max((t.collision_risk for t in trajectories), default=0.0)
            is_risk = (
                max_risk >= 0.40 or
                corridor_invasion_prob >= 0.35 or
                intent in [MotionIntent.CUTTING_IN, MotionIntent.ERRATIC_SWERVE, MotionIntent.CROSSING_PATH] or
                (obs.distance_m < 10.0 and not obs.is_static)
            )

            if is_risk:
                high_risk_ids.append(obs.id)

            predicted_agents.append(PredictedAgent(
                id=obs.id,
                obstacle_class=obs.obstacle_class,
                primary_intent=intent,
                trajectories=trajectories,
                is_high_risk=is_risk,
                corridor_invasion_prob=corridor_invasion_prob,
                time_to_conflict_s=time_to_conflict_s,
                explanation=explanation
            ))

        return PredictionOutput(
            timestamp=perception.timestamp,
            horizon_seconds=self.horizon_seconds,
            agents=predicted_agents,
            high_risk_agent_ids=high_risk_ids
        )
