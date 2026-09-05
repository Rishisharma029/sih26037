"""Unified multi-modal motion prediction engine for SIH26037."""
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
    collision risk against the autonomous ego vehicle.
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

        for obs in perception.obstacles:
            # 1. Infer dynamic intention
            intent = self.intent_classifier.classify_intent(obs)

            # 2. Generate multi-modal trajectory branches
            if self.mode == "learned":
                trajectories = self.learned_predictor.predict_modes(
                    obs, intent, perception.timestamp, ego_speed=ego_speed
                )
            elif self.mode == "ensemble":
                # Average probabilities from kinematic and learned
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

            # 3. Determine if actor poses high risk to ego path
            max_risk = max((t.collision_risk for t in trajectories), default=0.0)
            is_risk = (
                max_risk >= 0.40 or
                intent in [MotionIntent.CUTTING_IN, MotionIntent.ERRATIC_SWERVE, MotionIntent.CROSSING_PATH] or
                (obs.distance_m < 12.0 and not obs.is_static)
            )

            if is_risk:
                high_risk_ids.append(obs.id)

            predicted_agents.append(PredictedAgent(
                id=obs.id,
                obstacle_class=obs.obstacle_class,
                primary_intent=intent,
                trajectories=trajectories,
                is_high_risk=is_risk
            ))

        return PredictionOutput(
            timestamp=perception.timestamp,
            horizon_seconds=self.horizon_seconds,
            agents=predicted_agents,
            high_risk_agent_ids=high_risk_ids
        )
