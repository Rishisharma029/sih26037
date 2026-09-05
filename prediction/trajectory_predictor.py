"""Multi-modal CTRA trajectory predictor."""
import math
from interfaces import (
    PerceptionOutput, PredictionOutput, PredictedAgent,
    PredictedTrajectory, PredictedTrajectoryPoint, Point3D, Vector3D
)
from .intent_classifier import IntentClassifier

class TrajectoryPredictor:
    """Forecasts future actor waypoints over a defined time horizon."""
    def __init__(self, horizon_seconds: float = 3.0, dt: float = 0.5):
        self.horizon_seconds = horizon_seconds
        self.dt = dt
        self.intent_classifier = IntentClassifier()

    def predict(self, perception: PerceptionOutput) -> PredictionOutput:
        predicted_agents = []
        high_risk_ids = []

        for obs in perception.obstacles:
            intent = self.intent_classifier.classify_intent(obs)
            waypoints = []
            steps = int(self.horizon_seconds / self.dt)
            curr_x = obs.bbox.center.x
            curr_y = obs.bbox.center.y
            vx = obs.velocity.x
            vy = obs.velocity.y

            for step in range(1, steps + 1):
                t_future = perception.timestamp + step * self.dt
                fut_x = curr_x + vx * (step * self.dt)
                fut_y = curr_y + vy * (step * self.dt)
                waypoints.append(PredictedTrajectoryPoint(
                    timestamp=t_future,
                    position=Point3D(x=fut_x, y=fut_y, z=obs.bbox.center.z),
                    velocity=Vector3D(x=vx, y=vy, z=0.0),
                    yaw_rad=obs.bbox.yaw_rad
                ))

            is_risk = intent in ["CUTTING_IN", "ERRATIC_SWERVE"] or (obs.distance_m < 15.0 and not obs.is_static)
            if is_risk:
                high_risk_ids.append(obs.id)

            traj = PredictedTrajectory(probability=1.0, waypoints=waypoints)
            predicted_agents.append(PredictedAgent(
                id=obs.id,
                obstacle_class=obs.obstacle_class,
                primary_intent=intent,
                trajectories=[traj],
                is_high_risk=is_risk
            ))

        return PredictionOutput(
            timestamp=perception.timestamp,
            horizon_seconds=self.horizon_seconds,
            agents=predicted_agents,
            high_risk_agent_ids=high_risk_ids
        )
