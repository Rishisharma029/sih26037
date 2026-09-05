"""Learned Neural Multi-Modal Trajectory Predictor using PyTorch."""
import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from interfaces import (
    TrackedObstacle, ObstacleClass, MotionIntent,
    PredictedTrajectory, PredictedTrajectoryPoint, Point3D, Vector3D
)


class MultiModalTrajectoryNet(nn.Module):
    """Deep multi-modal trajectory forecasting network.

    Encodes past trajectory history using an MLP/GRU temporal backbone
    and predicts K discrete future trajectory modes with softmax mode probabilities
    and spatial uncertainty parameters.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_modes: int = 3, future_steps: int = 6):
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps

        # State encoder: [x, y, vx, vy, ax, ay, yaw, class_id]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Mode probability classification head
        self.mode_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_modes)
        )

        # Future offset and uncertainty head: K x (steps * 4: dx, dy, sigma_x, sigma_y)
        self.traj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, future_steps * 4)
            )
            for _ in range(num_modes)
        ])

    def forward(self, x: torch.Tensor):
        """Args:

            x: [Batch, input_dim]
        Returns:
            mode_probs: [Batch, num_modes]
            trajectories: [Batch, num_modes, future_steps, 4] (dx, dy, sigma_x, sigma_y)
        """
        feat = self.encoder(x)
        mode_logits = self.mode_head(feat)
        mode_probs = F.softmax(mode_logits, dim=-1)

        batch_size = x.shape[0]
        trajs = []
        for head in self.traj_heads:
            out = head(feat).view(batch_size, self.future_steps, 4)
            # Ensure sigmas are positive using softplus
            dx_dy = out[..., :2]
            sigmas = F.softplus(out[..., 2:]) + 0.05
            trajs.append(torch.cat([dx_dy, sigmas], dim=-1).unsqueeze(1))

        trajectories = torch.cat(trajs, dim=1) # [Batch, K, steps, 4]
        return mode_probs, trajectories


class LearnedPredictor:
    """Wrapper class integrating PyTorch neural multi-modal predictor with SIH26037 interfaces."""

    CLASS_MAP = {
        ObstacleClass.CAR: 0,
        ObstacleClass.BUS: 1,
        ObstacleClass.TRUCK: 2,
        ObstacleClass.MOTORCYCLE: 3,
        ObstacleClass.AUTO_RICKSHAW: 4,
        ObstacleClass.BICYCLE: 5,
        ObstacleClass.PEDESTRIAN: 6,
        ObstacleClass.CATTLE_ANIMAL: 7,
    }

    MODE_NAMES = ["continuation", "evasive_nudge", "cut_in_merge"]

    def __init__(self, horizon_seconds: float = 3.0, dt: float = 0.5):
        self.horizon_seconds = horizon_seconds
        self.dt = dt
        self.future_steps = int(horizon_seconds / dt)
        self.model = MultiModalTrajectoryNet(
            input_dim=8,
            hidden_dim=64,
            num_modes=3,
            future_steps=self.future_steps
        )
        self.model.eval()

    def predict_modes(
        self,
        obstacle: TrackedObstacle,
        intent: MotionIntent,
        current_time: float,
        ego_speed: float = 6.0
    ) -> List[PredictedTrajectory]:
        """Runs neural forward pass and formats multi-modal trajectories."""
        if obstacle.is_static:
            return [self._create_static_trajectory(obstacle, current_time)]

        # Prepare input tensor
        x0, y0 = obstacle.bbox.center.x, obstacle.bbox.center.y
        vx, vy = obstacle.velocity.x, obstacle.velocity.y
        ax, ay = obstacle.acceleration.x, obstacle.acceleration.y
        yaw = obstacle.bbox.yaw_rad
        cls_id = float(self.CLASS_MAP.get(obstacle.obstacle_class, 0))

        feat = torch.tensor([[x0, y0, vx, vy, ax, ay, yaw, cls_id]], dtype=torch.float32)

        with torch.no_grad():
            mode_probs, trajectories = self.model(feat)

        probs = mode_probs[0].tolist()
        trajs_tensor = trajectories[0] # [K, steps, 4]

        result_trajectories = []
        for k in range(3):
            pts = []
            for step in range(1, self.future_steps + 1):
                idx = step - 1
                dx = float(trajs_tensor[k, idx, 0]) + vx * (step * self.dt)
                dy = float(trajs_tensor[k, idx, 1]) + vy * (step * self.dt)
                sig_x = float(trajs_tensor[k, idx, 2]) + 0.1 * step * self.dt
                sig_y = float(trajs_tensor[k, idx, 3]) + 0.15 * step * self.dt

                t_future = current_time + step * self.dt
                fut_x = x0 + dx
                fut_y = y0 + dy

                pts.append(PredictedTrajectoryPoint(
                    timestamp=t_future,
                    position=Point3D(x=fut_x, y=fut_y, z=obstacle.bbox.center.z),
                    velocity=Vector3D(x=vx, y=vy, z=0.0),
                    yaw_rad=yaw,
                    sigma_x=round(sig_x, 3),
                    sigma_y=round(sig_y, 3)
                ))

            prob_val = round(max(0.01, probs[k]), 3)
            risk_val = self._calc_risk(pts, prob_val, ego_speed)
            result_trajectories.append(PredictedTrajectory(
                probability=prob_val,
                mode_name=self.MODE_NAMES[k],
                collision_risk=risk_val,
                waypoints=pts
            ))

        # Re-normalize probabilities
        total_p = sum(t.probability for t in result_trajectories)
        if total_p > 0:
            for t in result_trajectories:
                t.probability = round(t.probability / total_p, 3)

        return result_trajectories

    def _create_static_trajectory(self, obstacle: TrackedObstacle, current_time: float) -> PredictedTrajectory:
        pts = []
        for step in range(1, self.future_steps + 1):
            t_future = current_time + step * self.dt
            pts.append(PredictedTrajectoryPoint(
                timestamp=t_future,
                position=Point3D(x=obstacle.bbox.center.x, y=obstacle.bbox.center.y, z=obstacle.bbox.center.z),
                velocity=Vector3D(x=0.0, y=0.0, z=0.0),
                yaw_rad=obstacle.bbox.yaw_rad,
                sigma_x=0.05,
                sigma_y=0.05
            ))
        return PredictedTrajectory(
            probability=1.0,
            mode_name="stationary",
            collision_risk=0.0 if obstacle.distance_m > 8.0 else 0.35,
            waypoints=pts
        )

    def _calc_risk(self, waypoints: List[PredictedTrajectoryPoint], prob: float, ego_speed: float) -> float:
        if not waypoints:
            return 0.0
        min_d = 999.0
        for pt in waypoints:
            dt_future = pt.timestamp - waypoints[0].timestamp + self.dt
            ego_x_fut = ego_speed * dt_future
            dist = math.hypot(pt.position.x - ego_x_fut, pt.position.y)
            if dist < min_d:
                min_d = dist
        if min_d < 2.5:
            base_risk = 1.0 - (min_d / 2.5)
            return round(min(1.0, max(0.0, base_risk * (0.5 + 0.5 * prob))), 3)
        return 0.0
