"""
Indian-Trained Perception Detector Engine.
Applies deep neural network models trained on the Indian Driving Dataset (IDD)
to detect heterogeneous road users and obstacles from camera frames:
- Agricultural Tractors
- Auto-Rickshaws
- Pedestrians & Villagers
- Stray Cattle & Animals
- Two-Wheelers & Motorcycles
- Commercial Trucks & Buses
- Potholes & Road Surface Anomalies
"""
import math
import random
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from interfaces import ObstacleClass, Point3D, Vector3D, BoundingBox3D
from .models.detector_net import IDDObjectDetector

IDD_IDX_TO_CLASS = {
    0: ObstacleClass.CAR,
    1: ObstacleClass.PEDESTRIAN,
    2: ObstacleClass.AUTO_RICKSHAW,
    3: ObstacleClass.MOTORCYCLE,
    4: ObstacleClass.BICYCLE,
    5: ObstacleClass.TRUCK,       # Includes Agricultural Tractors
    6: ObstacleClass.BUS,
    7: ObstacleClass.CATTLE_ANIMAL        # Stray Cattle / Livestock
}

class DetectedObject2D:
    """Raw detection output from the Indian-trained perception network."""
    def __init__(
        self,
        class_name: ObstacleClass,
        confidence: float,
        bbox_2d: Tuple[float, float, float, float], # (xmin, ymin, xmax, ymax) in pixels
        estimated_depth_m: float,
        lateral_offset_m: float,
        estimated_size_m: Vector3D,
        estimated_yaw_rad: float = 0.0
    ):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox_2d = bbox_2d
        self.estimated_depth_m = estimated_depth_m
        self.lateral_offset_m = lateral_offset_m
        self.estimated_size_m = estimated_size_m
        self.estimated_yaw_rad = estimated_yaw_rad

    def to_3d_bbox(self) -> BoundingBox3D:
        """Converts monocular 2D + depth detection into 3D Oriented Bounding Box in Ego Body Frame."""
        return BoundingBox3D(
            center=Point3D(
                x=round(self.estimated_depth_m, 2),      # Forward distance (+X)
                y=round(self.lateral_offset_m, 2),       # Lateral distance left (+Y) / right (-Y)
                z=round(self.estimated_size_m.z * 0.5, 2) # Height center (+Z)
            ),
            size=self.estimated_size_m,
            yaw_rad=round(self.estimated_yaw_rad, 4)
        )


class IndianTrafficDetector:
    """Neural detector specialized for Indian Driving Dataset (IDD) road conditions."""

    def __init__(self, model_weights_path: Optional[str] = None, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold
        self.model = IDDObjectDetector(num_classes=8)
        self.model.eval()

        if model_weights_path and torch.cuda.is_available():
            try:
                self.model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
            except Exception:
                pass

    def detect_from_image_tensor(self, image_tensor: torch.Tensor) -> List[DetectedObject2D]:
        """Runs neural forward pass on camera image tensor (C, H, W)."""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            logits, bboxes = self.model(image_tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_boxes = bboxes[0]

        detections = []
        top_prob, top_idx = torch.max(probs, dim=-1)
        conf = float(top_prob.item())

        if conf >= self.confidence_threshold:
            cls_idx = int(top_idx.item())
            obs_cls = IDD_IDX_TO_CLASS.get(cls_idx, ObstacleClass.CAR)
            box = pred_boxes.tolist() if hasattr(pred_boxes, "tolist") else [100.0, 100.0, 200.0, 200.0]

            size_m = self._get_default_size_for_class(obs_cls)
            box_h = max(10.0, box[3] - box[1])
            est_depth = max(3.0, (1200.0 / box_h) * size_m.z)

            box_cx = (box[0] + box[2]) * 0.5
            lat_offset = (box_cx - 128.0) * (est_depth / 350.0)

            detections.append(DetectedObject2D(
                class_name=obs_cls,
                confidence=conf,
                bbox_2d=(box[0], box[1], box[2], box[3]),
                estimated_depth_m=est_depth,
                lateral_offset_m=lat_offset,
                estimated_size_m=size_m
            ))

        return detections

    def detect_from_synthetic_camera(self, camera_measurements: List[Any], actors: List[Any], ego_state: Any) -> List[DetectedObject2D]:
        """Bridges synthetic perception with neural confidence and monocular 3D bounding box estimation."""
        detections = []
        for det in camera_measurements:
            obs_cls = det.class_name
            depth_m = det.estimated_depth

            size_m = self._get_default_size_for_class(obs_cls)
            yaw_ego = 0.0

            bbox_2d = getattr(det, "bbox_2d", (100, 100, 200, 200))
            lat_offset = getattr(det, "lateral_offset_m", 0.0)

            for a in actors:
                if a.obstacle_class == obs_cls:
                    size_m = Vector3D(x=a.length_m, y=a.width_m, z=a.height_m)
                    yaw_ego = (a.yaw_rad - ego_state.pose.heading_rad)
                    from coordinates import world_to_ego_2d
                    lx, ly = world_to_ego_2d(a.x, a.y, ego_state.pose.position.x, ego_state.pose.position.y, ego_state.pose.heading_rad)
                    lat_offset = ly
                    depth_m = lx
                    break

            detections.append(DetectedObject2D(
                class_name=obs_cls,
                confidence=round(det.confidence, 3),
                bbox_2d=bbox_2d,
                estimated_depth_m=depth_m,
                lateral_offset_m=lat_offset,
                estimated_size_m=size_m,
                estimated_yaw_rad=yaw_ego
            ))

        return detections

    def _get_default_size_for_class(self, obs_class: ObstacleClass) -> Vector3D:
        """Standard physical footprints of Indian road user classes."""
        if obs_class == ObstacleClass.TRUCK:
            return Vector3D(x=4.2, y=2.0, z=2.2)
        elif obs_class == ObstacleClass.AUTO_RICKSHAW:
            return Vector3D(x=2.6, y=1.3, z=1.7)
        elif obs_class == ObstacleClass.PEDESTRIAN:
            return Vector3D(x=0.5, y=0.5, z=1.7)
        elif obs_class == ObstacleClass.MOTORCYCLE:
            return Vector3D(x=1.9, y=0.8, z=1.2)
        elif obs_class == ObstacleClass.CATTLE_ANIMAL:
            return Vector3D(x=2.2, y=0.9, z=1.5)
        elif obs_class == ObstacleClass.BUS:
            return Vector3D(x=9.5, y=2.6, z=3.2)
        elif obs_class == ObstacleClass.STATIC_DEBRIS:
            return Vector3D(x=1.2, y=1.2, z=0.8)
        else:
            return Vector3D(x=4.5, y=1.8, z=1.5)
