from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class IDDDetectionClass(str, Enum):
    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    AUTORICKSHAW = "autorickshaw"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    ANIMAL = "animal"

DETECTION_CLASS_TO_IDX = {cls_name.value: idx for idx, cls_name in enumerate(IDDDetectionClass)}
IDX_TO_DETECTION_CLASS = {idx: cls_name.value for idx, cls_name in enumerate(IDDDetectionClass)}

class IDDSegmentationClass(int, Enum):
    ROAD = 0             # Main paved carriageway
    DRIVABLE_AREA = 1    # Dirt shoulder, unpaved drivable verge
    ROADSIDE = 2         # Ditch, sidewalk, barricades, trees
    VEHICLES = 3         # Auto-rickshaw, car, truck, bus, bike
    PEDESTRIANS = 4      # Walking/crossing people
    OBSTACLES = 5        # Animals, boulders, potholes, pushcarts

class BoundingBox2D(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: IDDDetectionClass
    confidence: float = 1.0

class IDDImageSample(BaseModel):
    image_id: str
    file_path: str
    width: int = 640
    height: int = 360
    boxes: List[BoundingBox2D] = Field(default_factory=list)
    segmentation_mask_path: Optional[str] = None
    weather: str = "clear"
    road_type: str = "rural_village"
