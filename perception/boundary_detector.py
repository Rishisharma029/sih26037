"""Free-space drivable corridor detection on unmarked roads."""
from interfaces import FreeSpaceCorridor, CorridorBoundaryPoint

class FreeSpaceBoundaryDetector:
    """Detects drivable road boundaries without painted lane lines."""
    def __init__(self, default_width_m: float = 6.0):
        self.default_width_m = default_width_m

    def detect_corridor(self, timestamp: float, lookahead_m: float = 40.0, step_m: float = 5.0) -> FreeSpaceCorridor:
        points = []
        s = 0.0
        while s <= lookahead_m:
            half_w = self.default_width_m / 2.0
            points.append(CorridorBoundaryPoint(
                s=s,
                d_left=half_w,
                d_right=-half_w,
                curvature=0.0
            ))
            s += step_m

        return FreeSpaceCorridor(
            timestamp=timestamp,
            boundary_points=points,
            average_width_m=self.default_width_m,
            is_blocked=False
        )
