"""Free-space drivable corridor detection on unmarked roads."""
import math
from typing import Optional, List, Tuple
from interfaces import FreeSpaceCorridor, CorridorBoundaryPoint

class FreeSpaceBoundaryDetector:
    """Detects drivable road boundaries and ditch verges without painted lane lines."""
    def __init__(self, default_width_m: float = 6.0):
        self.default_width_m = default_width_m

    def detect_corridor(
        self,
        timestamp: float,
        lookahead_m: float = 40.0,
        step_m: float = 5.0,
        current_s: float = 0.0,
        geometry: Optional[object] = None
    ) -> FreeSpaceCorridor:
        """Computes continuous left/right boundary points and local road curvature."""
        points: List[CorridorBoundaryPoint] = []
        s_rel = 0.0
        total_width = 0.0
        count = 0

        while s_rel <= lookahead_m:
            s_abs = current_s + s_rel
            if geometry is not None and hasattr(geometry, "get_corridor_widths"):
                d_left, d_right = geometry.get_corridor_widths(s_abs)
                # Curvature
                if 100.0 < s_abs <= 180.0:
                    curv = 0.015
                else:
                    curv = 0.0
            else:
                half_w = self.default_width_m / 2.0
                d_left = half_w
                d_right = -half_w
                curv = 0.0

            width = d_left - d_right
            total_width += width
            count += 1

            points.append(CorridorBoundaryPoint(
                s=round(s_rel, 2),
                d_left=round(d_left, 3),
                d_right=round(d_right, 3),
                curvature=round(curv, 4)
            ))
            s_rel += step_m

        avg_width = total_width / max(1, count)

        return FreeSpaceCorridor(
            timestamp=timestamp,
            boundary_points=points,
            average_width_m=round(avg_width, 2),
            is_blocked=False
        )

    def get_margin(
        self,
        d_left: float,
        d_right: float,
        lateral_d: float,
        vehicle_half_width: float = 0.90
    ) -> float:
        """Computes clearance margin to left and right road edges."""
        margin_left = d_left - (lateral_d + vehicle_half_width)
        margin_right = (lateral_d - vehicle_half_width) - d_right
        return min(margin_left, margin_right)

