"""Topological router without lane markers."""
from interfaces import Point3D, MissionGoal

class GlobalRouter:
    """Generates coarse sequence of navigation waypoints."""
    def route_to_goal(self, start_pos: Point3D, goal: MissionGoal) -> list[Point3D]:
        return [start_pos, goal.target_pose.position]
