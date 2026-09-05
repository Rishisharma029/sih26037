"""Intent classification for non-lane-respecting actors."""
from interfaces import TrackedObstacle, MotionIntent

class IntentClassifier:
    """Infers intentions like cut-in, swerve, or crossing from trajectory cues."""
    def classify_intent(self, obstacle: TrackedObstacle) -> MotionIntent:
        if obstacle.is_static or abs(obstacle.velocity.x) < 0.1 and abs(obstacle.velocity.y) < 0.1:
            return MotionIntent.STATIONARY

        # Check for aggressive lateral motion (cutting in)
        if abs(obstacle.velocity.y) > 0.8:
            return MotionIntent.CUTTING_IN
        if abs(obstacle.velocity.x) > 0.1 and abs(obstacle.velocity.y) > 0.3:
            return MotionIntent.ERRATIC_SWERVE
        return MotionIntent.CRUISING
