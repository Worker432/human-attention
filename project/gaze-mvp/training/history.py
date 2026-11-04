"""Bounded, chronological fixation context, independent from output journals."""
from collections import deque
import math


class FixationHistory:
    def __init__(self, capacity=6):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ValueError('capacity must be a non-negative integer')
        self.capacity = capacity
        self._points = deque(maxlen=capacity)

    def append(self, x, y):
        x, y = float(x), float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError('fixations must be finite')
        self._points.append((x, y))

    def reset(self):
        self._points.clear()

    def window(self):
        """Oldest to newest; padding has mask True, never copied fixations."""
        points = list(self._points)
        missing = self.capacity - len(points)
        return points + [(0.0, 0.0)] * missing, [False] * len(points) + [True] * missing
