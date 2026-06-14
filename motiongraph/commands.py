"""Speed commands for the GenoView matcher.

A command is a schedule of (start_time, speed[m/s], heading[rad]); `state(t)` returns the
(speed, heading) active at time t. The matcher feeds that to its trajectory springs each
frame (motion_matching.step), so a time-varying command steers the gait (slow=walk,
fast=run) and direction.
"""
import numpy as np


class SpeedCommand:
    def __init__(self, schedule):
        # schedule: list of (t_start_sec, speed, heading_rad), sorted by t_start.
        self.schedule = sorted(schedule, key=lambda x: x[0])

    def state(self, t):
        """(speed, heading) active at time t seconds."""
        s = self.schedule[0][1:]
        for t0, spd, hd in self.schedule:
            if t >= t0:
                s = (spd, hd)
        return s


def demo_speed_schedule():
    """Stand -> walk forward -> turn left -> turn right -> slow (walking speeds only)."""
    d = np.deg2rad
    return SpeedCommand([
        (0.0, 0.0, d(0)),     # settle
        (1.0, 1.0, d(0)),     # walk +x
        (4.0, 1.1, d(90)),    # turn to +y
        (7.5, 1.2, d(10)),    # turn back toward +x
        (11.0, 1.0, d(-40)),  # veer right
        (13.5, 0.7, d(-40)),  # slow
    ])
