"""Speed commands -> desired future trajectory used to query the motion database.

A command is a schedule of (start_time, speed[m/s], heading[rad]). The predicted
trajectory slews the heading toward the target at a fixed turn rate and integrates
at the commanded speed, then is expressed in the character's local frame to match
the trajectory feature layout in features.py.
"""
import numpy as np
from . import config as C
from .features import _local

MAX_H = max(C.TRAJ_HORIZONS)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class SpeedCommand:
    def __init__(self, schedule, turn_rate=2.5):
        # schedule: list of (t_start_sec, speed, heading_rad), sorted by t_start.
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.turn_rate = turn_rate

    def state(self, t):
        """(speed, heading) active at time t seconds."""
        s = self.schedule[0][1:]
        for t0, spd, hd in self.schedule:
            if t >= t0:
                s = (spd, hd)
        return s

    def desired_velocity(self, t):
        spd, hd = self.state(t)
        return spd * np.array([np.cos(hd), np.sin(hd)])

    def trajectory(self, world_xy, world_yaw, t):
        """Predicted future path -> trajectory feature block (4*len(horizons),)."""
        spd, target = self.state(t)
        pos, head = np.asarray(world_xy, float).copy(), world_yaw
        traj = {0: (pos.copy(), head)}
        for f in range(1, MAX_H + 1):
            head = head + np.clip(_wrap(target - head), -self.turn_rate * C.DT, self.turn_rate * C.DT)
            pos = pos + spd * C.DT * np.array([np.cos(head), np.sin(head)])
            traj[f] = (pos.copy(), head)
        block = []
        for h in C.TRAJ_HORIZONS:
            p, hd = traj[h]
            block += list(_local(p - world_xy, world_yaw))
            block += list(_local(np.array([np.cos(hd), np.sin(hd)]), world_yaw))
        return np.array(block, np.float32)


def demo_speed_schedule():
    """Stand -> walk forward -> turn left -> speed up (run) -> turn right -> slow."""
    d = np.deg2rad
    return SpeedCommand([
        (0.0, 0.0, d(0)),     # settle
        (1.0, 1.1, d(0)),     # walk +x
        (4.0, 1.2, d(90)),    # turn to +y
        (7.0, 2.6, d(90)),    # run
        (10.0, 2.6, d(0)),    # turn back to +x
        (13.0, 1.0, d(-45)),  # slow, veer
    ])
