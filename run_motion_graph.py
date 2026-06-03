"""Motion graph demos on the G1.

  task1: greedy command following -- walk the graph, at each decision point taking
         the edge whose motion best matches the commanded velocity.
  task2: in-betweening by beam search -- plan an edge sequence that arrives at a
         terminal state at a fixed time.

Usage: python run_motion_graph.py [task1|task2|both]
"""
import sys
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.commands import SpeedCommand
from motiongraph.motion_graph import MotionGraph

START = 2500   # a steady walking frame in walk1_subject1


def _cmd_marker(out, command):
    def fn(t):
        i = min(t, len(out) - 1)
        spd, hd = command.state(i * C.DT)
        tip = out[i, 0:2] + max(0.4, spd) * 0.5 * np.array([np.cos(hd), np.sin(hd)])
        return [([tip[0], tip[1], 1.2], 0.06, [1, 0.2, 0.2, 1])]
    return fn


def _gentle_schedule():
    d = np.deg2rad
    return SpeedCommand([
        (0.0, 1.2, d(0)),    # walk +x
        (3.0, 1.5, d(35)),   # gentle veer left
        (6.0, 1.8, d(35)),   # jog
        (9.0, 1.4, d(-15)),  # veer right
        (12.0, 1.1, d(-15)),
    ])


def task1():
    lib = load_library()
    g = MotionGraph(lib)
    cmd = _gentle_schedule()
    out = g.follow_command(cmd, seconds=15.0, start_frame=START)
    from motiongraph.render import render_qpos
    render_qpos(out, f"{C.OUT_DIR}/mg_task1_speed.mp4", markers_fn=_cmd_marker(out, cmd))


def task2():
    lib = load_library()
    g = MotionGraph(lib)
    cmd = SpeedCommand([(0.0, 1.3, np.deg2rad(15))])
    target_xy = np.array([4.0, 1.0])
    target_yaw = np.deg2rad(20)
    term_frame = 2600
    out = g.plan_to(cmd, seconds=8.0, start_frame=START,
                    target_xy=target_xy, target_yaw=target_yaw, term_frame=term_frame)

    def marker(t):
        return [([target_xy[0], target_xy[1], 0.9], 0.12, [0.2, 1, 0.2, 1])]
    from motiongraph.render import render_qpos
    render_qpos(out, f"{C.OUT_DIR}/mg_task2_inbetween.mp4", markers_fn=marker)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("task1", "both"):
        task1()
    if which in ("task2", "both"):
        task2()
