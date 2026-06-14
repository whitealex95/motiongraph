"""Motion matching demo on the G1 (GenoView controller).

The MotionMatcher is now the reactive GenoView (Holden "Simple Motion Matching") controller
(see motion_matching.py). This demo drives it with a time-varying speed/heading command and
renders the result.

In-betweening (arrive at a terminal state at a fixed time) is a *planned* task and the
reactive GenoView MM does not do it -- use `run_motion_graph.py task2` (the motion graph's
A* planner) for that.

Usage: python run_motion_matching.py
"""
import sys
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.commands import demo_speed_schedule
from motiongraph.motion_matching import MotionMatcher
from motiongraph.cleanup import cleanup
from motiongraph.render import trace_labels, render_qpos

START = 1500   # a steady forward walk frame (walk1_subject5)


def _cmd_marker(out, command):
    def fn(t):                       # t is the frame index (see render.py)
        i = min(t, len(out) - 1)
        spd, hd = command.state(i * C.DT)
        tip = out[i, 0:2] + max(0.4, spd) * 0.5 * np.array([np.cos(hd), np.sin(hd)])
        return [([tip[0], tip[1], 1.2], 0.06, [1, 0.2, 0.2, 1])]   # red: command dir
    return fn


def gen_demo(mm=None, clean=True):
    """Speed-command-driven locomotion (walk <-> run) via the GenoView controller."""
    mm = mm or MotionMatcher(load_library(C.LOCO_MIRROR_LIB_PATH))
    cmd = demo_speed_schedule()
    out, tframe = mm.generate(cmd, seconds=15.0, start_frame=START, return_trace=True)
    trace = trace_labels(tframe, mm.lib)
    if clean:
        out = cleanup(out)                                # de-jitter root + foot-lock
    return out, _cmd_marker(out, cmd), trace


if __name__ == "__main__":
    out, mk, tr = gen_demo()
    render_qpos(out, f"{C.OUT_DIR}/mm_demo_speed.mp4", markers_fn=mk, trace=tr)
