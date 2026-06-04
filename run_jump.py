"""Jump-skill demos on the G1 (motion graph over a walk + walk->jump->walk library).

  task1: walk forward at 1 m/s, then JUMP on a trigger (action goal, not a target).
  task2: walk forward at 1 m/s and land the jump apex at a FIXED x location -- the
         trigger time is chosen by a search over the graph (motion-graph optimization).

Usage: python run_jump.py [task1|task2|both]
"""
import sys
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.commands import SpeedCommand
from motiongraph.motion_graph import MotionGraph
from motiongraph.cleanup import cleanup

START = 1500
SPEED = 1.0


def _airborne(out):
    return out[:, 2] > 0.90                      # pelvis high -> mid-jump


def _jump_marker(out):
    air = _airborne(out)
    def fn(t):
        i = min(t, len(out) - 1)
        if air[i]:
            return [([out[i, 0], out[i, 1], out[i, 2] + 0.4], 0.08, [1.0, 0.55, 0.0, 1])]
        return []
    return fn


def gen_task1(g=None, clean=True):
    g = g or MotionGraph(load_library(C.JUMP_LIB_PATH))
    cmd = SpeedCommand([(0.0, SPEED, 0.0)])
    out = g.follow_with_jump(cmd, seconds=8.0, start_frame=START, jump_at=3.5)
    if clean:
        out = cleanup(out)
    return out, _jump_marker(out)


def _apex_x(out):
    return float(out[out[:, 2].argmax(), 0])


def gen_task2(g=None, clean=True, target_x=4.0, seconds=9.0):
    """Search the jump trigger time so the apex lands at target_x (graph optimization)."""
    g = g or MotionGraph(load_library(C.JUMP_LIB_PATH))
    cmd = SpeedCommand([(0.0, SPEED, 0.0)])
    best, best_err, best_t = None, 1e9, None
    for jump_at in np.linspace(1.5, seconds - 3.0, 25):     # candidate trigger times
        o = g.follow_with_jump(cmd, seconds, start_frame=START, jump_at=float(jump_at))
        err = abs(_apex_x(o) - target_x)
        if err < best_err:
            best, best_err, best_t = o, err, float(jump_at)
    print(f"  task2: chose jump_at={best_t:.2f}s -> apex_x={_apex_x(best):.2f} (target {target_x}, err {best_err:.2f})")
    out = cleanup(best) if clean else best
    apex_y = float(out[out[:, 2].argmax(), 1])           # lateral position of the jump

    def marker(t):
        m = [([target_x, apex_y, 0.05], 0.14, [0.2, 1, 0.2, 1])]         # green: target x line
        air = _airborne(out)
        i = min(t, len(out) - 1)
        if air[i]:
            m.append(([out[i, 0], out[i, 1], out[i, 2] + 0.4], 0.08, [1.0, 0.55, 0.0, 1]))
        return m
    return out, marker


def _render(out, markers_fn, name):
    from motiongraph.render import render_qpos
    render_qpos(out, f"{C.OUT_DIR}/{name}.mp4", markers_fn=markers_fn)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    g = MotionGraph(load_library(C.JUMP_LIB_PATH))
    if which in ("task1", "both"):
        _render(*gen_task1(g), "jump_task1_oncommand")
    if which in ("task2", "both"):
        _render(*gen_task2(g), "jump_task2_fixedloc")
