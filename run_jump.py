"""Jump-skill demos on the G1 (motion graph over a walk + walk->jump->walk library).

  task1: walk forward at 1 m/s, then JUMP on a trigger (action goal, not a target).
  task2: walk forward at 1 m/s and land the jump apex at a FIXED x location -- the
         trigger time is chosen by a search over the graph (motion-graph optimization).

Videos show a HUD (current clip / frame index) and flash on graph transitions.

Usage: python run_jump.py [task1|task2|both]
"""
import sys
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.commands import SpeedCommand
from motiongraph.motion_graph import MotionGraph
from motiongraph.cleanup import cleanup
from motiongraph.render import trace_labels

START = 1500
SPEED = 1.0
SECONDS = 13.0           # longer clips: more walking before and after the jump


def _jump_marker(out):
    air = out[:, 2] > 0.90
    def fn(t):
        i = min(t, len(out) - 1)
        if air[i]:
            return [([out[i, 0], out[i, 1], out[i, 2] + 0.4], 0.08, [1.0, 0.55, 0.0, 1])]
        return []
    return fn


def gen_task1(g=None, clean=True):
    g = g or MotionGraph(load_library(C.JUMP_LIB_PATH))
    cmd = SpeedCommand([(0.0, SPEED, 0.0)])
    out, tframe, _ = g.follow_with_jump(cmd, SECONDS, start_frame=START, jump_at=5.5, return_trace=True)
    trace = trace_labels(tframe, g.lib)
    if clean:
        out = cleanup(out)
    return out, _jump_marker(out), trace


def _apex_x(out):
    return float(out[out[:, 2].argmax(), 0])


def gen_task2(g=None, clean=True, target_x=5.0):
    """Search the jump trigger time so the apex lands at target_x (graph optimization)."""
    g = g or MotionGraph(load_library(C.JUMP_LIB_PATH))
    cmd = SpeedCommand([(0.0, SPEED, 0.0)])
    best, best_err, best_t = None, 1e9, None
    for jump_at in np.linspace(2.0, SECONDS - 4.0, 28):     # candidate trigger times
        o = g.follow_with_jump(cmd, SECONDS, start_frame=START, jump_at=float(jump_at))
        err = abs(_apex_x(o) - target_x)
        if err < best_err:
            best, best_err, best_t = o, err, float(jump_at)
    print(f"  task2: chose jump_at={best_t:.2f}s -> apex_x={_apex_x(best):.2f} (target {target_x}, err {best_err:.2f})")
    out, tframe, _ = g.follow_with_jump(cmd, SECONDS, start_frame=START, jump_at=best_t, return_trace=True)
    trace = trace_labels(tframe, g.lib)
    if clean:
        out = cleanup(out)
    apex_y = float(out[out[:, 2].argmax(), 1])

    def marker(t):
        m = [([target_x, apex_y, 0.05], 0.14, [0.2, 1, 0.2, 1])]         # green: target x line
        i = min(t, len(out) - 1)
        if out[i, 2] > 0.90:
            m.append(([out[i, 0], out[i, 1], out[i, 2] + 0.4], 0.08, [1.0, 0.55, 0.0, 1]))
        return m
    return out, marker, trace


def _render(out, markers_fn, trace, name):
    from motiongraph.render import render_qpos
    render_qpos(out, f"{C.OUT_DIR}/{name}.mp4", markers_fn=markers_fn, trace=trace)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    g = MotionGraph(load_library(C.JUMP_LIB_PATH))
    if which in ("task1", "both"):
        _render(*gen_task1(g), "jump_task1_oncommand")
    if which in ("task2", "both"):
        _render(*gen_task2(g), "jump_task2_fixedloc")
