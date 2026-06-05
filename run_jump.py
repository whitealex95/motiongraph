"""Jump-skill demos on the G1, with BOTH motion matching and motion graph.

  task1: walk forward at 1 m/s, then JUMP on a trigger (action goal, not a target).
  task2: walk forward at 1 m/s and land the jump apex at a FIXED x location -- the
         trigger time is chosen by a search over candidates (graph/feature optimization).

A jump is only ever *entered* from its pre-take-off run-up (never mid-air). Videos show
a HUD (clip / frame index) and flash on transitions.

Usage: python run_jump.py [mm|mg|both]   (default: both algorithms, both tasks)
"""
import sys
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.commands import SpeedCommand
from motiongraph.motion_graph import MotionGraph
from motiongraph.motion_matching import MotionMatcher
from motiongraph.cleanup import cleanup
from motiongraph.render import trace_labels, render_qpos

START, SPEED, SECONDS = 1500, 1.0, 13.0
CMD = SpeedCommand([(0.0, SPEED, 0.0)])


def _jump(ctrl, jump_at, trace=False):
    """Generate a walk-then-jump roll-out from either controller; returns out[, tframe]."""
    if isinstance(ctrl, MotionGraph):
        r = ctrl.follow_with_jump(CMD, SECONDS, START, jump_at=jump_at, return_trace=trace)
    else:
        r = ctrl.generate(CMD, SECONDS, START, jump_at=jump_at, return_trace=trace)
    return (r[0], r[1]) if trace else r


def _marker(out, target_x=None):
    air = out[:, 2] > 0.90
    ay = float(out[out[:, 2].argmax(), 1])
    def fn(t):
        i = min(t, len(out) - 1)
        m = []
        if target_x is not None:
            m.append(([target_x, ay, 0.05], 0.14, [0.2, 1, 0.2, 1]))     # green target line
        if air[i]:
            m.append(([out[i, 0], out[i, 1], out[i, 2] + 0.4], 0.08, [1.0, 0.55, 0.0, 1]))
        return m
    return fn


def gen_task1(ctrl, clean=True):
    out, tf = _jump(ctrl, jump_at=5.5, trace=True)
    trace = trace_labels(tf, ctrl.lib)
    return (cleanup(out) if clean else out), _marker(out), trace


def gen_task2(ctrl, clean=True, target_x=5.0):
    best, err, bt = None, 1e9, None
    for ja in np.linspace(2.0, SECONDS - 4.0, 28):           # search the trigger time
        o = _jump(ctrl, float(ja))
        e = abs(float(o[o[:, 2].argmax(), 0]) - target_x)
        if e < err:
            best, err, bt = o, e, float(ja)
    print(f"  task2: jump_at={bt:.2f}s -> apex_x={best[best[:,2].argmax(),0]:.2f} (target {target_x}, err {err:.2f})")
    out, tf = _jump(ctrl, bt, trace=True)
    trace = trace_labels(tf, ctrl.lib)
    return (cleanup(out) if clean else out), _marker(out, target_x), trace


def run(tag, ctrl):
    out, mk, tr = gen_task1(ctrl)
    render_qpos(out, f"{C.OUT_DIR}/jump_{tag}_task1_oncommand.mp4", markers_fn=mk, trace=tr)
    out, mk, tr = gen_task2(ctrl)
    render_qpos(out, f"{C.OUT_DIR}/jump_{tag}_task2_fixedloc.mp4", markers_fn=mk, trace=tr)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    lib = load_library(C.JUMP_LIB_PATH)
    if which in ("mg", "both"):
        run("mg", MotionGraph(lib))
    if which in ("mm", "both"):
        run("mm", MotionMatcher(lib, traj_w=1.5, pose_w=1.0))
