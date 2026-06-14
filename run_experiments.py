"""Two path experiments for the GenoView motion matcher.

  exp1: walk the path (0,0)->(6,0)->(6,6)->(0,6)->(0,0)->(6,0).
  exp2: same path, but a box at (3,0) (crossed twice) the robot must jump over.

The matcher is reactive (no planner): it is STEERED toward each corner (a go-to-point heading
fed to the controller) and reactively jumps when the box is ahead and within one jump's reach.
So corners are rounded and the box jumps are best-effort -- the controller follows the path
rather than planning to it.

Usage: python run_experiments.py
"""
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.motion_matching import MotionMatcher
from motiongraph.mm_features import yaw_quat
from motiongraph.cleanup import cleanup
from motiongraph.render import render_qpos, trace_labels
from run_jump import _rz, START

PATH = [(6, 0), (6, 6), (0, 6), (0, 0), (6, 0)]
BOX = np.array([3.0, 0.0])
CAM = dict(cam_fixed=(3, 3, 0.4), cam_dist=12.5, cam_elev=-42, cam_azim=90, width=1000, height=1000)


def _corner_markers():
    def fn(t):
        return [([x, y, 0.05], 0.12, [0.2, 0.6, 1.0, 1]) for x, y in [(0, 0)] + PATH]
    return fn


def _box_dict(mm):
    e = int(mm.best_jump_entry(START)[0])
    jk = next(j for j, t in enumerate(mm.lib["jump_takeoff"]) if mm.lib["clip_id"][t] == mm.clip_id[e])
    half = [float(h) for h in mm.lib["jump_box"][jk]]
    return dict(pos=[BOX[0], BOX[1], half[2]], half=half, mat=_rz(0.0),
                rgba=[0.96, 0.45, 0.10, 1.0], label=f"BOX ({BOX[0]:.0f}, {BOX[1]:.0f})")


# -------- the GenoView controller, STEERED (reactive) --------
def gen_mm(mm, jump=False, clean=True, speed=1.0, reach=1.2, max_seconds=80):
    """Drive the step-based GenoView MM along the path by go-to-point steering: each frame
    feed it (speed, heading-toward-the-current-corner). Over the box, reactively trigger a
    jump when the box is ahead (heading-aligned) and within one jump's forward reach -- no
    planner, so corners round and box jumps are best-effort (GenoView-style)."""
    mm.reset(START)
    mm.rootPos = np.array([0.0, 0.0, mm.rootPos[2]])      # start at the world origin, facing +x
    mm.rootYaw = 0.0; mm.rootRot = yaw_quat(0.0); mm.desiredDir = np.array([1.0, 0.0, 0.0])
    box = BOX if jump else None
    wps = [(0, 0)] + PATH
    out, tframe, wp, armed, prev = [], [], 1, True, np.zeros(2)
    for _ in range(int(max_seconds * C.FPS)):
        pos, yaw = mm.rootPos[:2].copy(), mm.rootYaw
        while wp < len(wps):                              # advance corner when reached or passed
            tgt = np.array(wps[wp], float); seg = tgt - prev
            passed = seg @ (pos - tgt) > 0 if seg @ seg > 1e-6 else False
            if np.linalg.norm(tgt - pos) < reach or passed:
                prev = tgt; wp += 1
            else:
                break
        if wp >= len(wps):
            break
        tgt = np.array(wps[wp], float)
        heading = float(np.arctan2(tgt[1] - pos[1], tgt[0] - pos[0]))
        if box is not None and not mm.jumping:            # reactive box jump (box-ahead-and-in-reach)
            d, hv = box - pos, np.array([np.cos(yaw), np.sin(yaw)])
            aligned = abs(((np.arctan2(d[1], d[0]) - yaw + np.pi) % (2 * np.pi)) - np.pi) < 0.5
            je = mm.best_jump_entry()
            fwd = float(mm.qpos[mm.jump_apex_of[je[0]], 0] - mm.qpos[je[0], 0]) if je else 0.0
            if pos[0] < 1.0:
                armed = True
            if je and armed and aligned and 0 < (d @ hv) <= fwd:
                mm.trigger_jump(); armed = False
        out.append(mm.step(speed, heading)); tframe.append(mm.cur)
    out = np.asarray(out)
    if clean:
        out = cleanup(out)
    boxes = [_box_dict(mm)] if jump else None
    return out, _corner_markers(), trace_labels(tframe, mm.lib), boxes, mm.gizmo_trace


def _render(out, mk, tr, bx, gz, name):
    render_qpos(out, f"{C.OUT_DIR}/{name}.mp4", markers_fn=mk, trace=tr, boxes=bx, gizmo=gz, **CAM)


if __name__ == "__main__":
    mm = MotionMatcher(load_library(C.LIB_PATH))   # GenoView MM on the mirrored lib
    _render(*gen_mm(mm, jump=False), "exp1_mm_path")
    _render(*gen_mm(mm, jump=True), "exp2_mm_path_jump")
