"""Two path experiments, for both motion graph (MG) and motion matching (MM).

  exp1: walk the path (0,0)->(6,0)->(6,6)->(0,6)->(0,0)->(6,0).
  exp2: same path, but a box at (3,0) (crossed twice) the robot must jump over.

MG uses the A* planner (plan_to): corner-to-corner navigation + an A*-planned approach to
each jump, so corners are hit and both apexes land on the box. MM is the GenoView controller
(reactive, no planner): it is STEERED toward each corner (go-to-point heading) and reactively
jumps when the box is ahead and within one jump's reach -- so corners are rounded and the box
jump is best-effort (the 1st lands, the 2nd can drift). That gap is exactly what MG's planner
closes.

Usage: python run_experiments.py [mg|mm|both]
"""
import sys
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.commands import SpeedCommand
from motiongraph.motion_graph import MotionGraph
from motiongraph.motion_matching import MotionMatcher
from motiongraph.mm_features import yaw_quat
from motiongraph.cleanup import cleanup
from motiongraph.render import render_qpos, trace_labels
from motiongraph.kinematics import transform_qpos, alignment_to, rotz
from run_jump import _plan_jump, _rz, _yaw, START

CMD = SpeedCommand([(0.0, 1.0, 0.0)])
# path corners with the facing to hold on arrival (toward the next corner)
PATH = [(6, 0), (6, 6), (0, 6), (0, 0), (6, 0)]
FACE = [np.pi / 2, np.pi, -np.pi / 2, 0.0, 0.0]
BOX = np.array([3.0, 0.0])
CAM = dict(cam_fixed=(3, 3, 0.4), cam_dist=12.5, cam_elev=-42, cam_azim=90, width=1000, height=1000)


def _corner_markers():
    def fn(t):
        return [([x, y, 0.05], 0.12, [0.2, 0.6, 1.0, 1]) for x, y in [(0, 0)] + PATH]
    return fn


def _box_dict(g):
    e = int(g.best_jump_entry(START)[0])
    jk = next(j for j, t in enumerate(g.lib["jump_takeoff"]) if g.lib["clip_id"][t] == g.clip_id[e])
    half = [float(h) for h in g.lib["jump_box"][jk]]
    return dict(pos=[BOX[0], BOX[1], half[2]], half=half, mat=_rz(0.0),
                rgba=[0.96, 0.45, 0.10, 1.0], label=f"BOX ({BOX[0]:.0f}, {BOX[1]:.0f})")


# -------- A*-planned navigation (identical for MG and MM) --------
# Both controllers expose plan_to (shared A* in planner.py): MG plans over graph edges,
# MM over its feature-NN transitions. So one planned generator drives both.
def _plan_walk(ctrl, F, cxy, cyaw, txy, tyaw, sec=8.0):
    # world corner (txy,tyaw) -> the planner's local frame (current pose cxy,cyaw at origin):
    rel = rotz(-cyaw) @ (np.asarray(txy, float) - cxy)          # target xy in plan-local frame
    # ease="pose": pose-continuous hand-off WITHOUT dragging the root to the corner, so the
    # path rounds corners naturally instead of foot-skating to each exact corner point.
    loc = ctrl.plan_to(CMD, sec, int(F), rel, float(tyaw - cyaw), int(F), ease="pose")
    dy, pv, of = alignment_to(loc[0, :2], _yaw(loc[0]), cxy, cyaw)   # re-anchor plan back into world
    seg = transform_qpos(loc, dy, pv, of)
    return seg, int(F), seg[-1, :2], _yaw(seg[-1])               # next start: end world xy + heading


def gen_planned(ctrl, jump=False, clean=True):
    """A*-plan the path corner-to-corner; before each (..)->(6,0) leg (the box lies on it)
    A*-plan the approach to a jump and play the jump, so both apexes land on the box."""
    cxy, cyaw, F, segs = np.zeros(2), 0.0, START, []
    for txy, tyaw in [(PATH[i], FACE[i]) for i in range(len(PATH))]:
        if jump and txy == (6, 0):                          # box (3,0) lies on the two ->(6,0) legs
            seg, F, cxy, cyaw, *_ = _plan_jump(ctrl, F, cxy, cyaw, BOX)
            segs.append(seg)
        seg, F, cxy, cyaw = _plan_walk(ctrl, F, cxy, cyaw, txy, tyaw)
        segs.append(seg)
    out = np.concatenate(segs)
    if clean:
        out = cleanup(out)
    boxes = [_box_dict(ctrl)] if jump else None
    return out, _corner_markers(), None, boxes


# -------- motion matching: the GenoView controller, STEERED (reactive) --------
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
    return out, _corner_markers(), trace_labels(tframe, mm.lib), boxes


def _render(out, mk, tr, bx, name):
    render_qpos(out, f"{C.OUT_DIR}/{name}.mp4", markers_fn=mk, trace=tr, boxes=bx, **CAM)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("mg", "both"):
        g = MotionGraph(load_library(C.LOCO_LIB_PATH), n_neighbors=28, tgt_stride=1)
        _render(*gen_planned(g, jump=False), "exp1_mg_path")
        _render(*gen_planned(g, jump=True), "exp2_mg_path_jump")
    if which in ("mm", "both"):
        mm = MotionMatcher(load_library(C.LOCO_MIRROR_LIB_PATH))   # GenoView MM on the mirrored lib
        _render(*gen_mm(mm, jump=False), "exp1_mm_path")
        _render(*gen_mm(mm, jump=True), "exp2_mm_path_jump")
