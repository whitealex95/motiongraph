"""Two path experiments, for both motion graph (MG) and motion matching (MM).

  exp1: walk the path (0,0)->(6,0)->(6,6)->(0,6)->(0,0)->(6,0).
  exp2: same path, but a box at (3,0) (crossed twice) the robot must jump over.

Both MG and MM use the SAME A* planner (plan_to, shared planner.py): corner-to-corner
navigation + an A*-planned approach to each jump, so corners are hit and both apexes land
on the box. MG plans over its graph edges; MM plans over its feature-NN transitions. For
contrast we also render reactive MM (no planner): it traces the square but its 2nd box jump
drifts -- the gap A* closes.

Usage: python run_experiments.py [mg|mm|both]
"""
import sys
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.commands import SpeedCommand
from motiongraph.motion_graph import MotionGraph
from motiongraph.motion_matching import MotionMatcher
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


# -------- reactive MM (no planner) -- kept to show the contrast --------
def gen_mm_reactive(mm, jump=False, clean=True):
    """MM walks the path reactively: each interval the command heads toward the current
    waypoint (go-to-point). No planner -> approximate corners; box jumps are best-effort
    (the 1st lands, the 2nd drifts), which is exactly what A* planning fixes."""
    out, tframe = mm.walk_path([(0, 0)] + PATH, BOX if jump else None, start_frame=START,
                               return_trace=True)
    if clean:
        out = cleanup(out)
    boxes = [_box_dict(mm)] if jump else None
    return out, _corner_markers(), trace_labels(tframe, mm.lib), boxes


def _render(out, mk, tr, bx, name):
    render_qpos(out, f"{C.OUT_DIR}/{name}.mp4", markers_fn=mk, trace=tr, boxes=bx, **CAM)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    lib = load_library(C.JUMP_LIB_PATH)
    if which in ("mg", "both"):
        g = MotionGraph(lib, n_neighbors=28, tgt_stride=1)
        _render(*gen_planned(g, jump=False), "exp1_mg_path")
        _render(*gen_planned(g, jump=True), "exp2_mg_path_jump")
    if which in ("mm", "both"):
        mm = MotionMatcher(lib, traj_w=1.5, pose_w=1.0)
        _render(*gen_planned(mm, jump=False), "exp1_mm_path")       # MM + A*
        _render(*gen_planned(mm, jump=True), "exp2_mm_path_jump")   # MM + A*: lands both jumps
        _render(*gen_mm_reactive(mm, jump=True), "exp2_mm_reactive")  # contrast: no planner
