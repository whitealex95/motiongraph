"""Two path experiments, for both motion graph (MG) and motion matching (MM).

  exp1: walk the path (0,0)->(6,0)->(6,6)->(0,6)->(0,0)->(6,0).
  exp2: same path, but a box at (3,0) (crossed twice) the robot must jump over.

MG traces the path with A*-PLANNED corner-to-corner navigation (plan_to) and A*-plans
each jump approach, so corners are hit and jumps land on the box. MM has no planner, so it
follows the path reactively (go-to-point command) -- approximate, and its box jumps are
best-effort (documented limitation).

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
from motiongraph.g1_model import quat_wxyz_yaw
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


# ---------------- motion graph (A*-planned) ----------------
def _mg_plan_walk(g, F, cxy, cyaw, txy, tyaw, sec=8.0):
    rel = rotz(-cyaw) @ (np.asarray(txy, float) - cxy)
    loc = g.plan_to(CMD, sec, int(F), rel, float(tyaw - cyaw), int(F))
    dy, pv, of = alignment_to(loc[0, :2], _yaw(loc[0]), cxy, cyaw)
    seg = transform_qpos(loc, dy, pv, of)
    return seg, int(F), seg[-1, :2], _yaw(seg[-1])


def gen_mg(g, jump=False, clean=True):
    cxy, cyaw, F, segs, marks = np.zeros(2), 0.0, START, [], []
    plan = [(PATH[i], FACE[i]) for i in range(len(PATH))]
    for k, (txy, tyaw) in enumerate(plan):
        if jump and txy == (6, 0):                          # box (3,0) lies on the two (..)->(6,0) legs
            seg, F, cxy, cyaw, *_ = _plan_jump(g, F, cxy, cyaw, BOX)
            segs.append(seg)
        seg, F, cxy, cyaw = _mg_plan_walk(g, F, cxy, cyaw, txy, tyaw)
        segs.append(seg)
    out = np.concatenate(segs)
    if clean:
        out = cleanup(out)
    boxes = [_box_dict(g)] if jump else None
    return out, _corner_markers(), None, boxes


# ---------------- motion matching (reactive, best-effort) ----------------
def gen_mm(mm, jump=False, clean=True):
    """MM walks the path reactively: at each interval the command heads toward the current
    waypoint (go-to-point). No planner -> approximate corners; box jumps are best-effort."""
    out, tframe = mm.walk_path([(0, 0)] + PATH, BOX if jump else None, start_frame=START,
                               return_trace=True)
    if clean:
        out = cleanup(out)
    boxes = [_box_dict_mm(mm)] if jump else None
    return out, _corner_markers(), trace_labels(tframe, mm.lib), boxes


def _box_dict_mm(mm):
    e = int(mm.best_jump_entry(START)[0])
    jk = next(j for j, t in enumerate(mm.lib["jump_takeoff"]) if mm.lib["clip_id"][t] == mm.clip_id[e])
    half = [float(h) for h in mm.lib["jump_box"][jk]]
    return dict(pos=[BOX[0], BOX[1], half[2]], half=half, mat=_rz(0.0),
                rgba=[0.96, 0.45, 0.10, 1.0], label=f"BOX ({BOX[0]:.0f}, {BOX[1]:.0f})")


def _render(out, mk, tr, bx, name):
    render_qpos(out, f"{C.OUT_DIR}/{name}.mp4", markers_fn=mk, trace=tr, boxes=bx, **CAM)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    lib = load_library(C.JUMP_LIB_PATH)
    if which in ("mg", "both"):
        g = MotionGraph(lib, n_neighbors=28, tgt_stride=1)
        _render(*gen_mg(g, jump=False), "exp1_mg_path")
        _render(*gen_mg(g, jump=True), "exp2_mg_path_jump")
    if which in ("mm", "both"):
        mm = MotionMatcher(lib, traj_w=1.5, pose_w=1.0)
        _render(*gen_mm(mm, jump=False), "exp1_mm_path")
        _render(*gen_mm(mm, jump=True), "exp2_mm_path_jump")
