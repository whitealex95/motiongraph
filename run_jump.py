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
from motiongraph.kinematics import transform_qpos, alignment_to, rotz
from motiongraph.g1_model import quat_wxyz_yaw

START, SPEED, SECONDS = 2640, 1.0, 13.0
CMD = SpeedCommand([(0.0, SPEED, 0.0)])


def _jump(ctrl, jump_at, trace=False, target_xy=None):
    """Generate a walk-then-jump roll-out from either controller; returns out[, tframe].
    target_xy steers the motion-graph walk toward the box (so it converges to the box y)."""
    if isinstance(ctrl, MotionGraph):
        r = ctrl.follow_with_jump(CMD, SECONDS, START, jump_at=jump_at,
                                  target_xy=target_xy, return_trace=trace)
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


def _rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _box(ctrl, out, tframe, pos=None, label=None):
    """A box (half-extents from the used jump clip), at a predefined (x, y) -- or at the
    jump apex. It sits on the ground, oriented along the jump (when free), and the
    character jumps over it."""
    lib = ctrl.lib
    al = int(out[:, 2].argmax())
    cid = lib["clip_id"][int(tframe[al])]
    half = lib["jump_box"][0]
    for k, t in enumerate(lib["jump_takeoff"]):
        if lib["clip_id"][t] == cid:
            half = lib["jump_box"][k]
            break
    half = [float(h) for h in half]
    if pos is None:                                          # free: at the apex, along the jump
        a, b = max(0, al - 5), min(len(out) - 1, al + 5)
        yaw = float(np.arctan2(out[b, 1] - out[a, 1], out[b, 0] - out[a, 0]))
        bxy = [float(out[al, 0]), float(out[al, 1])]
    else:                                                    # predefined (x, y), axis-aligned (+x)
        yaw, bxy = 0.0, [float(pos[0]), float(pos[1])]
    return dict(pos=[bxy[0], bxy[1], half[2]], half=half, mat=_rz(yaw),
                rgba=[0.96, 0.45, 0.10, 1.0], label=label)


def gen_task1(ctrl, clean=True):
    out, tf = _jump(ctrl, jump_at=5.5, trace=True)
    trace = trace_labels(tf, ctrl.lib)
    box = _box(ctrl, out, tf, label="box (jumped over)")
    return (cleanup(out) if clean else out), _marker(out), trace, box


def gen_task2(ctrl, clean=True, target=(5.0, 0.0)):
    target = np.asarray(target, float)                       # PREDEFINED box at (x, y=0)
    best, err, bt = None, 1e9, None
    for ja in np.linspace(2.0, SECONDS - 4.0, 28):           # search the trigger time
        o = _jump(ctrl, float(ja), target_xy=target)         # MG steers toward the box
        a = int(o[:, 2].argmax())
        e = float(np.linalg.norm(o[a, 0:2] - target))        # apex distance to the box (x AND y)
        if e < err:
            best, err, bt = o, e, float(ja)
    a = int(best[:, 2].argmax())
    print(f"  task2: jump_at={bt:.2f}s -> apex xy={best[a,:2].round(2)} (box {target}, dist {err:.2f})")
    out, tf = _jump(ctrl, bt, trace=True, target_xy=target)
    trace = trace_labels(tf, ctrl.lib)
    # box is PREDEFINED at a fixed (x, y): the character walks to it, then jumps over it.
    box = _box(ctrl, out, tf, pos=target, label=f"PREDEFINED BOX  ({target[0]:.1f}, {target[1]:.1f})")
    return (cleanup(out) if clean else out), _marker(out), trace, box       # box = the target


def render_raw(lib):
    """Render the ORIGINAL jump clips with their predefined box (a sanity view)."""
    for k, (cid_name) in enumerate(["walk_jump_walk", "walk_jump_walk2"]):
        cid = list(lib["clip_names"]).index(cid_name)
        gi = np.where(lib["clip_id"] == cid)[0]
        q = lib["qpos"][gi]
        apex = int(lib["jump_apex"][k] - gi[0])
        half = [float(h) for h in lib["jump_box"][k]]
        a, b = max(0, apex - 5), min(len(q) - 1, apex + 5)
        yaw = float(np.arctan2(q[b, 1] - q[a, 1], q[b, 0] - q[a, 0]))
        box = dict(pos=[float(q[apex, 0]), float(q[apex, 1]), half[2]], half=half,
                   mat=_rz(yaw), rgba=[0.96, 0.45, 0.10, 1.0], label=f"box {2*half[0]:.2f}x{2*half[1]:.2f}x{2*half[2]:.2f}m")
        seg = q[max(0, apex - 45):apex + 55]
        render_qpos(seg, f"{C.OUT_DIR}/jump_raw_{cid_name}.mp4", box=box)


def _boxes_from_route(ctrl, out, tframe):
    """One box per jump, placed at that jump's apex, oriented along the jump."""
    lib = ctrl.lib
    air = out[:, 2] > 0.95
    idx = np.where(air)[0]
    boxes = []
    if len(idx):
        for k, s in enumerate(np.split(idx, np.where(np.diff(idx) > 5)[0] + 1)):
            ap = int(s[out[s, 2].argmax()])
            cid = lib["clip_id"][int(tframe[ap])]
            half = lib["jump_box"][0]
            for j, t in enumerate(lib["jump_takeoff"]):
                if lib["clip_id"][t] == cid:
                    half = lib["jump_box"][j]
                    break
            half = [float(h) for h in half]
            a, b = max(0, ap - 5), min(len(out) - 1, ap + 5)
            yaw = float(np.arctan2(out[b, 1] - out[a, 1], out[b, 0] - out[a, 0]))
            boxes.append(dict(pos=[float(out[ap, 0]), float(out[ap, 1]), half[2]], half=half,
                              mat=_rz(yaw), rgba=[0.96, 0.45, 0.10, 1.0],
                              label=f"BOX {k + 1}  ({out[ap,0]:.1f}, {out[ap,1]:.1f})"))
    return boxes


def gen_loop(ctrl, clean=True):
    """Jump over a box, walk a loop, jump again -- two world-anchored jumps with the
    walk in-betweened between them (one box per jump apex)."""
    circ = [(round(5 + 5 * np.sin(a), 1), round(5 - 5 * np.cos(a), 1))
            for a in np.linspace(np.deg2rad(25), np.deg2rad(330), 9)]
    wps = [(5, 0, True)] + [(x, y, False) for x, y in circ] + [(2.0, 0.0, False), (5, 0, True), (9, 0, False)]
    out, tf, tp = ctrl.follow_route(wps, start_frame=START, max_seconds=60,
                                    straighten=0.7, return_trace=True)
    if clean:
        out = cleanup(out)
    return out, _marker(out), trace_labels(tf, ctrl.lib), _boxes_from_route(ctrl, out, tf)


def _yaw(frame):
    return float(quat_wxyz_yaw(frame[None, 3:7])[0])


def _place(seg, end_xy, end_yaw):
    """Re-anchor a segment so its first frame continues from (end_xy, end_yaw)."""
    dy, pv, of = alignment_to(seg[0, :2], _yaw(seg[0]), end_xy, end_yaw)
    return transform_qpos(seg, dy, pv, of)


def _plan_jump(g, start_frame, cur_xy, cur_yaw, box, seconds=6.0):
    """A*-plan a precise in-between to a jump's pre-take-off entry placed so the apex
    lands on `box`, then play the jump. Returns the world segment + end state + entry."""
    entry, land = g.best_jump_entry(int(start_frame))
    apex = g.jump_apex_of[entry]
    disp = rotz(-g.yaw[entry]) @ (g.qpos[apex, 0:2] - g.qpos[entry, 0:2])   # entry->apex, entry-local
    world_target = box - disp                                              # so the apex lands on box
    local_target = rotz(-cur_yaw) @ (world_target - cur_xy)                # plan in its own frame
    local = g.plan_to(CMD, seconds, int(start_frame), local_target, float(-cur_yaw), int(entry))
    approach = _place(local, cur_xy, cur_yaw)
    e_xy, e_yaw = approach[-1, :2], _yaw(approach[-1])
    dy, pv, of = alignment_to(g.xy[entry], g.yaw[entry], e_xy, e_yaw)
    after_end = land + 1 + C.PHASE_TOUCHDOWN + C.PHASE_AFTER       # ready..after
    jump = transform_qpos(g.qpos[np.arange(entry, after_end)], dy, pv, of)
    seg = np.concatenate([approach, jump])
    tf = np.concatenate([np.full(len(approach), entry), np.arange(entry, after_end)])
    return seg, int(land + 29), seg[-1, :2], _yaw(seg[-1]), entry, tf


def gen_loop_same_box(g, box=(5.0, 0.0), clean=True):
    """HARD constraint: jump over ONE box, loop, jump over the SAME box again. Both jump
    approaches are A*-PLANNED in-betweens to the jump's pre-take-off entry pose (precise,
    no drift), so both apexes land on the same box; the loop between is the greedy graph."""
    box = np.asarray(box, float)
    seg1, F1, xy1, yaw1, e1, tf1 = _plan_jump(g, START, np.zeros(2), 0.0, box)   # jump 1
    # loop: greedy navigation of a world circle around the box, starting from jump1's end
    circ = [(round(box[0] + 5 * np.sin(a), 1), round(box[1] + 5 - 5 * np.cos(a), 1))
            for a in np.linspace(np.deg2rad(25), np.deg2rad(330), 9)]
    wps = [(x, y, False) for x, y in circ] + [(box[0] - 4.0, box[1], False)]
    init = alignment_to(g.xy[F1], g.yaw[F1], xy1, yaw1)
    seg2, tf2, _, (F2, xy2, yaw2) = g.follow_route(wps, start_frame=F1, init_align=init,
                                                   max_seconds=45, straighten=0.7,
                                                   return_trace=True, return_state=True)
    seg3, F3, xy3, yaw3, e3, tf3 = _plan_jump(g, F2, xy2, yaw2, box)             # jump 2 (same box)
    out = np.concatenate([seg1, seg2, seg3])
    tf = np.concatenate([tf1, tf2, tf3])

    jk = next(j for j, t in enumerate(g.lib["jump_takeoff"]) if g.lib["clip_id"][t] == g.clip_id[e1])
    half = [float(h) for h in g.lib["jump_box"][jk]]
    boxd = dict(pos=[box[0], box[1], half[2]], half=half, mat=_rz(0.0),
                rgba=[0.96, 0.45, 0.10, 1.0], label=f"BOX ({box[0]:.1f}, {box[1]:.1f})  (same box x2)")
    if clean:
        out = cleanup(out)
    return out, _marker(out), trace_labels(tf, g.lib), [boxd]


def run(tag, ctrl):
    out, mk, tr, bx = gen_task1(ctrl)
    render_qpos(out, f"{C.OUT_DIR}/jump_{tag}_task1_oncommand.mp4", markers_fn=mk, trace=tr, box=bx)
    out, mk, tr, bx = gen_task2(ctrl)
    render_qpos(out, f"{C.OUT_DIR}/jump_{tag}_task2_fixedloc.mp4", markers_fn=mk, trace=tr, box=bx)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    lib = load_library(C.JUMP_LIB_PATH)
    if which == "loop":                                  # jump -> loop -> jump (two jumps)
        out, mk, tr, bx = gen_loop(MotionGraph(lib))
        render_qpos(out, f"{C.OUT_DIR}/jump_mg_loop_twice.mp4", markers_fn=mk, trace=tr, boxes=bx)
    if which == "samebox":                               # SAME box twice (loop + A*-planned return)
        g = MotionGraph(lib, n_neighbors=28, tgt_stride=1)
        out, mk, tr, bx = gen_loop_same_box(g)
        render_qpos(out, f"{C.OUT_DIR}/jump_mg_samebox_twice.mp4", markers_fn=mk, trace=tr, boxes=bx)
    if which in ("raw", "both"):
        render_raw(lib)
    if which in ("mg", "both"):
        run("mg", MotionGraph(lib))
    if which in ("mm", "both"):
        run("mm", MotionMatcher(load_library(C.LOCO_MIRROR_LIB_PATH)))   # GenoView MM on mirrored loco
