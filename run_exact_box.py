"""Exact box-jump for the MM-only controller.

The reactive trigger in `run_experiments.py` is *best-effort*: the jump is a locked clip
segment played from wherever the steered walk happens to arrive, so the apex lands ~0.1-0.2 m
off a fixed box. Here we make it **exact and repeatable** -- a "planned" sequence (scheduled
trigger + motion-warp, not graph A*) where the apex lands on the box every run:

  1. Search the jump-trigger time so the apex lands *near* the fixed box (as in run_jump.task2).
  2. Apply a small **airborne root-warp**: a smoothstep bump over the ascent/descent that nudges
     the apex onto the box *exactly*. The warp touches only frames where both feet are off the
     ground (no foot skating) and ramps back to zero by landing (jumps stay independent, so a
     whole course of boxes works with no drift accumulation).

Two scenarios:
  single -- one fixed box, apex warped exactly onto it.
  course -- several fixed boxes in a row; reactively trigger each jump, warp each apex onto its
            own box.

Usage: conda activate deploy_mujoco; MUJOCO_GL=glfw python run_exact_box.py [single|course|all]
"""
import sys
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.motion_matching import MotionMatcher
from motiongraph.mm_features import yaw_quat
from motiongraph.cleanup import cleanup
from motiongraph.render import trace_labels, render_qpos
from run_jump import _walk_jump, _box, _marker, SECONDS, START


# ---------------------------------------------------------------- warp helpers
def _ground_thr(z):
    """Walk-height baseline + a threshold halfway to the apex (a frame is 'airborne' above it)."""
    ground = float(np.median(np.sort(z)[: max(1, len(z) // 2)]))
    return ground, ground + 0.5 * (float(z.max()) - ground)


def _flight_segments(z, thr):
    """Contiguous airborne spans -> [(takeoff, apex, land), ...] (frame indices)."""
    air = z > thr
    segs, i, n = [], 0, len(z)
    while i < n:
        if air[i]:
            j = i
            while j < n and air[j]:
                j += 1
            segs.append((i, i + int(z[i:j].argmax()), j - 1))
            i = j
        else:
            i += 1
    return segs


def _bump_warp(out, t0, a, t1, target_xy):
    """Shift the airborne arc [t0..t1] so out[a, 0:2] == target_xy, ramping the correction in
    over the ascent (t0->a) and back out over the descent (a->t1) with a smoothstep. Airborne
    only (no skating); the arc returns to its original line by t1 (jumps stay independent)."""
    d = np.asarray(target_xy, float) - out[a, 0:2]
    t = np.arange(len(out)).astype(float)
    w = np.zeros(len(out))
    up = (t >= t0) & (t <= a)
    dn = (t > a) & (t <= t1)
    su = np.clip((t[up] - t0) / max(1, a - t0), 0, 1)
    sd = np.clip((t1 - t[dn]) / max(1, t1 - a), 0, 1)
    w[up] = su * su * (3 - 2 * su)
    w[dn] = sd * sd * (3 - 2 * sd)
    out[:, 0] += d[0] * w
    out[:, 1] += d[1] * w
    return out


# ---------------------------------------------------------------- single box
def gen_single(mm, box=(5.0, 0.0), clean=True):
    """One fixed box at `box`: search the trigger time, then warp the apex exactly onto it."""
    tx, ty = box
    err, bt = 1e9, None
    for ja in np.linspace(2.0, SECONDS - 4.0, 28):            # search the trigger time
        o, _ = _walk_jump(mm, float(ja))
        a = int(o[:, 2].argmax())
        e = abs(float(o[a, 0]) - tx)
        if e < err:
            err, bt = e, float(ja)
    out, tf = _walk_jump(mm, bt)                              # final rollout (sets mm.gizmo_trace)
    if clean:
        out = cleanup(out)
    _, thr = _ground_thr(out[:, 2])
    seg = min(_flight_segments(out[:, 2], thr), key=lambda s: abs(s[1] - int(out[:, 2].argmax())))
    _bump_warp(out, *seg, (tx, ty))
    a = seg[1]
    print(f"  single: jump_at={bt:.2f}s  search err={err:.3f} m  ->  warped apex "
          f"({out[a, 0]:.3f}, {out[a, 1]:.3f})  target ({tx:.2f}, {ty:.2f})")
    boxd = _box(mm, out, tf, pos=(tx, ty), label=f"EXACT BOX ({tx:.0f}, {ty:.0f})")
    return out, _marker(out, target_x=tx), trace_labels(tf, mm.lib), boxd, mm.gizmo_trace


# ---------------------------------------------------------------- multi-box course
def _walk_course(mm, boxes, speed=1.0, seconds=22.0):
    """Walk +x past a row of fixed boxes; reactively trigger a jump as each box comes within one
    jump's forward reach (re-arming after each box is passed). Returns (qpos (T,36), indices)."""
    mm.reset(START)
    mm.rootPos = np.array([0.0, 0.0, mm.rootPos[2]]); mm.rootYaw = 0.0
    mm.rootRot = yaw_quat(0.0); mm.desiredDir = np.array([1.0, 0.0, 0.0])
    bi, armed, out, tf = 0, True, [], []
    for _ in range(int(seconds * C.FPS)):
        pos, yaw = mm.rootPos[:2].copy(), mm.rootYaw
        while bi < len(boxes) and pos[0] > boxes[bi][0] + 0.5:   # passed this box -> next
            bi += 1; armed = True
        if bi >= len(boxes) and not mm.jumping and pos[0] > boxes[-1][0] + 1.5:
            break
        if bi < len(boxes) and not mm.jumping:                   # reactive trigger toward box bi
            box = np.array(boxes[bi], float)
            hv = np.array([np.cos(yaw), np.sin(yaw)])
            ahead = float((box - pos) @ hv)
            je = mm.best_jump_entry()
            fwd = float(mm.qpos[mm.jump_apex_of[je[0]], 0] - mm.qpos[je[0], 0]) if je else 0.0
            if ahead > fwd + 0.4:
                armed = True
            if je and armed and 0 < ahead <= fwd:
                mm.trigger_jump(); armed = False
        out.append(mm.step(speed, 0.0)); tf.append(mm.cur)
    return np.asarray(out), np.array(tf)


def gen_course(mm, boxes=((3.0, 0.0), (6.0, 0.0), (9.0, 0.0)), clean=True):
    """A row of fixed boxes; jump each and warp every apex exactly onto its box."""
    out, tf = _walk_course(mm, boxes)
    if clean:
        out = cleanup(out)
    _, thr = _ground_thr(out[:, 2])
    segs = _flight_segments(out[:, 2], thr)
    for (t0, a, t1) in segs:                                  # match each flight to nearest box
        k = min(range(len(boxes)), key=lambda j: abs(boxes[j][0] - out[a, 0]))
        _bump_warp(out, t0, a, t1, boxes[k])
        print(f"  course: flight apex frame {a:4d}  ->  box {k + 1} {boxes[k]}  "
              f"(apex now {out[a, 0]:.3f}, {out[a, 1]:.3f})")
    boxds = [_box(mm, out, tf, pos=b, label=f"BOX {i + 1} ({b[0]:.0f},{b[1]:.0f})")
             for i, b in enumerate(boxes)]
    tops = [(b["pos"][0], b["pos"][1], b["pos"][2] + b["half"][2]) for b in boxds]

    def mk(t):
        return [([x, y, z], 0.07, [0.2, 1.0, 0.2, 1]) for (x, y, z) in tops]   # apex target dots
    return out, mk, trace_labels(tf, mm.lib), boxds, mm.gizmo_trace


# ---------------------------------------------------------------- render
SINGLE_CAM = dict(cam_dist=4.5, cam_elev=-12, cam_azim=120, width=900, height=620)
COURSE_CAM = dict(cam_fixed=(6.0, 0.0, 0.5), cam_dist=11.0, cam_elev=-10, cam_azim=90,
                  width=1100, height=560)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    mm = MotionMatcher(load_library(C.LIB_PATH))
    if which in ("single", "all"):
        out, mk, tr, bx, gz = gen_single(mm)
        render_qpos(out, f"{C.OUT_DIR}/exact_single_box.mp4", markers_fn=mk, trace=tr,
                    box=bx, gizmo=gz, **SINGLE_CAM)
    if which in ("course", "all"):
        out, mk, tr, bxs, gz = gen_course(mm)
        render_qpos(out, f"{C.OUT_DIR}/exact_course.mp4", markers_fn=mk, trace=tr,
                    boxes=bxs, gizmo=gz, **COURSE_CAM)
