"""Exact box-jump for the MM-only controller.

The reactive box trigger in `run_experiments.py` is *best-effort*: the locked jump segment plays
from wherever the steered walk arrives, so the apex misses a fixed box by ~0.1-0.2 m. Here we make
the apex land on the box **exactly and repeatably** -- a deterministic "planned" sequence -- WITHOUT
distorting the jump.

Why not warp the airborne arc: apex-x is a *step* function of the trigger time (the jump entry
snaps to a discrete `ready` frame, so apex-x jumps ~0.37 m between adjacent triggers). Forcing the
~0.1-0.2 m residual through the short airborne arc visibly distorts the horizontal velocity (the
jump appears to stall at the top). So instead we move the *geometry*, never the arc:

  single -- one FIXED box at a chosen (x, y): rigidly translate the whole rollout so the natural
            apex lands exactly on the box. A rigid shift changes no velocity => zero distortion;
            the only effect is the robot's start point moves by the (sub-0.2 m) residual.
  course -- a row of jumps: the box is LOCKED to each jump's natural apex ("motion baked into the
            object"). Exact by construction, again zero distortion.

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
from run_jump import _walk_jump, _box, _marker, _rz, SECONDS, START


# ---------------------------------------------------------------- helpers
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


def _rigid_to(out, target_xy):
    """Translate the WHOLE rollout so its (global) apex lands exactly on target_xy. A rigid shift,
    so every velocity/contact is preserved -- no distortion at all. Returns (new_out, apex_frame)."""
    out = out.copy()
    a = int(out[:, 2].argmax())
    d = np.asarray(target_xy, float) - out[a, 0:2]
    out[:, 0] += d[0]
    out[:, 1] += d[1]
    return out, a


def _box_tops(boxds):
    return [(b["pos"][0], b["pos"][1], b["pos"][2] + b["half"][2]) for b in boxds]


# ---------------------------------------------------------------- single fixed box
def gen_single(mm, box=(5.0, 0.0), clean=True):
    """One FIXED box at `box`. Coarse-search a trigger time whose natural apex is nearest the box
    (just to keep the robot starting near the origin), then rigidly translate the rollout so the
    apex sits exactly on the box -- exact and undistorted."""
    tx, ty = box
    err, bt = 1e9, None
    for ja in np.linspace(2.0, SECONDS - 4.0, 28):       # nearest natural apex => smallest shift
        o, _ = _walk_jump(mm, float(ja))
        e = abs(float(o[int(o[:, 2].argmax()), 0]) - tx)
        if e < err:
            err, bt = e, float(ja)
    out, tf = _walk_jump(mm, bt)                          # final rollout (sets mm.gizmo_trace)
    if clean:
        out = cleanup(out)
    out, a = _rigid_to(out, box)
    print(f"  single: jump_at={bt:.2f}s  rigid shift={err:.3f} m  ->  apex "
          f"({out[a, 0]:.3f}, {out[a, 1]:.3f}) == box ({tx:.2f}, {ty:.2f})  [no arc distortion]")
    boxd = _box(mm, out, tf, pos=(tx, ty), label=f"FIXED BOX ({tx:.0f}, {ty:.0f})")
    return out, _marker(out, target_x=tx), trace_labels(tf, mm.lib), boxd, mm.gizmo_trace


# ---------------------------------------------------------------- multi-box course
def _walk_jumps(mm, jump_times, speed=1.0, seconds=13.0):
    """Walk along the +x line and trigger a jump at each time in `jump_times`. A gentle go-to-point
    heading (aim 2 m ahead on y=0) holds the course straight despite reactive drift. Returns
    (qpos (T,36), indices)."""
    mm.reset(START)
    mm.rootPos = np.array([0.0, 0.0, mm.rootPos[2]]); mm.rootYaw = 0.0
    mm.rootRot = yaw_quat(0.0); mm.desiredDir = np.array([1.0, 0.0, 0.0])
    jt, out, tf = list(jump_times), [], []
    for s in range(int(seconds * C.FPS)):
        if jt and s * C.DT >= jt[0] and not mm.jumping:
            mm.trigger_jump(); jt.pop(0)
        pos = mm.rootPos[:2]
        heading = float(np.arctan2(-pos[1], 2.0))            # steer back onto the y=0 line
        out.append(mm.step(speed, heading)); tf.append(mm.cur)
    return np.asarray(out), np.array(tf)


def gen_course(mm, jump_times=(2.4, 5.4, 8.4), clean=True):
    """A row of jumps; each box is LOCKED to its jump's natural apex (motion baked into the box),
    so every apex lands exactly on its box with no warp."""
    out, tf = _walk_jumps(mm, jump_times)
    if clean:
        out = cleanup(out)
    _, thr = _ground_thr(out[:, 2])
    segs = _flight_segments(out[:, 2], thr)
    if segs:                                                 # trim the trailing walk after the last land
        end = min(len(out), segs[-1][2] + 30)
        out, tf = out[:end], tf[:end]
        segs = _flight_segments(out[:, 2], thr)
    boxes = [(float(out[a, 0]), float(out[a, 1])) for (_, a, _) in segs]
    boxds = [_box(mm, out, tf, pos=b, label=f"BOX {i + 1} ({b[0]:.2f}, {b[1]:.2f})")
             for i, b in enumerate(boxes)]
    for i, (_, a, _) in enumerate(segs):
        print(f"  course: jump {i + 1} apex frame {a:4d}  ->  box at "
              f"({out[a, 0]:.3f}, {out[a, 1]:.3f})  [baked]")
    tops = _box_tops(boxds)

    def mk(t):
        return [([x, y, z], 0.07, [0.2, 1.0, 0.2, 1]) for (x, y, z) in tops]   # apex == box dots
    return out, mk, trace_labels(tf, mm.lib), boxds, mm.gizmo_trace[:len(out)]


# ---------------------------------------------------------------- planned: bake motion into FIXED boxes
def _Rz(th, v):
    c, s = np.cos(th), np.sin(th)
    return np.array([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]]).T


def _entry_deltas(mm):
    """For every jump `ready` entry e: the apex offset in the entry's local frame, Δ_local(e).
    Because the jump root is integrated purely from the clip's velocities, the world apex is
    apex = entry_world + Rz(yaw_switch)·Δ_local(e) (approach-independent, up to the inertialization
    residual we calibrate out). This is the 'object baked into the clip'."""
    ents = [int(e) for e in mm.jump_enter]
    DL = np.zeros((len(ents), 2))
    for i, e in enumerate(ents):
        a = int(mm.jump_apex_of[e]); th = float(mm.simThetaDB[e])
        c, s = np.cos(-th), np.sin(-th)
        v = mm.simPosDB[a][:2] - mm.simPosDB[e][:2]
        DL[i] = [c * v[0] - s * v[1], s * v[0] + c * v[1]]
    return np.array(ents), DL


def _steady_walk(mm, n):
    mm.reset(START)
    mm.rootPos = np.array([0.0, 0.0, mm.rootPos[2]]); mm.rootYaw = 0.0
    mm.rootRot = yaw_quat(0.0); mm.desiredDir = np.array([1.0, 0.0, 0.0])
    for _ in range(n):
        mm.step(1.0, float(np.arctan2(-mm.rootPos[1], 2.0)))


def _probe_jump(mm, ents, DL, box, shortlist=14, ndelay=7, horizon=44):
    """What-if search at the current state: try the most-promising entries (by analytic apex), at
    each of the next few switch frames, by ACTUALLY simulating the locked jump and measuring the
    true apex. Returns (best_entry, switch_delay, predicted_apex) for the smallest apex-box error.
    Real simulation, so it captures inertialization exactly -- no warp, just selection."""
    order = np.argsort(np.linalg.norm(mm.rootPos[:2] + _Rz(mm.rootYaw, DL) - box, axis=1))
    cand = ents[order[:shortlist]]
    snap0 = mm.state()
    best = (1e9, int(cand[0]), 0, None)
    for delay in range(ndelay):
        snap_d = mm.state()
        for e in cand:
            mm.set_state(snap_d)
            mm.trigger_jump(entry=int(e))
            qs = np.array([mm.step(1.0, 0.0) for _ in range(horizon)])
            a = int(qs[:, 2].argmax())
            err = float(np.linalg.norm(qs[a, 0:2] - box))
            if err < best[0]:
                best = (err, int(e), delay, qs[a, 0:2].copy())
        mm.set_state(snap_d)
        mm.step(1.0, float(np.arctan2(-mm.rootPos[1], 2.0)))   # advance one walk frame
    mm.set_state(snap0)
    return best[1], best[2], best[3]


def _box_for_entry(mm, entry, pos, label):
    """A box at FIXED `pos`, sized from the jump clip that `entry` belongs to (so it clears)."""
    lib, cid = mm.lib, int(mm.clip_id[int(entry)])
    half = lib["jump_box"][0]
    for k, t in enumerate(lib["jump_takeoff"]):
        if int(lib["clip_id"][t]) == cid:
            half = lib["jump_box"][k]; break
    half = [float(h) for h in half]
    return dict(pos=[float(pos[0]), float(pos[1]), half[2]], half=half, mat=_rz(0.0),
                rgba=[0.96, 0.45, 0.10, 1.0], label=label)


def gen_planned(mm, boxes=((5.0, 0.0), (10.0, 0.0), (15.0, 0.0)), clean=True, seconds=26.0):
    """FIXED boxes at `boxes`: plan each jump so its apex lands on the box, choosing only WHICH
    `ready` entry (clip + switch point) and WHEN to switch -- no motion warp. An analytic apex
    estimate gates the approach; near the trigger point a real-simulation probe (`_probe_jump`)
    picks the entry+switch-frame whose true apex is closest to the fixed box."""
    ents, DL = _entry_deltas(mm)
    _steady_walk(mm, 0)                                   # reset to origin, facing +x
    bi, out, tf, chosen, n = 0, [], [], [], int(seconds * C.FPS)
    while len(out) < n:
        pos, yaw = mm.rootPos[:2].copy(), mm.rootYaw
        if bi < len(boxes) and not mm.jumping:
            box = np.array(boxes[bi], float)
            apex = pos + _Rz(yaw, DL)                            # analytic gate (rough)
            err = np.linalg.norm(apex - box, axis=1)
            err_n = np.linalg.norm(pos + np.array([np.cos(yaw), np.sin(yaw)]) * C.DT
                                   + _Rz(yaw, DL) - box, axis=1)
            if err.min() < 0.8 and err.min() <= err_n.min():    # near the analytic local min
                e_star, delay, ap = _probe_jump(mm, ents, DL, box)   # real-sim refinement
                for _ in range(delay):
                    h = float(np.arctan2(-mm.rootPos[1], 2.0))
                    out.append(mm.step(1.0, h)); tf.append(mm.cur)
                mm.trigger_jump(entry=e_star)
                chosen.append((bi, e_star, ap)); bi += 1
                continue
        out.append(mm.step(1.0, float(np.arctan2(-pos[1], 2.0)))); tf.append(mm.cur)
        if bi >= len(boxes) and not mm.jumping and pos[0] > boxes[-1][0] + 1.5:
            break
    out = np.asarray(out)
    if clean:
        out = cleanup(out)
    # report realized apex error at each box, build the fixed boxes
    _, thr = _ground_thr(out[:, 2])
    segs = _flight_segments(out[:, 2], thr)
    for (bi_, e_star, ap), seg in zip(chosen, segs):
        a = seg[1]
        bx = np.asarray(boxes[bi_], float)
        print(f"  planned: box {bi_ + 1} @ ({bx[0]:.1f},{bx[1]:.1f})  entry {e_star} "
              f"({mm.lib['clip_names'][int(mm.clip_id[e_star])]})  realized apex "
              f"({out[a, 0]:.3f}, {out[a, 1]:.3f})  err {np.linalg.norm(out[a, 0:2] - bx):.3f} m")
    boxds = [_box_for_entry(mm, e_star, boxes[bi_], f"BOX {bi_ + 1} ({boxes[bi_][0]:.0f}m)")
             for (bi_, e_star, _) in chosen]
    if segs:
        out, tf = out[:min(len(out), segs[-1][2] + 30)], tf[:min(len(tf), segs[-1][2] + 30)]
    tops = _box_tops(boxds)

    def mk(t):
        return [([x, y, z], 0.07, [0.2, 1.0, 0.2, 1]) for (x, y, z) in tops]
    return out, mk, trace_labels(tf, mm.lib), boxds, mm.gizmo_trace[:len(out)]


# ---------------------------------------------------------------- render
SINGLE_CAM = dict(cam_dist=4.5, cam_elev=-12, cam_azim=120, width=900, height=620)
COURSE_CAM = dict(cam_dist=5.0, cam_elev=-12, cam_azim=115, width=960, height=600)


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
    if which in ("planned", "all"):
        out, mk, tr, bxs, gz = gen_planned(mm)
        render_qpos(out, f"{C.OUT_DIR}/exact_planned.mp4", markers_fn=mk, trace=tr,
                    boxes=bxs, gizmo=gz, **COURSE_CAM)
