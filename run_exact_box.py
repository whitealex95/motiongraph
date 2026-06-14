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
from run_jump import _walk_jump, _box, _marker, SECONDS, START


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
