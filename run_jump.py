"""Jump-skill demos on the G1 with the GenoView motion matcher.

  task1: walk forward at 1 m/s, then JUMP on a trigger (action goal, not a target).
  task2: jump so the apex lands near a FIXED x -- the trigger time is chosen by a search
         over candidates (walk further before jumping -> apex lands further along +x).
  raw:   render the original jump clips with their predefined box (a sanity view).

A jump is only ever entered from its pre-take-off run-up (never mid-air). Videos show a HUD
(clip / frame index) and flash on transitions.

Usage: python run_jump.py [task1|task2|raw|all]   (default: all)
"""
import sys
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.motion_matching import MotionMatcher
from motiongraph.mm_features import yaw_quat
from motiongraph.cleanup import cleanup
from motiongraph.render import trace_labels, render_qpos

START, SECONDS = 1500, 13.0      # a walk frame; roll-out length


def _walk_jump(mm, jump_at, speed=1.0):
    """Reset the matcher to the world origin facing +x, walk forward, and trigger one jump
    at `jump_at` seconds; ride it through landing. Returns (qpos (T,36), library indices)."""
    mm.reset(START)
    mm.rootPos = np.array([0.0, 0.0, mm.rootPos[2]]); mm.rootYaw = 0.0
    mm.rootRot = yaw_quat(0.0); mm.desiredDir = np.array([1.0, 0.0, 0.0])
    out, tf = [], []
    for s in range(int(SECONDS * C.FPS)):
        if s * C.DT >= jump_at and not mm.jumping:
            mm.trigger_jump()
        out.append(mm.step(speed, 0.0)); tf.append(mm.cur)
    return np.asarray(out), np.array(tf)


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


def _box(mm, out, tframe, pos=None, label=None):
    """A box (half-extents from the jump clip used), at a predefined (x, y) -- or at the
    jump apex. It sits on the ground, oriented along the jump (when free)."""
    lib = mm.lib
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


def gen_task1(mm, clean=True):
    out, tf = _walk_jump(mm, jump_at=5.5)
    box = _box(mm, out, tf, label="box (jumped over)")
    return (cleanup(out) if clean else out), _marker(out), trace_labels(tf, mm.lib), box


def gen_task2(mm, clean=True, target_x=5.0):
    best, err, bt = None, 1e9, None
    for ja in np.linspace(2.0, SECONDS - 4.0, 28):           # search the trigger time
        o, _ = _walk_jump(mm, float(ja))
        a = int(o[:, 2].argmax())
        e = abs(float(o[a, 0]) - target_x)                   # apex x distance to the target
        if e < err:
            best, err, bt = o, e, float(ja)
    out, tf = _walk_jump(mm, bt)
    a = int(out[:, 2].argmax())
    print(f"  task2: jump_at={bt:.2f}s -> apex x={out[a,0]:.2f} (target x={target_x}, err {err:.2f})")
    box = _box(mm, out, tf, pos=(target_x, float(out[a, 1])),
               label=f"PREDEFINED BOX  (x = {target_x:.1f})")
    return (cleanup(out) if clean else out), _marker(out, target_x), trace_labels(tf, mm.lib), box


def render_raw(lib):
    """Render the ORIGINAL jump clips with their predefined box (a sanity view)."""
    for k, cid_name in enumerate(["walk_jump_walk", "walk_jump_walk2"]):
        cid = list(lib["clip_names"]).index(cid_name)
        gi = np.where(lib["clip_id"] == cid)[0]
        q = lib["qpos"][gi]
        apex = int(lib["jump_apex"][k] - gi[0])
        half = [float(h) for h in lib["jump_box"][k]]
        a, b = max(0, apex - 5), min(len(q) - 1, apex + 5)
        yaw = float(np.arctan2(q[b, 1] - q[a, 1], q[b, 0] - q[a, 0]))
        box = dict(pos=[float(q[apex, 0]), float(q[apex, 1]), half[2]], half=half, mat=_rz(yaw),
                   rgba=[0.96, 0.45, 0.10, 1.0],
                   label=f"box {2*half[0]:.2f}x{2*half[1]:.2f}x{2*half[2]:.2f}m")
        render_qpos(q[max(0, apex - 45):apex + 55], f"{C.OUT_DIR}/jump_raw_{cid_name}.mp4", box=box)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    lib = load_library(C.LIB_PATH)               # GMR loco+jump, mirrored (GenoView MM)
    if which in ("raw", "all"):
        render_raw(lib)
    if which in ("task1", "task2", "all"):
        mm = MotionMatcher(lib)
        if which in ("task1", "all"):
            out, mk, tr, bx = gen_task1(mm)
            render_qpos(out, f"{C.OUT_DIR}/jump_mm_task1_oncommand.mp4", markers_fn=mk, trace=tr,
                        box=bx, gizmo=mm.gizmo_trace)
        if which in ("task2", "all"):
            out, mk, tr, bx = gen_task2(mm)
            render_qpos(out, f"{C.OUT_DIR}/jump_mm_task2_fixedloc.mp4", markers_fn=mk, trace=tr,
                        box=bx, gizmo=mm.gizmo_trace)
