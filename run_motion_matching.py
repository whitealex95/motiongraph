"""Motion matching demos on the G1.

  task1: locomotion driven by a time-varying speed command.
  task2: motion in-betweening -- follow a command, then arrive at a terminal state
         at a fixed time (the far trajectory samples are steered to the target and
         the final pose is eased onto the terminal frame).

Usage: python run_motion_matching.py [task1|task2|both]
"""
import sys
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.commands import SpeedCommand, demo_speed_schedule
from motiongraph.motion_matching import MotionMatcher
from motiongraph.kinematics import transform_qpos, alignment_to, blend_qpos
from motiongraph.features import _local


def _cmd_marker(out, command):
    def fn(t):                       # t is the frame index (see render.py)
        i = min(t, len(out) - 1)
        spd, hd = command.state(i * C.DT)
        root = out[i, 0:3]
        tip = root[:2] + max(0.4, spd) * 0.5 * np.array([np.cos(hd), np.sin(hd)])
        return [([tip[0], tip[1], 1.2], 0.06, [1, 0.2, 0.2, 1])]   # red: command dir
    return fn


def task1():
    lib = load_library()
    mm = MotionMatcher(lib, traj_w=1.5, pose_w=1.0)
    cmd = demo_speed_schedule()
    out = mm.generate(cmd, seconds=15.0, start_frame=200)
    from motiongraph.render import render_qpos
    render_qpos(out, f"{C.OUT_DIR}/mm_task1_speed.mp4", markers_fn=_cmd_marker(out, cmd))


def task2():
    lib = load_library()
    mm = MotionMatcher(lib, traj_w=1.5, pose_w=1.0)
    T, tail = 9.0, 2.5
    target_xy = np.array([3.5, 1.5])
    target_yaw = np.deg2rad(90)
    term_frame = 500                                  # a mid-clip walk pose for the terminal
    # terminal qpos placed at the target world pose
    dy, pv, of = alignment_to(lib["qpos"][term_frame, 0:2], lib["yaw"][term_frame], target_xy, target_yaw)
    term_world = transform_qpos(lib["qpos"][term_frame], dy, pv, of)[0]

    cmd = SpeedCommand([(0.0, 0.0, 0.0), (1.0, 1.3, np.deg2rad(20))])

    def traj_fn(t, xy, yaw):
        if t < T - tail:
            return cmd.trajectory(xy, yaw, t)
        rem = max(C.DT, T - t)
        block = []
        for h in C.TRAJ_HORIZONS:
            frac = min(1.0, (h * C.DT) / rem)
            p = (1 - frac) * xy + frac * target_xy
            ang = yaw + frac * ((target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi)
            face = np.array([np.cos(ang), np.sin(ang)])
            block += list(_local(p - xy, yaw)) + list(_local(face, yaw))
        return np.array(block, np.float32)

    out = mm.generate(cmd, seconds=T, start_frame=200, traj_fn=traj_fn)
    # ease the final pose (joints, orientation AND position) exactly onto the terminal state
    k = int(tail * C.FPS)
    for j in range(k):
        w = (j + 1) / k
        idx = len(out) - k + j
        out[idx] = _ease(out[idx], term_world, w)

    def marker(t):
        return [([target_xy[0], target_xy[1], 0.9], 0.12, [0.2, 1, 0.2, 1])]  # green: target
    from motiongraph.render import render_qpos
    render_qpos(out, f"{C.OUT_DIR}/mm_task2_inbetween.mp4", markers_fn=marker)


def _ease(cur, term, w):
    out = blend_qpos(cur, term, w)
    out[0:3] = (1 - w) * cur[0:3] + w * term[0:3]
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("task1", "both"):
        task1()
    if which in ("task2", "both"):
        task2()
