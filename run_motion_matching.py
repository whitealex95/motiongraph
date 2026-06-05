"""Motion matching demos on the G1.

  task1: locomotion driven by a time-varying speed command.
  task2: motion in-betweening -- follow a command, then arrive at a terminal state
         at a fixed time (the far trajectory samples are steered to the target and
         the final pose is eased onto the terminal frame).

The gen_* functions return (qpos_sequence, markers_fn) so tools/diagnose.py can
analyse the exact same motion that gets rendered.

Usage: python run_motion_matching.py [task1|task2|both]
"""
import sys
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.commands import SpeedCommand, demo_speed_schedule
from motiongraph.motion_matching import MotionMatcher
from motiongraph.kinematics import transform_qpos, alignment_to, ease_to_terminal
from motiongraph.cleanup import cleanup
from motiongraph.render import trace_labels
from motiongraph.features import _local

START = 2640   # steady forward walk, arms down (walk1_subject5)


def _cmd_marker(out, command):
    def fn(t):                       # t is the frame index (see render.py)
        i = min(t, len(out) - 1)
        spd, hd = command.state(i * C.DT)
        tip = out[i, 0:2] + max(0.4, spd) * 0.5 * np.array([np.cos(hd), np.sin(hd)])
        return [([tip[0], tip[1], 1.2], 0.06, [1, 0.2, 0.2, 1])]   # red: command dir
    return fn


def gen_task1(mm=None, clean=True):
    mm = mm or MotionMatcher(load_library(), traj_w=1.5, pose_w=1.0)
    cmd = demo_speed_schedule()
    out, tframe = mm.generate(cmd, seconds=15.0, start_frame=START, return_trace=True)
    trace = trace_labels(tframe, mm.lib)
    if clean:
        out = cleanup(out)                                # de-jitter root + foot-lock
    return out, _cmd_marker(out, cmd), trace


def gen_task2(mm=None, clean=True):
    lib = load_library()
    mm = mm or MotionMatcher(lib, traj_w=1.5, pose_w=1.0)
    T, tail = 9.0, 2.5
    target_xy = np.array([3.5, 1.5])
    target_yaw = np.deg2rad(90)
    term_frame = 1200                                 # a mid-clip walk pose for the terminal
    dy, pv, of = alignment_to(lib["qpos"][term_frame, 0:2], lib["yaw"][term_frame], target_xy, target_yaw)
    term_world = transform_qpos(lib["qpos"][term_frame], dy, pv, of)[0]
    cmd = SpeedCommand([(0.0, 0.0, 0.0), (1.0, 1.1, np.deg2rad(20))])

    def traj_fn(t, xy, yaw):
        if t < T - tail:
            return cmd.trajectory(xy, yaw, t)
        rem = max(C.DT, T - t)                         # steer far samples onto the target
        block = []
        for h in C.TRAJ_HORIZONS:
            frac = min(1.0, (h * C.DT) / rem)
            p = (1 - frac) * xy + frac * target_xy
            ang = yaw + frac * ((target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi)
            face = np.array([np.cos(ang), np.sin(ang)])
            block += list(_local(p - xy, yaw)) + list(_local(face, yaw))
        return np.array(block, np.float32)

    out = mm.generate(cmd, seconds=T, start_frame=START, traj_fn=traj_fn)
    if clean:
        out = cleanup(out, lock=False)                 # smooth root (lock comes after the ease)
    out = ease_to_terminal(out, term_world, int(tail * C.FPS))   # arrive exactly
    if clean:
        from motiongraph.footlock import footlock      # lock feet of the eased tail too
        out = footlock(out)

    def marker(t):
        return [([target_xy[0], target_xy[1], 0.9], 0.12, [0.2, 1, 0.2, 1])]  # green: target
    return out, marker, None                              # in-betweening: no per-frame HUD


def _render(out, markers_fn, trace, name):
    from motiongraph.render import render_qpos
    render_qpos(out, f"{C.OUT_DIR}/{name}.mp4", markers_fn=markers_fn, trace=trace)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("task1", "both"):
        _render(*gen_task1(), "mm_task1_speed")
    if which in ("task2", "both"):
        _render(*gen_task2(), "mm_task2_inbetween")
