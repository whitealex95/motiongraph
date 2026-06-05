"""Motion graph demos on the G1.

  task1: greedy command following -- walk the graph, at each decision point taking
         the edge whose motion best matches the commanded velocity.
  task2: in-betweening by A* search -- plan a least-cost edge sequence that arrives
         at a terminal state at a fixed time.

The gen_* functions return (qpos_sequence, markers_fn) for tools/diagnose.py.

Usage: python run_motion_graph.py [task1|task2|both]
"""
import sys
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.commands import SpeedCommand
from motiongraph.motion_graph import MotionGraph
from motiongraph.kinematics import transform_qpos, alignment_to, ease_to_terminal
from motiongraph.cleanup import cleanup
from motiongraph.render import trace_labels

START = 2640   # steady forward walk, arms down (walk1_subject5)


def _cmd_marker(out, command):
    def fn(t):
        i = min(t, len(out) - 1)
        spd, hd = command.state(i * C.DT)
        tip = out[i, 0:2] + max(0.4, spd) * 0.5 * np.array([np.cos(hd), np.sin(hd)])
        return [([tip[0], tip[1], 1.2], 0.06, [1, 0.2, 0.2, 1])]
    return fn


def _gentle_schedule():
    d = np.deg2rad
    return SpeedCommand([
        (0.0, 1.0, d(0)),    # walk +x
        (3.0, 1.2, d(35)),   # gentle veer left
        (6.0, 1.2, d(35)),   # keep walking
        (9.0, 1.0, d(-15)),  # veer right
        (12.0, 0.9, d(-15)),
    ])


def gen_task1(g=None, clean=True):
    g = g or MotionGraph(load_library())
    cmd = _gentle_schedule()
    out, tframe, _ = g.follow_command(cmd, seconds=15.0, start_frame=START, return_trace=True)
    trace = trace_labels(tframe, g.lib)
    if clean:
        out = cleanup(out)                                # de-jitter root + foot-lock
    return out, _cmd_marker(out, cmd), trace


def gen_task2(g=None, clean=True):
    g = g or MotionGraph(load_library())
    cmd = SpeedCommand([(0.0, 1.0, np.deg2rad(15))])   # cruise; planner steers onto target in the tail
    target_xy = np.array([5.5, 1.5])                   # reachable at walking speed over 8 s
    target_yaw = np.deg2rad(60)                        # arrive facing a different way (true in-between)
    term_frame = 2600
    out = g.plan_to(cmd, seconds=8.0, start_frame=START,
                    target_xy=target_xy, target_yaw=target_yaw, term_frame=term_frame)
    if clean:
        out = cleanup(out, lock=False)                 # smooth root (lock comes after the ease)
        dy, pv, of = alignment_to(g.qpos[term_frame, 0:2], g.yaw[term_frame], target_xy, target_yaw)
        out = ease_to_terminal(out, transform_qpos(g.qpos[term_frame], dy, pv, of)[0], int(0.7 * C.FPS))
        from motiongraph.footlock import footlock      # lock feet of the eased tail too
        out = footlock(out)

    def marker(t):
        return [([target_xy[0], target_xy[1], 0.9], 0.12, [0.2, 1, 0.2, 1])]
    return out, marker, None                              # in-betweening: no per-frame HUD


def _render(out, markers_fn, trace, name):
    from motiongraph.render import render_qpos
    render_qpos(out, f"{C.OUT_DIR}/{name}.mp4", markers_fn=markers_fn, trace=trace)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("task1", "both"):
        _render(*gen_task1(), "mg_task1_speed")
    if which in ("task2", "both"):
        _render(*gen_task2(), "mg_task2_inbetween")
