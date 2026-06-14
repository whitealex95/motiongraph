"""Multimodal locomotion demo for motion matching: the GenoView clips we have retargeted
(walk1_subject5 + run1_subject5) + the jump skill, all in one feature database.

A single speed command ramps walk -> run -> walk; motion matching picks walk frames at low
speed and run frames at high speed (the trajectory feature encodes the commanded pace), then
a trigger jumps. This is the classic motion-matching payoff: gait chosen by the data, not a
state machine. (pushAndStumble, the 3rd GenoView clip, has no public G1 retarget, so it is
omitted -- see config.LOCO_JUMP_CLIPS.)

Usage: conda activate deploy_mujoco; MUJOCO_GL=glfw python run_locomotion.py
"""
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.commands import SpeedCommand
from motiongraph.motion_matching import MotionMatcher
from motiongraph.cleanup import cleanup
from motiongraph.render import render_qpos, trace_labels

START = 1500           # a clean walk frame in walk1_subject5
SECONDS = 14.0
JUMP_AT = 10.5         # s; trigger a jump from a run-up after the run stretch
# (start_s, speed[m/s], heading[rad]): stroll -> run -> ease back to a walk
CMD = SpeedCommand([(0.0, 1.0, 0.0), (3.0, 3.2, 0.0), (8.0, 1.1, 0.0)])
CAM = dict(cam_dist=4.5, cam_elev=-12, cam_azim=130, width=900, height=620)  # root-tracking


def main():
    lib = load_library(C.LOCO_LIB_PATH)               # walk + run + jump (auto-builds once)
    mm = MotionMatcher(lib, traj_w=2.0, pose_w=1.0)   # weight the trajectory so speed steers gait
    out, tframe = mm.generate(CMD, SECONDS, start_frame=START, jump_at=JUMP_AT, return_trace=True)
    out = cleanup(out)
    render_qpos(out, f"{C.OUT_DIR}/loco_mm_walk_run_jump.mp4",
                trace=trace_labels(tframe, lib), **CAM)
    # report the gait timeline
    names, clip = lib["clip_names"], lib["clip_id"][tframe]
    print("active clip over time:")
    for s in range(0, len(tframe), 30):
        print(f"  t={s / C.FPS:4.1f}s  {names[clip[s]]}")


if __name__ == "__main__":
    main()
