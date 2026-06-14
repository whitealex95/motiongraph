# motiongraph — GenoView Motion Matching on the Unitree G1 (`mm-only`)

A faithful port of [GenoView](https://github.com/orangeduck/GenoViewPython)'s ("Simple Motion
Matching", Daniel Holden) controller, driving the Unitree **G1** over GMR-retargeted LAFAN1
**walk / run / push-and-stumble** clips plus a phase-segmented **jump** skill. Reactive
nearest-neighbour search — no neural network, no planner, no authored blend trees.

> This is the **motion-matching-only** branch. The motion graph + A\* planner (and the
> MM-vs-MG experiments) live on `master`.

Live demo / writeup: the GitHub Pages site in `docs/`.

## Quickstart

```bash
# MuJoCo-capable Python env (mujoco>=3.3, numpy, scipy, imageio[ffmpeg], matplotlib).
# Reference env on this machine:
conda activate deploy_mujoco            # MUJOCO_GL=glfw, DISPLAY=:0 for offscreen render

python run_locomotion.py                # speed-driven walk -> run -> jump (the headline demo)
python run_motion_matching.py           # command-following locomotion
python run_experiments.py               # square-path experiments (+ reactive box jump)
python run_exact_box.py                 # exact box-jump: fixed single box + multi-box course
python run_jump.py                      # jump-on-command / jump-at-a-searched-location / raw
```

The first run builds the cached feature library `data/motion_lib.npz` (auto). The G1 clips
live as GMR `.pkl` under `data/g1_gmr_lafan1/` and the CAMDM jump CSVs under `data/g1_jump/`.

## How it works

Each frame `step(speed, heading)` does four things (`mm_g1`-style, in `motion_matching.py`):

1. **Desired trajectory** — the command's `(speed, heading)` is slewed by critically-damped
   springs into a predicted future path at +10/+20/+30 frames (`springs.py`).
2. **Search** — every 0.15 s, query `[current pose | desired trajectory]` and take the nearest
   neighbour over **per-clip KD-trees** (each clip's last 30 frames trimmed from the search so a
   full future always exists), biased toward staying in the current clip. Jump frames are
   excluded — a jump happens only on a trigger.
3. **Inertialize** — the pose discontinuity (joints + pelvis-local pos/rot **and their
   velocities**) is captured as a decaying-spring offset and bled to zero over ~0.075 s. No
   cross-fade; momentum is preserved.
4. **Integrate the root** — a per-clip Savitzky–Golay-smoothed "simulation root" (ground
   position + facing) carries the character; the pelvis is placed back on it as a local offset.

The **27-D feature** (`mm_features.py`, in the sim-root frame, per-block normalized): foot
positions (6) + foot & pelvis velocities (9) + future sim-root offset (6) and facing (6). The
database is **L/R mirrored** (each clip added twice). Commanding a higher speed pulls the match
into the **run** clip; a lower speed back into **walk**.

## Data

GMR-retargeted ([GMR](https://github.com/YanjieZe/GMR)) LAFAN1 `subject5` clips — walk, run,
pushAndStumble — to a 29-DOF G1 whose joint order matches our menagerie model (so the `.pkl`
maps straight to qpos). Each clip uses GenoView's exact per-clip window (`config.CLIP_TRIM`) and
is mirrored. The jump skill is the CAMDM `walk→jump→walk` clips, phase-labeled (`ready / takeoff
/ flight / touchdown / after`); a jump is entered only from its `ready` run-up.

## Repository layout

```
motiongraph/
  config.py          paths, joint layout, GenoView MM settings
  g1_model.py        CSV/PKL→qpos, FK feet, sagittal mirror_qpos
  data.py            build/load the L/R-mirrored motion library (LIB_PATH)
  mm_features.py     GenoView feature DB (sim-root, per-block norm)
  quat.py / springs.py   quaternion + spring/inertialization helpers
  motion_matching.py GenoView MotionMatcher: step(speed,heading) + generate()
  jumps.py           pre-take-off run-up index for the jump trigger
  commands.py        speed/heading command schedule
  cleanup.py / footlock.py   root de-jitter + foot-lock IK
  render.py          offline MuJoCo -> MP4 (+ GenoView command-trajectory gizmo)
run_locomotion.py / run_motion_matching.py / run_experiments.py / run_exact_box.py / run_jump.py
tools/diagnose.py    quality metrics (teleport / jitter / skating)
tools/make_web.py    compress demo videos -> docs/videos, refresh the site
```

## Credits

- **G1 model** — [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  `unitree_g1`.
- **Motion data** — [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
  (Ubisoft La Forge), retargeted with [GMR](https://github.com/YanjieZe/GMR); jump clips from
  CAMDM.
- **Approach** — GenoView's real-time motion-matching demo; a sibling real-time keyboard viewer
  is `~/Projects/motionmatching-g1`.
