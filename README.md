# motiongraph — Motion Matching & Motion Graphs on the Unitree G1

A compact testbed for two classic data-driven animation algorithms — **motion
matching** and **motion graphs** — driving the Unitree G1 humanoid, using the
**LAFAN1** mocap dataset retargeted to the G1. Each algorithm is exercised on two
tasks and rendered offline (MuJoCo → MP4) so results can be reviewed without a
display.

| Algorithm | Task 1 — speed command | Task 2 — in-betweening |
|---|---|---|
| Motion matching | `outputs/mm_task1_speed.mp4` | `outputs/mm_task2_inbetween.mp4` |
| Motion graph    | `outputs/mg_task1_speed.mp4` | `outputs/mg_task2_inbetween.mp4` |

- **Task 1 (speed command):** the character follows a time-varying velocity command
  (speed + heading), producing walk/turn/run locomotion.
- **Task 2 (in-betweening):** the character follows a command *and* must arrive at a
  specified terminal state (position + heading + pose) at a fixed time. The red
  sphere marks the commanded direction; the green sphere marks the terminal target.

---

## 1. Quickstart

```bash
# 1. Use a MuJoCo-capable Python env (mujoco>=3.3, numpy, scipy, imageio[ffmpeg]).
#    Reference env on this machine:
conda activate deploy_mujoco
export MUJOCO_GL=glfw          # offscreen rendering on an NVIDIA GPU + X display

# 2. Download the G1-retargeted LAFAN1 clips (CSV, 30 FPS) from HuggingFace.
bash scripts/download_data.sh           # locomotion subset (walk/run/sprint)
# bash scripts/download_data.sh all     # every G1 clip (~80 MB)

# 3. Build the unified motion library (qpos + forward-kinematics foot positions).
python -m motiongraph.data

# 4. Run the demos (renders MP4s into outputs/).
python run_motion_matching.py both
python run_motion_graph.py both
```

`requirements.txt` lists pip deps if you prefer a fresh env. Rendering needs a GL
backend; on a headless NVIDIA box use `MUJOCO_GL=egl` (or `glfw` if an X display is
available, as here).

---

## 2. Data

- **Source:** [`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
  (public mirror of the Unitree retargeting set). LAFAN1 mocap retargeted to the G1
  by IK + interaction-mesh optimization (kinematic only — no dynamics).
- **Format:** one CSV per clip, 30 FPS. Each row is 36 floats:
  `root (x, y, z, qx, qy, qz, qw)` followed by 29 joint angles in canonical Unitree
  order (`left_hip_pitch … right_wrist_yaw`).
- **Clips used:** the walk-heavy locomotion subset — all 12 `walk*` clips plus 4
  `run*` and 2 `sprint*` (~130 k frames after trimming).
- **T-pose trim:** every clip begins and ends in a T-pose that blends into the motion
  over ~1.5 s, so `data.py` drops `TRIM = 45` frames from each end.
- **Robot model:** the menagerie `unitree_g1` (`g1_29dof_rev_1_0`) is vendored under
  `assets/unitree_g1/`. Its joint order matches the CSV columns exactly; only the
  root quaternion is reordered (dataset `xyzw` → MuJoCo `wxyz`).

`motiongraph/data.py` concatenates the clips into one library
(`data/motion_lib.npz`), precomputing per-frame heading and **foot world positions
via MuJoCo forward kinematics** (used for motion-matching pose features).

---

## 3. How it works

### Shared machinery
- **`g1_model.py`** — CSV→qpos quaternion reorder; MuJoCo FK for foot positions.
- **`kinematics.py`** — planar **root-motion stitching**. A clip fragment is played
  under a fixed yaw+translation *alignment*; at a jump the alignment is recomputed so
  the new frame coincides with the current world pose, giving a C0-continuous root
  path. Joint angles + root orientation are cross-faded over a few frames to hide
  pose pops.
- **`render.py`** — headless MuJoCo renderer; sets `qpos`, runs kinematics, tracks
  the root with a chase camera, overlays marker spheres, writes MP4 via imageio.
- **`commands.py`** — a `SpeedCommand` (schedule of speed + heading) predicts a
  future trajectory with smooth heading slew; this is the steering input for both
  algorithms.

### Motion matching (`motion_matching.py`)

**Feature database** (`features.py`, 27-D per frame, all in the frame's root-local
space): a *trajectory* part — future root offset `(dx,dy)` and facing `(cosθ,sinθ)`
at horizons {10,20,30} frames (≈0.33/0.67/1.0 s) — and a *pose* part — both feet
positions, both feet velocities, and root velocity. Features are z-scored; the
trajectory and pose groups carry separate weights.

**Task 1 — speed-command runtime.** Every `MM_SEARCH_INTERVAL` frames we build a query
`q = [ trajectory-from-command | pose-of-current-frame ]`. The trajectory half comes
from the `SpeedCommand`: integrating the commanded speed along a heading that slews
toward the commanded heading gives predicted future world points, expressed in the
character's local frame — exactly the trajectory feature layout. The pose half is the
*current* frame's pose features, so a match continues the current motion. We query a
cKDTree (restricted to upright frames) for the nearest database frame and **only jump
to it if it is clearly better than continuing** the current clip (hysteresis: jump iff
`d_best < d_continue·(1−margin)`). This keeps the character on long continuous
fragments. Root motion is stitched continuously (`kinematics.py`) and a jump cross-fades
the pose over `BLEND_FRAMES`.

**Task 2 — in-betweening.** Identical machinery, but the trajectory half of the query is
*overridden* in the final `tail` seconds so the search is pulled onto the terminal. For
each horizon `h`, with `frac = min(1, h·dt / time_remaining)`:

```
future_pos_h   = lerp(current_xy,  target_xy,  frac)      # straight line to the target
future_facing_h= slerp(current_yaw, target_yaw, frac)     # turn toward the terminal facing
```

expressed in the local frame. Because `frac` hits 1 exactly when `time_remaining`
elapses, the nearest-neighbour search naturally selects frames that decelerate and
arrive *on time* — the character walks onto the target instead of being teleported.
Finally the last `tail·FPS` frames are eased onto the terminal state with
`ease_to_terminal` (joint + root-orientation blend, root-position lerp), so the terminal
position, heading **and** pose are met exactly. The terminal pose itself is any chosen
library frame placed at the target via the same planar alignment used for stitching.

### Motion graph (`motion_graph.py`)

**Build.** A per-frame descriptor (29 joint angles, joint velocities, root height,
root planar velocity, yaw rate) is z-scored and **PCA-reduced to 16-D** — without this,
KDTree queries on the raw 62-D descriptor degrade to near-brute-force (the original
build took 174 s; 16-D takes ~4 s). Frames whose descriptors are within an adaptive
radius become directed **transition edges** `i→j` (a good blend point); every frame
also has its natural successor `i→i+1`. The graph (`{i: [(j, dist), …]}`) is cached to
`data/motion_graph.pkl`.

**Task 1 — command following (greedy).** The runtime state is `(frame i, world
alignment A)`, where `A` is a planar yaw+translation mapping clip space to the world.
Playing advances `i→i+1` under a fixed `A`, and the clip's own root motion carries the
character. A *decision* is taken every `SEARCH_INTERVAL` frames (or at a clip end) over
the candidate set `{continue: i+1} ∪ {transition targets j}`. Each candidate `f` is
scored by how its **intrinsic** motion matches the command:

```
v_local(f) = avg planar velocity over the next H frames of f, in f's OWN local frame
w_local    = R(−yaw_world) · command.desired_velocity(t)      # desired vel in robot-local frame
cost(f)    = ‖ v_local(f) − w_local ‖  +  penalty(f)          # 0 for continue, small for a switch
```

and we take the arg-min. Scoring in the *local* frame is the crux: `v_local` is the
clip fragment's intrinsic turn/speed, independent of where it is placed in the world, so
comparing it to the desired direction-relative-to-current-facing lets the graph actually
**steer** (an earlier version aligned every candidate to the current heading first, which
made them all look "forward" and prevented steering). On a transition, `A` is recomputed
so frame `j` lands on the current world pose (C0-continuous) and the pose is cross-faded;
the switch penalty biases toward long continuous fragments.

**Task 2 — in-betweening (beam search).** We plan an edge sequence that arrives at the
terminal at a fixed time. A search node is `(frame, A, world_xy, world_yaw, t, cost)`;
each macro-step plays `K` frames along an edge. Step cost is
`cmd_w·‖avg_world_vel − desired‖ + transition_penalty`, where `desired` follows the
speed command while cruising and switches to **reach-target pacing** in the final `tail`
seconds (toward the target at `remaining_distance / remaining_time`, capped at
`max_speed`) so the path genuinely arrives. After each round we keep the best `beam`
nodes by `cost + admissible_distance_heuristic` (`max(0, dist_to_target −
max_speed·time_left)`). When `t ≥ N`, the goal cost adds
`w_pos·‖xy−target‖ + w_yaw·|Δyaw| + w_pose·pose_dist(frame, terminal)`; the best leaf is
back-tracked, replayed with blends, and eased onto the terminal.

### Motion quality (`cleanup.py`, `footlock.py`, `tools/diagnose.py`)

Kinematic stitching introduces two artifacts, both measured by `tools/diagnose.py`
(time-axis plots of root speed and per-foot sole-contact slip, saved as
`outputs/diag_*.png`):

- **Root jitter** — the per-frame pose cross-fade keeps the root *position* continuous
  but not its *velocity*, so each jump is a velocity step. A short Savitzky-Golay filter
  on the root translation removes the resulting high-frequency sawtooth (jerk RMS down
  ~90 %).
- **Foot skating** — a jump can move a planted foot, and the retargeted data itself
  slides ~0.2 m/s while planted. `footlock.py` detects steady stance (a sole sphere
  grounded *and* not moving vertically — so heel-strike/toe-off are not mistaken for
  slip) and pins the two lowest **sole contact spheres** (not the ankle, which
  legitimately translates as the foot rolls) at a fixed world point, solving damped
  least-squares IK on that leg's 6 joints. Root and the rest of the body are untouched.

`cleanup()` runs *smooth → foot-lock*; for in-betweening the order is *smooth →
ease-to-terminal → foot-lock* (lock last, so the eased tail's feet are also clean).
Because foot-lock never moves the root, the exact terminal arrival is preserved.

| demo | root-jump [m] | jitter (jerk RMS) | sole slip [m/s] |
|---|---|---|---|
| raw LAFAN1 walk (reference) | — | — | 0.21 |
| mm_task1 | 0.10 | 147 (was 1263) | 0.12 |
| mm_task2 | 0.06 | 123 (was 686) | 0.23 |
| mg_task1 | 0.05 | 29 (was 178) | 0.04 |
| mg_task2 | 0.05 (was 4.06) | 79 (was 17470) | 0.26 |

---

## 4. Repository layout

```
assets/unitree_g1/      vendored MuJoCo G1 model + floor scene
scripts/download_data.sh download G1 LAFAN1 CSVs from HuggingFace
motiongraph/
  config.py             paths, skeleton layout, feature/search settings
  g1_model.py           qpos conversion + MuJoCo forward kinematics
  data.py               build/load the unified motion library
  kinematics.py         planar root-motion stitching + pose blending
  features.py           motion-matching feature vectors
  commands.py           speed command -> predicted trajectory
  render.py             offline MuJoCo -> MP4 renderer
  motion_matching.py    feature DB + nearest-neighbour controller
  motion_graph.py       transition graph + greedy follower + beam planner
  footlock.py           foot-lock IK (sole-sphere pin via damped least squares)
  cleanup.py            post-process: root de-jitter -> foot-lock
tools/diagnose.py       quality metrics + time-axis plots (root speed, foot slip)
run_motion_matching.py  MM demos (task1 / task2)
run_motion_graph.py     MG demos (task1 / task2)
outputs/                rendered MP4s + diag_*.png (git-ignored)
```

---

## 5. Notes & limitations

- Retargeting was **kinematic only**; playback here is likewise kinematic (set
  `qpos`, no physics), which is the standard way to review retargeted motion. The
  robot is not guaranteed to be dynamically feasible. Residual foot slip (~0.2 m/s)
  is inherent to the retargeted data; `footlock.py` brings the demos to that floor.
- Run `python -m tools.diagnose` to regenerate the quality metrics and
  `outputs/diag_*.png` plots for any change you make.
- The library here is the locomotion subset (walk/run/sprint). Add clips with
  `download_data.sh all` and rebuild for richer behavior.
- The motion graph is more constrained than motion matching (it can only switch at
  precomputed transition points), so it follows gentler commands more faithfully
  than very sharp ones — expected for the algorithm.
- Search targets are restricted to upright frames (`z ≥ 0.6`) so jumps never land in
  crouch/get-up poses present in the raw data.

## 6. Build log (steps taken)

1. Probed the environment; chose `deploy_mujoco` (mujoco 3.3.3) + `MUJOCO_GL=glfw`.
2. Located the public HF mirror of the G1 LAFAN1 retargeting set; confirmed the
   36-column / 30 FPS CSV layout and joint order against the menagerie G1.
3. Verified headless rendering and that the menagerie G1 joint order matches the CSV.
4. Built the data pipeline (download → library with FK foot positions) and renderer.
5. Implemented motion matching (features, command, NN controller) + both tasks.
6. Implemented the motion graph (descriptor, PCA+KDTree transitions, greedy + beam)
   + both tasks. Cached the graph; reduced descriptor dimensionality after finding
   62-D KDTree queries were near-brute-force.
7. Rendered all four demos and wrote this README.
8. **Quality pass** (driven by `tools/diagnose.py` measurements + visual review):
   trimmed T-pose frames and expanded to all walk clips; fixed a 4 m root teleport in
   MG in-betweening (planner now uses reach-target pacing + a `blend_qpos` argument bug
   that snapped eased frames to the target); cut root jitter ~90 % with Savitzky-Golay
   smoothing + MM jump hysteresis; and added foot-lock IK to remove foot skating down
   to the raw-data floor.
