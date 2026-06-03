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
1. **Feature database** (`features.py`, 27-D per frame, all root-local):
   future root offsets + facing at 3 horizons (≈0.33/0.67/1.0 s) and pose features
   (foot positions, foot velocities, root velocity).
2. **Runtime:** every ~0.33 s build a query =
   `[ trajectory-from-command | pose-of-current-frame ]`, find the nearest database
   frame (cKDTree, restricted to upright frames), jump there, and continue. Root
   motion is stitched; jumps are cross-faded.
3. **In-betweening:** near the deadline the far trajectory samples are steered toward
   the terminal, and the final ~0.7 s of pose is eased exactly onto the terminal
   frame placed at the target — so it arrives precisely.

### Motion graph (`motion_graph.py`)
1. **Build:** a per-frame pose+velocity descriptor (joints, joint velocities, root
   height/velocity) is PCA-reduced to 16-D (so high-dim KDTree queries stay fast).
   Frames whose descriptors are close become **transition edges**; every frame also
   has a natural successor edge. The graph is cached to `data/motion_graph.pkl`.
2. **Task 1 — greedy:** at each decision point, among `{continue} ∪ {transitions}`
   pick the edge whose intrinsic local velocity best matches the commanded velocity
   (compared in the robot's local frame, so the choice actually steers).
3. **Task 2 — beam search:** plan an edge sequence (beam width 64, 10-frame macro
   steps) that loosely tracks the command and minimizes terminal
   position/heading/pose error at the deadline; the winning path is replayed with
   blends and eased onto the terminal.

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
run_motion_matching.py  MM demos (task1 / task2)
run_motion_graph.py     MG demos (task1 / task2)
outputs/                rendered MP4s (git-ignored)
```

---

## 5. Notes & limitations

- Retargeting was **kinematic only**; playback here is likewise kinematic (set
  `qpos`, no physics), which is the standard way to review retargeted motion. There
  is some foot sliding and the robot is not guaranteed to be dynamically feasible.
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
