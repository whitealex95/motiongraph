# Paper notes — Motion Matching & Motion Graphs with a Jump Skill on the Unitree G1

A single, living document collecting every technical detail of this project for paper
writing. (Usage/quickstart lives in `README.md`; this file is the methods reference.)
Sections are written so they can be lifted into Method / Implementation / Experiments.

---

## 1. Overview & contributions

We build a compact testbed that drives the **Unitree G1** humanoid with two classic
data-driven controllers — **motion matching (MM)** and **motion graphs (MG)** — on
**LAFAN1** mocap retargeted to the G1, and extend MG with a **discrete skill (jump)** that
is *phase-segmented* and *world-anchored*.

Contributions / things demonstrated:
1. A shared kinematic backbone (root-motion stitching, FK features, foot-lock IK, offline
   rendering) under which MM and MG are directly comparable.
2. MM and MG locomotion driven by **speed commands** and **in-betweening** to a terminal.
3. A **jump skill**: per-frame 5-phase labels (ready/take-off/flight/touch-down/after),
   with the hard guarantee that a jump is **entered only in `ready`** and **exited only
   after `after`**.
4. **World-anchored jumps**: a jump's apex is pinned to a fixed obstacle (a box). The walk
   into/out of the jump is an **in-between**, computed either reactively (greedy/NN) or by
   **beam-search planning** (`plan_to`) when precision is required.
5. A composite task that satisfies a hard constraint: **jump over one box, loop, and jump
   over the same box again**, with both apexes within ~0.25 m of the box.

---

## 2. Data

- **Source.** LAFAN1 (Ubisoft) retargeted to Unitree robots; we use the public mirror
  `lvhaidong/LAFAN1_Retargeting_Dataset` (G1, 30 FPS). Retargeting was IK +
  interaction-mesh, **kinematic only** (no dynamics), so playback is likewise kinematic.
- **Locomotion library.** A *single* clip **`walk1_subject5`** (~258 s, 7750 frames after
  trimming) — the walk motion CAMDM uses as its main `walk`, with a natural **arms-down**
  posture (`walk1_subject2`, used earlier, walks with the hands raised). One clip →
  **unimodal** distribution (one subject, one gait) so matching/graph never hop styles.
- **T-pose trimming (applied).** Every LAFAN1 clip starts/ends in a T-pose (arms out) that
  blends in over ~1.5 s; `data.py:_load_clip` drops the first/last `TRIM = 45` frames of
  every clip, so the T-pose never enters the library or any generated motion.
- **Jump library.** `walk1_subject5` + three G1-retargeted `walk_jump_walk*` clips (from
  `~/Projects/CAMDM`): short, nearly straight `walk → jump → walk` sequences. Concatenated
  into `data/motion_lib_jump.npz` (8290 frames).
- **Format.** Per frame, CSV row = 36 floats: root `(x,y,z, qx,qy,qz,qw)` (quat **xyzw**)
  + 29 joint angles in canonical Unitree order. MuJoCo free-joint qpos uses quat **wxyz**,
  so only the root quaternion is reordered. The menagerie `unitree_g1` model's joint order
  matches the CSV columns exactly.
- **Library arrays.** `qpos (N,36)`, `feet_world (N,2,3)` (FK), `yaw (N)`, `clip_id`,
  `frame_in_clip`, `lengths`, `clip_names`; for the jump library also `skill (N)`,
  `phase (N)`, and per-jump `jump_entry/jump_takeoff/jump_apex/jump_land/jump_continues/
  jump_box`.

---

## 3. Representation & kinematic backbone

- **State.** A frame is the 36-D `qpos`. Forward kinematics (MuJoCo `mj_kinematics`) gives
  foot/body world positions for features and rendering.
- **Root-motion stitching** (`kinematics.py`). A clip fragment is played under a fixed
  planar **alignment** `A = (Δyaw, pivot, offset)`: `p' = R(Δyaw)(p − pivot) + offset`,
  `q' = R_z(Δyaw) ⊗ q`. At a jump/transition, `A` is recomputed so the new frame coincides
  with the current world pose → **C0-continuous** root path. Joint angles + root
  orientation are cross-faded over `BLEND_FRAMES = 12`.
- **Post-process** (`cleanup.py`): Savitzky-Golay smoothing of root **xy** (window 9; z is
  left untouched so jump peaks survive) → **foot-lock IK** (§7).

---

## 4. Motion matching (`motion_matching.py`)

- **Feature vector** (`features.py`, 27-D, all root-local):
  - *trajectory*: future root offset `(dx,dy)` and facing `(cosθ,sinθ)` at horizons
    `{10,20,30}` frames (≈0.33/0.67/1.0 s) — 12-D.
  - *pose*: both feet positions, both feet velocities, root velocity — 15-D.
  - z-scored; trajectory and pose groups carry separate weights `traj_w, pose_w`.
- **Criterion.** Query `q = [trajectory-from-command | pose-of-current-frame]`. Match =
  nearest neighbour in the **weighted standardized feature space**: `d = ‖q_std −
  F_std[j]‖`, found with a cKDTree restricted to upright **walk** frames.
- **Hysteresis.** Jump to the NN only if clearly better than continuing the current clip:
  `d_best < d_continue · (1 − jump_margin)`, `jump_margin = 0.35`, search every
  `MM_SEARCH_INTERVAL = 15` frames. Keeps long continuous fragments → less jitter/skating.
- **In-betweening (terminal).** In the final `tail` s, the trajectory half of the query is
  overridden to steer onto the target: for horizon `h`,
  `future_pos = lerp(cur_xy, target_xy, frac)`, `frac = min(1, h·dt/time_left)`; the last
  `tail·FPS` frames are eased exactly onto the terminal pose.

MM is **reactive** (one NN per interval, no multi-step look-ahead). It has no beam planner.

---

## 5. Motion graph (`motion_graph.py`)

- **Transition descriptor** (per frame): `[29 joint angles, joint velocities, root height,
  root planar velocity, yaw rate]` (62-D), z-scored, **PCA-reduced to 16-D** (raw 62-D
  KDTree queries are near-brute-force; 16-D build ≈ 4 s vs 174 s).
- **Edges.** For each (subsampled) source frame, the `n_neighbors` nearest descriptor
  neighbours within an adaptive radius `τ = median(NN1)·2.5` become directed transition
  edges (a good blend point); every frame also has its successor edge. Normal transition
  **targets exclude jump frames** (`skill==1`) so locomotion never produces a jump.
  Cached per library size. Default `n_neighbors=16`; the loop/same-box demos use
  `n_neighbors=28, tgt_stride=1` (denser ⇒ tighter turns).
- **Greedy following** (`follow_command`/`follow_route`). State = (frame `i`, alignment
  `A`). Every `SEARCH_INTERVAL` frames choose over `{continue} ∪ {transitions}`:
  `cost(f) = ‖v_local(f) − w_local‖ + penalty(f)`, where `v_local(f)` is `f`'s **intrinsic**
  average planar velocity (in its own local frame) and `w_local = R(−yaw)·want`. Scoring in
  the *local* frame is what lets the graph steer (an early version aligned every candidate
  to the current heading first, which made them all look "forward"). Reactive, 1 candidate.
- **Beam-search planning** (`plan_to`) — the only multi-step optimizer. Finds an edge
  sequence that reaches a terminal `(xy, yaw, pose)` at a fixed time:
  - node = `(frame, A, xy, yaw, t, cost, parent)`; a macro-step plays `K` frames along
    `{continue} ∪ {transitions}`; step cost = `cmd_w·‖avg_vel − desired‖ + transition_pen`.
  - After each round keep the best `beam = 64` nodes by `cost + admissible_distance_to_go`.
  - At `t ≥ N`: goal cost `= w_pos·‖xy−target‖ + w_yaw·|Δyaw| + w_pose·pose_dist(frame,
    terminal)`. Backtrack best leaf, replay with blends, ease onto terminal.
  - `desired` follows the speed command while cruising, then **reach-target pacing**
    (toward the target at `remaining_dist/remaining_time`) in the final `tail` s.

**MM vs MG.** Both walk reactively (NN vs greedy edge). Only MG has the **beam planner**;
MM does not. The fixed-location "search over trigger time" used in the jump tasks is a
separate 1-D grid search (same wrapper for MM and MG), not beam search.

---

## 6. Jump skill — phases & confinement

- **Phase labels** (`data.py:_label_jump`, stored as `phase (N)`; codes in
  `config.JUMP_PHASES`). Flight = both feet above `foot_thr=0.13`. Around each flight
  (`takeoff`=first airborne, `land`=last airborne):

  | phase | code | frames | length (cfg) | meaning |
  |---|---|---|---|---|
  | ready | 1 | `[takeoff−22, takeoff−10)` | `PHASE_READY=12` | run-up; **enter here** |
  | takeoff | 2 | `[takeoff−10, takeoff)` | `PHASE_TAKEOFF=10` | push-off / loading |
  | flight | 3 | `[takeoff, land]` | — | airborne |
  | touchdown | 4 | `(land, land+6]` | `PHASE_TOUCHDOWN=6` | landing impact |
  | after | 5 | `(land+6, land+24]` | `PHASE_AFTER=18` | absorption / recovery; **exit here** |

  `apex` = arg-max pelvis z over flight. `skill=1` over the whole `ready..after` span.

- **Entry confined to `ready`.** `jumps.py:jump_entries` returns only frames with
  `phase==ready` (and `continues==True`), mapping each to its `land`/`apex`. Both MM and MG
  call `best_jump_entry(cur)` = the ready frame whose features/descriptor best match the
  current frame (smooth take-off). **A jump can never be cut into mid-air.**

- **Exit confined to `after`.** A triggered jump is played as a **locked segment**
  `entry → after_end = land+1+PHASE_TOUCHDOWN+PHASE_AFTER` (no decisions inside), then
  locomotion resumes. Because normal transition targets exclude all `skill==1` frames,
  resumption necessarily lands on a **walk** frame past `after`. Hence exit ⊂ post-`after`.

- **Why two mechanisms.** Entry confinement = restricting the candidate set
  (`best_jump_entry`). Exit confinement = (a) locked playback through `after` + (b)
  skill-based exclusion of jump frames from normal targets.

---

## 7. World-anchored box jumps & the hard-constraint task

- **Heuristic box** (`data.py:_heuristic_box`, per-jump `jump_box` half-extents). Centred
  under the apex; forward half `hx=0.13`, lateral half `hy=0.28`; height = foot clearance
  over its forward footprint × 0.92, clamped to `[0.13, 0.24]` half. The box is one the
  jump provably clears (feet over the footprint stay above the top). The data jumps are low
  hops, so boxes are low (~0.12–0.18 m tall).

- **Anchoring.** With the jump nearly straight, apex ≈ entry + `fwd` (entry→apex forward,
  ≈1.55 m). To land the apex on a box at `(bx,by)`, the character must reach the **entry
  pose** at `(bx − fwd, by)` facing the jump direction; the pinned clip then carries
  take-off/flight/landing.

- **Jump on command / at fixed location** (`run_jump.py`, MM+MG). `task1`: walk 1 m/s,
  enter a jump at a trigger. `task2`: a box is predefined at `(x,y)`; the trigger time is
  chosen by a search minimizing the apex's 2-D distance to the box (≈0.1–0.2 m).

- **HARD: same box twice with a loop** (`gen_loop_same_box`, MG). Reactive steering cannot
  guarantee the precise return after a loop (single-clip greedy drifts ~0.85 m laterally,
  invariant to waypoint/steer tuning). So **both jump approaches are beam-planned
  in-betweens**: `plan_to` navigates precisely to the `ready` entry pose at `(bx−fwd, by)`
  facing +x; the loop between is greedy. Result: one box, both apexes within ~0.25 m
  `(4.97, 0.24)` and `(4.99, 0.20)` of `(5,0)`. Segments are stitched in world coordinates
  (`follow_route(init_align=…, return_state=…)`, `plan_to(start_frame=…)` + a placement
  transform).

This is the clean statement for a paper: *reactive control reaches the box approximately;
graph planning (beam) reaches the world-anchored keyframe exactly, so the same obstacle can
be re-used.*

---

## 8. Motion quality

Measured by `tools/diagnose.py` (time-axis plots of root speed + per-foot sole-contact
slip). Two artifacts and their fixes:

- **Root jitter.** The pose cross-fade keeps root *position* continuous but not its
  *velocity* (a step at each jump). A short Savitzky-Golay filter on root xy removes the
  high-frequency sawtooth (jerk-RMS down ~90 %).
- **Foot skating.** Measured at the **sole contact spheres** (not the ankle, which
  legitimately translates as the foot rolls) and gated on vertical velocity (so heel-strike
  isn't miscounted). `footlock.py` pins the two lowest sole spheres during steady stance
  via damped-least-squares IK on the 6 leg joints (root untouched, so terminal arrivals are
  preserved). Brings demos to/below the raw-data slip floor (~0.21 m/s).

Single-clip metrics (after the quality pass): jitter 30–150 (vs 1300–17000 before), root
jumps < 0.1 m (vs a 4.06 m teleport before), sole slip 0.04–0.23 m/s.

---

## 9. Visualization (`tools/visualize_graph.py`)

Self-contained interactive HTML (Plotly): a 3-D animated G1 skeleton (fixed-scale box) +
the full motion graph in 2-D pose space (PCA of the transition descriptor), nodes coloured
by skill, jump-entry (`ready`) frames marked, `jump→walk` edges highlighted, and the
traversal animated **in sync** with playback (passed edges bold, upcoming edges dotted,
current node a marker). Rendered MP4s carry a HUD (clip name + frame index) and flash a
border + banner on each non-consecutive (transition) frame.

---

## 10. Limitations

- Kinematic playback (no dynamics); residual foot slip is inherent to the retargeted data.
- Single walk clip ⇒ limited maneuverability: turn radius ~5 m, so a "square" loop is in
  practice a rounded loop; reactive return drifts (hence beam planning for precision).
- Jump heights are low (data are small hops) ⇒ low boxes.
- MM currently has no beam planner; the same-box guarantee is shown with MG only (a
  trajectory-optimizing beam over the MM feature DB would give MM the same guarantee).

---

## 11. Hyperparameters (defaults)

| name | value | where |
|---|---|---|
| FPS, DT | 30, 1/30 | global |
| TRIM | 45 | clip T-pose trim |
| TRAJ_HORIZONS | {10,20,30} | MM features |
| MM_SEARCH_INTERVAL / jump_margin | 15 / 0.35 | MM |
| SEARCH_INTERVAL | 10 | MG greedy |
| BLEND_FRAMES | 12 | cross-fade |
| SMOOTH_WINDOW | 9 | root de-jitter |
| MG n_neighbors / tgt_stride | 16 / 2 (28 / 1 for loop) | graph build |
| pca_dim / tau_factor | 16 / 2.5 | graph build |
| plan_to beam / K | 64 / 10 | beam search |
| PHASE_READY/TAKEOFF/TOUCHDOWN/AFTER | 12 / 10 / 6 / 18 | jump phases |
| foot_thr | 0.13 m | flight detection |
| box hx/hy | 0.13 / 0.28 m | box heuristic |

---

## 12. Code map

```
motiongraph/
  config.py         constants (skeleton, features, phases, paths)
  g1_model.py       qpos<->csv, MuJoCo FK
  data.py           library build; skill + 5-phase labels; box heuristic
  kinematics.py     root stitching, blending, ease-to-terminal
  features.py       MM feature vectors
  commands.py       SpeedCommand -> predicted trajectory
  jumps.py          jump entries confined to `ready` (+ land/apex maps)
  motion_matching.py  feature DB + NN controller (+ jump_at)
  motion_graph.py   descriptor/edges, greedy follow_command/route, beam plan_to, jump
  footlock.py       foot-lock IK (sole-sphere DLS)
  cleanup.py        root de-jitter -> foot-lock
  render.py         offline MuJoCo -> MP4 (HUD, transition flash, boxes)
run_motion_matching.py / run_motion_graph.py   walk demos (task1/task2)
run_jump.py          jump demos: mm/mg task1/2, raw, loop, samebox
tools/diagnose.py    quality metrics + plots
tools/visualize_graph.py   interactive web graph
```
