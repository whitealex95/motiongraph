# Paper notes — GenoView Motion Matching with a Jump Skill on the Unitree G1 (mm-only)

A living methods reference for the **motion-matching-only** branch. (Usage/quickstart lives
in `README.md`.) Sections can be lifted into Method / Implementation / Experiments. The
motion graph + A\* planner are on `master`.

---

## 1. Overview & contributions

We drive the **Unitree G1** humanoid with a faithful port of **GenoView**'s ("Simple Motion
Matching", Holden) real-time controller, on **LAFAN1** mocap retargeted to the G1 with GMR,
plus a *phase-segmented* **jump** skill.

Things demonstrated:
1. A reactive, real-time matcher (`step(speed, heading)`) — per-clip KD-trees, critically
   damped trajectory springs, **inertialized** transitions, a velocity-integrated simulation
   root — with an offline `generate()` wrapper for rendering.
2. **Speed-driven gait**: one command selects walk vs run from a multimodal database (no
   state machine), with an L/R-mirrored library.
3. A **jump skill**: per-frame 5-phase labels (ready/take-off/flight/touch-down/after), with
   the hard guarantee that a jump is **entered only in `ready`** and **exited only after
   `after`**; the matcher triggers it from a run-up, reactively over a box.

---

## 2. Data

- **Source.** LAFAN1 (Ubisoft), the three GenoView `subject5` clips — **walk, run,
  pushAndStumble** — retargeted to the G1 with **GMR** (General Motion Retargeting;
  `data/g1_gmr_lafan1/*.pkl`, copied from `~/Projects/GMR`). GMR's `.pkl` (`root_pos`,
  `root_rot` xyzw, `dof_pos`@29, 30 fps) maps straight to qpos (`data._gmr_to_qpos`); GMR
  targets `g1_mocap_29dof.xml`, whose joint order is identical to our menagerie G1 (verified
  by render). Retargeting is **kinematic only**, so playback is likewise kinematic.
- **Trim.** Each clip uses **GenoView's exact per-clip window** (`config.CLIP_TRIM`, 60 fps
  ranges halved: walk `[160:15518]`, run `[172:14136]`, pushAndStumble `[397:706]` →
  `[198:353]`, the ~5 s in-place stumble event). Clips not listed fall back to a symmetric
  `TRIM = 45` T-pose cut.
- **Jump skill.** Three CAMDM `walk_jump_walk*` clips (`data/g1_jump/`, 30 fps CSV): short,
  nearly straight `walk → jump → walk`. Phase-labeled (`_label_jump`); only these carry
  `skill==1`, so locomotion (incl. running's natural flight) never matches into a jump.
- **Library** (`data.build_library` → `config.LIB_PATH = motion_lib.npz`, 30712 frames).
  Locomotion + jump clips concatenated, each added twice (normal + **L/R mirrored**,
  GenoView-style, `g1_model.mirror_qpos`). Arrays: `qpos (N,36)`, `feet_world (N,2,3)` (FK),
  `yaw`, `clip_id`, `frame_in_clip`, `lengths`, `clip_names`, `skill`, `phase`, and per-jump
  `jump_entry/takeoff/apex/land/continues/box`.
- **Format.** qpos = 36 floats: root `(x,y,z)` + quat (**wxyz** in qpos; the GMR pkl and CSV
  store **xyzw**, reordered on load) + 29 joint angles in canonical Unitree order.

---

## 3. Representation & root

- **State.** A frame is the 36-D `qpos`. Forward kinematics (MuJoCo `mj_kinematics`) gives
  foot/body world positions for the features and rendering.
- **Root.** The matcher does *not* stitch fragments by re-alignment; it carries a per-clip
  Savitzky-Golay-smoothed **simulation root** (ground xy + facing) and integrates it from the
  matched clip's smooth velocity (§4), placing the pelvis back as a local offset. So the
  world path is continuous by construction.
- **Post-process** (`cleanup.py`): Savitzky-Golay smoothing of root **xy** (window 9; z is
  left untouched so jump peaks survive) → **foot-lock IK** (§8).

---

## 4. Motion matching (`motion_matching.py`) — GenoView controller

A faithful port of the GenoView ("Simple Motion Matching", Holden) real-time controller
(reference: `~/Projects/motionmatching-g1`). Reactive, step-based (`step(speed, heading)`);
`generate()` wraps it to roll out an offline sequence from a `SpeedCommand`.

- **Simulation root + DB** (`mm_features.py`). Per clip, a Savitzky-Golay-smoothed sim root
  (ground xy + heading; windows 15/31) is what the matcher tracks; the pelvis is a *local
  offset* of it. The library is **L/R mirrored** (each clip added twice, `g1_model.mirror_qpos`).
- **Feature vector** (27-D, in the sim-root frame, **per-block** standardized): foot
  positions (6) + foot & pelvis velocities (9) + future sim-root offset (6) and facing (6)
  at horizons `{10,20,30}` frames (= pose 15 + trajectory 12).
- **Search.** Every `SEARCH_TIME = 0.15 s`, query `= [current pose | desired trajectory]`;
  nearest neighbour over **per-clip cKDTrees** (each trims its last `HORIZONS[-1]=30` frames
  so a full future always exists; jump frames excluded), seeded with a stay-in-clip bias
  `CURRENT_BIAS = 0.01` and `eps = APPROX_BIAS`.
- **Transitions = inertialization.** On a switch, the pose discontinuity (joints +
  pelvis-local pos/rot **and their velocities**) is captured as a decaying-spring offset and
  bled to zero over `INERT_HALFLIFE = 0.075 s` (`springs.py`) — no cross-fade, momentum
  preserved. The desired trajectory is predicted with **critically-damped springs**
  (`VEL/ROT_HALFLIFE = 0.2 s`); the world root is integrated from the matched clip's smooth
  velocity (`rootPos += R(rootRot)·clipVelLocal·dt`, `rootYaw += yawRate·dt`).
- **Jump** is triggered (not searched): inertialize into the best-matching `ready` run-up,
  ride through landing. In the path experiments the matcher is *steered* (go-to-point heading)
  and jumps when the box is ahead and within one jump's forward reach — reactive, no planner,
  so the box jumps are best-effort (the apex lands on the box by triggering one jump-length
  ahead).

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
  `phase==ready` (and `continues==True`), mapping each to its `land`/`apex`. The matcher's
  `best_jump_entry(cur)` = the ready frame whose features best match the current frame (for a
  smooth take-off). **A jump can never be cut into mid-air.**

- **Exit confined to `after`.** A triggered jump is played as a **locked segment**
  `entry → after_end = land+1+PHASE_TOUCHDOWN+PHASE_AFTER` (no decisions inside), then
  locomotion resumes. Because normal transition targets exclude all `skill==1` frames,
  resumption necessarily lands on a **walk** frame past `after`. Hence exit ⊂ post-`after`.

- **Why two mechanisms.** Entry confinement = restricting the candidate set
  (`best_jump_entry`). Exit confinement = (a) locked playback through `after` + (b)
  skill-based exclusion of jump frames from normal targets.

---

## 7. Exact box-jump — rigid placement / baked box (`run_exact_box.py`)

The reactive box trigger (§Experiments) is *best-effort*: the locked jump segment plays from
wherever the steered walk arrives, so the apex misses a **fixed** box by ~0.1–0.2 m. We make the
apex land on the box **exactly and repeatably** — the "planned sequence" feel of the `master`
A\* demo, but on the reactive matcher — **without distorting the jump**.

**Why arc-warping fails.** Apex x is a *step* function of the trigger time: `best_jump_entry`
snaps to a discrete `ready` frame, so apex x jumps ~0.37 m between adjacent triggers (a finer
trigger search does **not** help — measured residual floor ~0.1–0.18 m). Pushing that residual
through the ~14-frame airborne arc (e.g. a smoothstep root bump) visibly distorts horizontal
velocity — forward speed dips to ~0.4 m/s at the top then lurches to ~3 m/s on the way down (the
jump "stalls" at the apex). So we move the **geometry**, never the arc.

1. **Single fixed box** (`gen_single`). Coarse-search the trigger whose natural apex is nearest
   the box (only to keep the start near the origin), then **rigidly translate the entire rollout**
   by `d = box_xy − apex_xy` (`_rigid_to`). A rigid shift preserves every velocity and foot
   contact ⇒ **zero distortion**; the jump is the untouched mocap, just repositioned. Apex == box
   exactly; the only visible effect is the robot's start moving by the sub-0.2 m residual.
2. **Course** (`gen_course` + `_walk_jumps`). Walk the +x line (a gentle go-to-point heading,
   aim 2 m ahead on `y=0`, holds it straight against reactive drift), trigger jumps at scheduled
   times, segment the flights (`_flight_segments`), and **lock each box to its jump's natural
   apex** ("motion baked into the object"). Exact by construction, again no warp. The trailing
   walk after the last landing is trimmed.

**Why not literal A\*.** A\* expands a discrete transition graph; the matcher is continuous and
reactive with no such graph (building one = the `master` motion-graph planner). For "hit this
fixed point," rigid placement / baking is exact *and* artifact-free. A\* only earns its keep with
branching choices (which box / what order). Demos: `exact_single_box.mp4`, `exact_course.mp4`.

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

## 10. Limitations

- Kinematic playback (no dynamics); residual foot slip is inherent to the retargeted data.
- LAFAN1 walk/run ⇒ ~5 m turn radius, so a "square" loop is in practice rounded.
- Jump heights are low (data are small hops) ⇒ low boxes.
- Reactive, no planner: the matcher follows a steered command and can't be guaranteed to
  re-acquire a fixed world point after a loop (the box jumps are best-effort).

---

## 11. Hyperparameters (defaults)

| name | value | where |
|---|---|---|
| FPS, DT | 30, 1/30 | global |
| TRIM / CLIP_TRIM | 45 / GenoView windows | clip trim |
| HORIZONS | {10,20,30} frames | features |
| SEARCH_TIME / CURRENT_BIAS / APPROX_BIAS | 0.15 s / 0.01 / 0.01 | search |
| INERT / VEL / ROT half-life | 0.075 / 0.2 / 0.2 s | spring + inertialization |
| ROOT_POS/DIR_SMOOTH | 15 / 31 | sim-root savgol |
| search-tail (per clip) | 30 frames | full-future guarantee |
| MAX_SPEED / WALK_SCALE | 5 m/s / 0.4 | command speeds |
| SMOOTH_WINDOW | 9 | root de-jitter (cleanup) |
| PHASE_READY/TAKEOFF/TOUCHDOWN/AFTER | 12 / 10 / 6 / 18 | jump phases |
| foot_thr | 0.13 m | flight detection |
| box hx/hy | 0.13 / 0.28 m | box heuristic |

---

## 12. Code map

```
motiongraph/
  config.py          constants (skeleton, jump phases, paths, MM settings)
  g1_model.py        qpos<->csv/pkl, MuJoCo FK, sagittal mirror_qpos
  data.py            library build (mirrored); skill + 5-phase labels; box heuristic
  mm_features.py     GenoView feature DB (sim-root, per-block norm)
  quat.py / springs.py   quaternion + spring/inertialization helpers
  commands.py        SpeedCommand schedule -> (speed, heading)
  jumps.py           jump entries confined to `ready` (+ land/apex maps)
  motion_matching.py GenoView MotionMatcher: step(speed,heading) + generate()
  footlock.py        foot-lock IK (sole-sphere DLS)
  cleanup.py         root de-jitter -> foot-lock
  render.py          offline MuJoCo -> MP4 (HUD, transition flash, boxes, GenoView command gizmo)
run_locomotion.py    speed-driven walk -> run -> jump demo
run_motion_matching.py   command-following locomotion demo
run_experiments.py   square-path experiments (+ reactive box jump)
run_exact_box.py     exact box-jump: fixed single box (rigid placement) + course (box baked to apex)
run_jump.py          jump demos: on command / at a searched location / raw
tools/diagnose.py    quality metrics + plots
```
