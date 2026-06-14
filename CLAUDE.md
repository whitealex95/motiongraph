# CLAUDE.md — project working notes (motiongraph, mm-only branch)

This branch is **motion-matching only**: the GenoView ("Simple Motion Matching", Holden)
controller, no motion graph. (The motion graph + A* planner live on `master`.)

## ALWAYS update the website
There is a GitHub Pages project site in `docs/` (served at the repo's github.io page).
**Whenever a demo video, a method/algorithm, a heuristic, or a hyperparameter changes,
you MUST also update `docs/`**: regenerate/copy the affected web video into
`docs/videos/` and update the text in `docs/index.html` (algorithm / heuristics /
hyperparameters sections) so the site never drifts from the code. Web videos are
compressed copies (see `tools/make_web.py`). Do this in the same change set as the code.

## Conventions
- Respond in **English** (replies, Markdown, comments) even when asked in Korean.
- Methods reference for paper writing lives in `PAPER.md` (keep it current too).
- Run env: `conda activate deploy_mujoco`, `MUJOCO_GL=glfw`. Python:
  `/home/jkim3662/miniconda3/envs/deploy_mujoco/bin/python`.

## Key entry points
- Locomotion demo (speed command -> command-following): `run_motion_matching.py`.
- Multimodal walk·run·jump (speed-driven): `run_locomotion.py`.
- Jump demos (jump on command / at a searched location / raw): `run_jump.py`.
- Path experiments (square ± box, reactive steering + reactive jump): `run_experiments.py`.
- Diagnostics (teleport/jitter/skating plots): `tools/diagnose.py`.
- Web build: `tools/make_web.py` (compress videos -> docs/videos, refresh site).

## Code map (MM-only)
- `motion_matching.py` — GenoView `MotionMatcher.step(speed,heading)` + offline `generate()`.
- `mm_features.py` — sim-root feature DB (`build_db`); `quat.py`, `springs.py` — math helpers.
- `data.py` — `build_library` (GMR loco + jump clips, L/R mirrored) -> `config.LIB_PATH`.
- `g1_model.py` (FK + `mirror_qpos`), `jumps.py` (run-up entries), `commands.py`,
  `cleanup.py` + `footlock.py` (root de-jitter + foot-lock IK), `render.py`.
- A faithful sibling port is `~/Projects/motionmatching-g1` (real-time keyboard viewer).
