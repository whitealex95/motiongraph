# CLAUDE.md — project working notes (motiongraph)

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
- Walk demos: `run_motion_graph.py`, `run_motion_matching.py`.
- Jump demos: `run_jump.py` (mm/mg, raw, loop, samebox).
- Multimodal MM (walk+run+jump, speed-driven): `run_locomotion.py` (uses `LOCO_LIB_PATH`).
- Experiments (square path ± box): `run_experiments.py`.
- Diagnostics / web graph: `tools/diagnose.py`, `tools/visualize_graph.py`.
- Web build: `tools/make_web.py` (compress videos -> docs/videos, refresh site).
