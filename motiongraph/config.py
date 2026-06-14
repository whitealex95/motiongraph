"""Shared constants: paths, skeleton layout, and GenoView motion-matching settings."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_XML = os.path.join(ROOT, "assets", "unitree_g1", "scene.xml")
OUT_DIR = os.path.join(ROOT, "outputs")

FPS = 30
DT = 1.0 / FPS

# The motion library: the three GenoView clips (subject5) -- walk, run, pushAndStumble --
# retargeted to the G1 with GMR (General Motion Retargeting; .pkl under data/g1_gmr_lafan1/,
# copied from ~/Projects/GMR), plus the CAMDM walk->jump->walk clips. Built once into LIB_PATH,
# each clip added twice (normal + L/R mirrored, GenoView-style). All locomotion is one skill;
# the jump clips are the only separate skill (entered only via a run-up).
GMR_DATA_DIR = os.path.join(ROOT, "data", "g1_gmr_lafan1")        # GMR-retargeted LAFAN1 (.pkl)
JUMP_DATA_DIR = os.path.join(ROOT, "data", "g1_jump")             # CAMDM jump clips (.csv)
LIB_PATH = os.path.join(ROOT, "data", "motion_lib.npz")
LOCO_CLIPS = ["walk1_subject5", "run1_subject5", "pushAndStumble1_subject5"]
JUMP_CLIPS = ["walk_jump_walk", "walk_jump_walk2", "walk_jump_stop"]
SKILLS = ["walk", "jump"]

# Per-frame jump phase (5 phases + walk). Flight = both feet airborne; the window lengths
# (frames) carve the surrounding run-up / push-off / landing. A jump is ENTERED only in
# `ready` and EXITED only after `after`.
JUMP_PHASES = ["walk", "ready", "takeoff", "flight", "touchdown", "after"]
PHASE_READY = 12       # run-up before the push-off (the only place to enter a jump)
PHASE_TAKEOFF = 10     # push-off / loading on the ground, just before lift-off
PHASE_TOUCHDOWN = 6    # landing impact (just after the feet hit)
PHASE_AFTER = 18       # landing absorption / recovery walk (the only place to exit)

# qpos layout (36-D), shared by the dataset and MuJoCo (same joint order):
#   [0:3]  root position (x, y, z) in world metres
#   [3:7]  root orientation quaternion -- DATASET stores xyzw, MuJoCo qpos stores wxyz
#          (csv_to_qpos / mirror_qpos handle the reorder; see g1_model.py)
#   [7:36] 29 joint angles (radians)
JOINTS = slice(7, 36)

# Foot bodies used for the pose features (names from menagerie g1.xml).
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]

# --- GenoView motion matching (Holden "Simple Motion Matching"); the MotionMatcher is a
# faithful port of ~/Projects/motionmatching-g1 (genoview_g1.py). All math/params mirror it.
HORIZONS = [10, 20, 30]        # future trajectory taps (frames) ~0.33/0.67/1.0 s @30fps
SEARCH_TIME = 0.15             # seconds between database searches
INERT_HALFLIFE = 0.075         # inertialization (pose-transition) blend half-life
VEL_HALFLIFE = 0.2             # desired-trajectory position spring half-life
ROT_HALFLIFE = 0.2             # desired-trajectory rotation spring half-life
CURRENT_BIAS = 0.01            # stay-in-clip bias seeded onto the current frame's distance
APPROX_BIAS = 0.01             # cKDTree eps: slightly approximate (faster) nearest-neighbour
ROOT_POS_SMOOTH = 15           # Savitzky-Golay window for the smoothed sim-root position
ROOT_DIR_SMOOTH = 31           # Savitzky-Golay window for the smoothed sim-root heading
MAX_SPEED = 5.0                # full-command speed (m/s); run pace
WALK_SCALE = 0.4               # walk = MAX_SPEED * WALK_SCALE
SMOOTH_WINDOW = 9              # Savitzky-Golay window (frames) for root de-jitter (cleanup.py)

# Per-clip [start:stop] trim windows MATCHING GenoView (orangeduck), 60 fps ranges halved to
# our 30 fps. GenoView hand-trims each clip to a chosen window rather than a symmetric T-pose
# cut; clips not listed fall back to the symmetric TRIM. pushAndStumble's window isolates the
# ~5 s stumble event out of the otherwise-ordinary clip.
#   GenoView 60 fps:  walk [160:15518]   run [172:14136]   pushAndStumble [397:706]
TRIM = 45
CLIP_TRIM = {
    "walk1_subject5": (80, 7759),
    "run1_subject5": (86, 7068),
    "pushAndStumble1_subject5": (198, 353),
}
