"""Shared constants: paths, skeleton layout, and feature settings."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "g1")
LIB_PATH = os.path.join(ROOT, "data", "motion_lib.npz")
SCENE_XML = os.path.join(ROOT, "assets", "unitree_g1", "scene.xml")
OUT_DIR = os.path.join(ROOT, "outputs")

FPS = 30
DT = 1.0 / FPS

# Jump skill: a walk base clip + CAMDM walk->jump->walk clips (G1-retargeted LAFAN1).
JUMP_DATA_DIR = os.path.join(ROOT, "data", "g1_jump")
JUMP_LIB_PATH = os.path.join(ROOT, "data", "motion_lib_jump.npz")
JUMP_BASE_WALK = "walk1_subject5"                       # CAMDM's main walk (arms down)
JUMP_CLIPS = ["walk_jump_walk", "walk_jump_walk2", "walk_jump_stop"]   # from JUMP_DATA_DIR
SKILLS = ["walk", "jump"]

# Per-frame jump phase (5 phases + walk). Flight = both feet airborne; the window
# lengths (frames) carve the surrounding run-up / push-off / landing into phases.
# Controllers may ENTER a jump only in `ready` and EXIT only after `after`.
JUMP_PHASES = ["walk", "ready", "takeoff", "flight", "touchdown", "after"]
PHASE_READY = 12       # run-up before the push-off (the only place to enter a jump)
PHASE_TAKEOFF = 10     # push-off / loading on the ground, just before lift-off
PHASE_TOUCHDOWN = 6    # landing impact (just after the feet hit)
PHASE_AFTER = 18       # landing absorption / recovery walk (the only place to exit)

# CSV / qpos layout. Dataset row = [pos(3), quat_xyzw(4), joints(29)] = 36.
# MuJoCo free-joint qpos = [pos(3), quat_wxyz(4), joints(29)] = 36 (same joint order).
NQ = 36
NJ = 29
ROOT_POS = slice(0, 3)
QUAT_XYZW = slice(3, 7)   # dataset order
JOINTS = slice(7, 36)

# Foot bodies used for motion-matching pose features (names from menagerie g1.xml).
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]

# Motion-graph transition descriptor: which feature decides "can I splice frame i->j?".
#   "mm_pose"    -- MM's 15-D pose feature (feet pos/vel + root vel); same representation MM
#                   matches on, so MG and MM share the pose space (no PCA needed; low-dim).
#   "mm_pose_vh" -- mm_pose + root height (z) + yaw rate (17-D): adds the height & turn-velocity
#                   that the local feet feature underrepresents.
#   "joint_pca"  -- the 62-D joint pose+velocity descriptor reduced to 16-D by PCA (Kovar-style
#                   full-body continuity).
MG_DESCRIPTOR = "mm_pose"

# Motion-matching feature config.
TRAJ_HORIZONS = [10, 20, 30]   # future sample frames (~0.33/0.67/1.0 s ahead)
SEARCH_INTERVAL = 10           # motion-graph decision interval (frames, ~0.33 s)
MM_SEARCH_INTERVAL = 15        # motion-matching search interval (~0.5 s; fewer jumps)
BLEND_FRAMES = 12              # cross-fade length at a jump/transition (~0.4 s)
SMOOTH_WINDOW = 9              # Savitzky-Golay window (frames) for root de-jitter

# Every LAFAN1 clip begins and ends in a T-pose (arms out) that blends into the motion
# over ~1.5 s. We DROP the first/last TRIM frames of every clip (see data.py:_load_clip)
# so the T-pose never appears in the library or the generated motion.
TRIM = 45

# A SINGLE continuous walking sequence -> a unimodal motion distribution (one subject,
# one gait), so matching/graph never hop between styles or speeds. We use
# walk1_subject5 (the walk motion CAMDM uses as its main `walk`): natural arms-down
# posture (walk1_subject2 walks with the hands raised). ~258 s, speeds up to ~1.3 m/s.
LOCO_CLIPS = ["walk1_subject5"]
