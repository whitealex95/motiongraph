"""Shared constants: paths, skeleton layout, and feature settings."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "g1")
LIB_PATH = os.path.join(ROOT, "data", "motion_lib.npz")
SCENE_XML = os.path.join(ROOT, "assets", "unitree_g1", "scene.xml")
OUT_DIR = os.path.join(ROOT, "outputs")

FPS = 30
DT = 1.0 / FPS

# CSV / qpos layout. Dataset row = [pos(3), quat_xyzw(4), joints(29)] = 36.
# MuJoCo free-joint qpos = [pos(3), quat_wxyz(4), joints(29)] = 36 (same joint order).
NQ = 36
NJ = 29
ROOT_POS = slice(0, 3)
QUAT_XYZW = slice(3, 7)   # dataset order
JOINTS = slice(7, 36)

# Foot bodies used for motion-matching pose features (names from menagerie g1.xml).
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]

# Motion-matching feature config.
TRAJ_HORIZONS = [10, 20, 30]   # future sample frames (~0.33/0.67/1.0 s ahead)
SEARCH_INTERVAL = 10           # motion-graph decision interval (frames, ~0.33 s)
MM_SEARCH_INTERVAL = 15        # motion-matching search interval (~0.5 s; fewer jumps)
BLEND_FRAMES = 12              # cross-fade length at a jump/transition (~0.4 s)
SMOOTH_WINDOW = 9              # Savitzky-Golay window (frames) for root de-jitter

# Trim this many frames from each clip's start/end (LAFAN1 clips begin/end in a
# T-pose that blends into the motion over ~1.5 s).
TRIM = 45

# A SINGLE continuous walking sequence -> a unimodal motion distribution (one
# subject, one gait), so matching/graph never hop between styles or speeds.
# walk1_subject2 is ~258 s with lots of turning and walk speeds up to ~1.3 m/s.
LOCO_CLIPS = ["walk1_subject2"]
