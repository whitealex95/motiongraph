"""Post-process a generated qpos sequence: de-jitter the root, then lock feet.

Stitching/searching makes the root velocity discontinuous at every jump (the pose
cross-fade does not smooth translation). A short Savitzky-Golay filter on the root
removes that high-frequency sawtooth; foot-lock IK then removes the residual foot
sliding on the smoothed motion.
"""
import numpy as np
from scipy.signal import savgol_filter

from . import config as C
from .footlock import footlock


def cleanup(qpos_seq, model=None, smooth=True, lock=True):
    out = np.array(qpos_seq, dtype=np.float64)
    if smooth and len(out) > C.SMOOTH_WINDOW:
        out[:, 0:3] = savgol_filter(out[:, 0:3], C.SMOOTH_WINDOW, 3, axis=0)  # root pos only
    if lock:
        out = footlock(out, model)
    return out
