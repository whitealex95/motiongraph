"""Planar root-motion stitching: place library frames into a continuous world path.

A segment is played under one planar "alignment" (a yaw rotation about world Z +
xy translation). Playing contiguous library frames under a fixed alignment lets the
clip's own root motion carry the character; at a jump we recompute the alignment so
the new frame coincides with the current world pose -> C0-continuous root path.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def transform_qpos(qpos, dyaw, pivot, offset):
    """Apply planar alignment T(p)=Rz(dyaw)(p-pivot)+offset to qpos rows (..,36)."""
    q = np.atleast_2d(qpos).astype(np.float64).copy()
    q[:, 0:2] = (rotz(dyaw) @ (q[:, 0:2] - pivot).T).T + offset
    xyzw = q[:, [4, 5, 6, 3]]                                   # wxyz -> xyzw
    rot = R.from_euler("z", dyaw) * R.from_quat(xyzw)
    q[:, 3:7] = rot.as_quat()[:, [3, 0, 1, 2]]                  # xyzw -> wxyz
    return q


def alignment_to(lib_xy, lib_yaw, world_xy, world_yaw):
    """(dyaw, pivot, offset) mapping a library planar pose onto a target world pose."""
    return world_yaw - lib_yaw, np.asarray(lib_xy, float), np.asarray(world_xy, float)


def ease_to_terminal(out, term, k):
    """Ease the last k frames of `out` onto terminal pose `term` (pose + position)."""
    k = min(k, len(out))
    for j in range(k):
        w = (j + 1) / k
        idx = len(out) - k + j
        planned = out[idx].copy()                       # save before blend overwrites position
        eased = blend_qpos(planned, term, w)            # joints + orientation -> terminal
        eased[0:3] = (1 - w) * planned[0:3] + w * term[0:3]   # position lerp from planned pose
        out[idx] = eased
    return out


def blend_qpos(frozen, live, w):
    """Ease joints + root orientation from a frozen pose toward the live pose (w:0->1).

    Root position is taken from `live` so the world path stays continuous; only the
    body pose (joint angles + root tilt/heading) is smoothed across a jump.
    """
    out = live.copy()
    out[7:36] = (1 - w) * frozen[7:36] + w * live[7:36]
    from scipy.spatial.transform import Slerp
    key = R.concatenate([R.from_quat(frozen[[4, 5, 6, 3]]), R.from_quat(live[[4, 5, 6, 3]])])
    out[3:7] = Slerp([0, 1], key)([w])[0].as_quat()[[3, 0, 1, 2]]
    return out
