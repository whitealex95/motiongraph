"""Foot-lock IK cleanup: remove foot skating from a kinematic qpos sequence.

Skating is measured/fixed at the *sole contact spheres* (heel/toe corners), not the
ankle origin -- the ankle legitimately translates as the foot rolls. For each stance
phase we pin the dominant (lowest) sole sphere at a fixed world point and solve a
small damped-least-squares IK on that leg's 6 joints so the contact stops sliding.
Root and the rest of the body are untouched.
"""
import numpy as np
import mujoco

from . import config as C
from .g1_model import G1Model

_LEG_QPOS = {0: np.arange(7, 13), 1: np.arange(13, 19)}
_LEG_DOF = {0: np.arange(6, 12), 1: np.arange(12, 18)}
SOLE_GEOMS = {0: [15, 16, 17, 18], 1: [30, 31, 32, 33]}   # menagerie g1 foot corner spheres


def _sole_positions(g, qpos_seq):
    """World positions of the 4 sole spheres per foot -> (T, 2, 4, 3)."""
    m, d = g.model, g.data
    out = np.zeros((len(qpos_seq), 2, 4, 3))
    for t, q in enumerate(qpos_seq):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        for k, gs in SOLE_GEOMS.items():
            out[t, k] = [d.geom_xpos[gi] for gi in gs]
    return out


def footlock(qpos_seq, model=None, contact_z=0.03, vz_thr=0.3,
             min_len=3, iters=8, damping=0.04):
    """Return a copy of qpos_seq with planted-foot sliding removed.

    A foot is "planted" when its lowest sole sphere is grounded (height < contact_z)
    and not moving vertically (|dz/dt| < vz_thr) -- note we gate on *vertical* speed,
    not horizontal, so frames that are sliding horizontally still get locked.
    """
    g = model or G1Model()
    m, d = g.model, g.data
    out = np.array(qpos_seq, dtype=np.float64)

    sole = _sole_positions(g, out)                          # (T, 2, 4, 3)
    low = sole[:, :, :, 2].min(2)                           # lowest sphere height per foot
    vz = np.zeros_like(low)
    vz[1:] = np.abs(np.diff(low, axis=0)) / C.DT            # vertical speed of the lowest sphere
    contact = (low < contact_z) & (vz < vz_thr)

    jacp = np.zeros((3, m.nv))
    for k in (0, 1):
        gids, qcols, dcols = SOLE_GEOMS[k], _LEG_QPOS[k], _LEG_DOF[k]
        lo, hi = m.jnt_range[k * 6 + 1:k * 6 + 7].T            # this leg's joint limits
        for a, b in _intervals(contact[:, k], min_len):
            plant = a + int(sole[a:b, k, :, 2].min(1).argmin())        # most-planted frame
            pins = list(np.argsort(sole[plant, k, :, 2])[:2])          # two lowest spheres
            geoms = [gids[i] for i in pins]
            locks = [sole[plant, k, i, :].copy() for i in pins]        # fixed world targets
            for t in range(a, b):
                q = out[t].copy()
                for _ in range(iters):
                    d.qpos[:] = q
                    mujoco.mj_kinematics(m, d)
                    mujoco.mj_comPos(m, d)                  # required for mj_jac (fills cdof)
                    err = np.concatenate([lk - d.geom_xpos[g] for g, lk in zip(geoms, locks)])
                    if np.linalg.norm(err) < 1e-4:
                        break
                    rows = []
                    for g in geoms:                         # stack a 3-row block per pinned point
                        mujoco.mj_jac(m, d, jacp, None, d.geom_xpos[g].copy(), m.geom_bodyid[g])
                        rows.append(jacp[:, dcols].copy())
                    J = np.vstack(rows)                     # (3*npins) x 6
                    dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(len(err)), err)
                    dq = np.clip(dq, -0.1, 0.1)             # step clamp -> stable, no divergence
                    q[qcols] = np.clip(q[qcols] + dq, lo, hi)
                out[t] = q
    return out


def _intervals(mask, min_len):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return
    for s in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(s) >= min_len:
            yield int(s[0]), int(s[-1]) + 1
