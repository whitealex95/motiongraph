"""Quantify motion quality: root teleport/jitter and foot skating, with plots.

Metrics per generated clip:
  - teleport: max frame-to-frame root horizontal jump (m). A clean stitch stays
    under ~0.1 m/frame (3 m/s @ 30 fps); a spike means the root jumped.
  - jitter: RMS of root horizontal jerk (d3 x / dt3). Lower is smoother.
  - skating: mean horizontal speed of a foot while it is planted (m/s). ~0 is good.

Usage: python -m tools.diagnose [mm1 mm2 mg1 mg2 ...]   (default: all four)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mujoco
from motiongraph import config as C
from motiongraph.g1_model import G1Model
from motiongraph.footlock import SOLE_GEOMS

CONTACT_Z = 0.02   # a sole sphere is "in contact" below this world height


def _sole(model, out):
    """World positions of the 4 sole spheres per foot -> (T, 2, 4, 3)."""
    m, d = model.model, model.data
    pos = np.zeros((len(out), 2, 4, 3))
    for t, q in enumerate(out):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        for k, gs in SOLE_GEOMS.items():
            pos[t, k] = [d.geom_xpos[gi] for gi in gs]
    return pos


def metrics(out, model):
    xy = out[:, 0:2]
    v = np.linalg.norm(np.diff(xy, axis=0), axis=1) / C.DT            # root speed (m/s)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)               # per-frame jump (m)
    jerk = np.diff(xy, n=3, axis=0) / C.DT**3
    sole = _sole(model, out)                                          # (T, 2, 4, 3)
    h = sole[:, :, :, 2].min(2)                                       # lowest sole height per foot
    sspeed = np.linalg.norm(np.diff(sole[:, :, :, 0:2], axis=0), axis=3) / C.DT   # (T-1, 2, 4)
    vz = np.abs(np.diff(sole[:, :, :, 2], axis=0)) / C.DT             # per-sphere vertical speed
    # "planted" = grounded AND not landing/lifting; excludes normal heel-strike/toe-off motion
    contact = (sole[:-1, :, :, 2] < CONTACT_Z) & (vz < 0.25)
    skate = float(sspeed[contact].mean()) if contact.any() else 0.0  # mean slip while planted
    return dict(v=v, step=step, h=h, sspeed=sspeed, contact=contact,
                teleport=float(step.max()),
                jitter=float(np.sqrt((jerk**2).sum(1).mean())),
                skating=skate)


def plot(out, name, model, path):
    m = metrics(out, model)
    t = np.arange(len(out)) * C.DT
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    ax[0].plot(t[1:], m["v"], lw=1)
    ax[0].axhline(0, color="k", lw=0.3)
    ax[0].set_ylabel("root speed [m/s]")
    ax[0].set_title(f"{name}   teleport(max step)={m['teleport']:.3f} m   "
                    f"jitter(jerk RMS)={m['jitter']:.1f}   skating={m['skating']:.3f} m/s")

    for k, c, lab in [(0, "tab:blue", "L"), (1, "tab:orange", "R")]:
        ax[1].plot(t, m["h"][:, k], c=c, lw=1, label=f"foot {lab} sole height")
    ax[1].axhline(CONTACT_Z, color="k", lw=0.4, ls=":")
    ax[1].set_ylabel("sole height [m]")
    ax[1].legend(loc="upper right", fontsize=8)

    for k, c, lab in [(0, "tab:blue", "L"), (1, "tab:orange", "R")]:
        s = m["sspeed"][:, k].mean(1)                               # mean sole-sphere slip
        planted = np.where(m["contact"][:, k].any(1), s, np.nan)    # only while a sphere touches
        ax[2].plot(t[1:], s, c=c, lw=0.4, alpha=0.4)
        ax[2].plot(t[1:], planted, c=c, lw=1.6, label=f"foot {lab} in contact")
    ax[2].set_ylabel("sole horiz speed [m/s]\n(bold = contact = skating)")
    ax[2].set_xlabel("time [s]")
    ax[2].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return m


def main(which):
    import run_motion_matching as MM
    import run_motion_graph as MG
    gens = {"mm1": ("mm_task1", MM.gen_task1), "mm2": ("mm_task2", MM.gen_task2),
            "mg1": ("mg_task1", MG.gen_task1), "mg2": ("mg_task2", MG.gen_task2)}
    model = G1Model()
    os.makedirs(C.OUT_DIR, exist_ok=True)
    print(f"{'clip':10} {'teleport[m]':>12} {'jitter':>10} {'skating[m/s]':>13}")
    for key in which:
        name, gen = gens[key]
        out, _ = gen()
        path = f"{C.OUT_DIR}/diag_{name}.png"
        m = plot(out, name, model, path)
        print(f"{name:10} {m['teleport']:12.3f} {m['jitter']:10.1f} {m['skating']:13.3f}  -> {path}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["mm1", "mm2", "mg1", "mg2"])
