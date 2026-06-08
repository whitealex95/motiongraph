"""Ablation: how the MG transition descriptor and edge density affect blend quality.

We measure joint-space discontinuity at transition edges (the "pop" at a cut), split into
LEG joints (0:12) and UPPER-body joints (12:), in rad (L2):
  - all-edges mean/p90      : typical cut quality across the graph
  - best-per-node (min)     : the smoothest cut available from each frame (lower = more
                              choice for a continuity-seeking controller)
  - taken (greedy follow)   : the cuts greedy command-following actually makes

Usage: python -m tools.descriptor_ablation
"""
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.motion_graph import MotionGraph
from motiongraph.commands import SpeedCommand

CMD = SpeedCommand([(0.0, 1.0, 0.0), (4.0, 1.1, np.deg2rad(35))])   # walks + turns -> transitions


def gaps(q, pairs):
    """(leg, upper) L2 joint gaps for a list of (i, j) edges."""
    leg = np.array([np.linalg.norm(q[i, :12] - q[j, :12]) for i, j in pairs])
    up = np.array([np.linalg.norm(q[i, 12:] - q[j, 12:]) for i, j in pairs])
    return leg, up


def edge_stats(g, q):
    all_pairs, best_pairs = [], []
    for i, es in g.trans.items():
        ps = [(i, j) for j, _ in es]
        all_pairs += ps
        up = [np.linalg.norm(q[i, 12:] - q[j, 12:]) for j, _ in es]
        best_pairs.append(ps[int(np.argmin(up))])           # smoothest upper-body cut from i
    return all_pairs, best_pairs


def taken_gaps(g, q):
    _, tf, tj = g.follow_command(CMD, 12.0, start_frame=1500, return_trace=True)
    pairs = [(int(tf[k - 1]), int(tf[k])) for k in range(1, len(tf)) if tj[k]]
    return gaps(q, pairs) if pairs else (np.array([0.]), np.array([0.]))


def run(lib):
    q = lib["qpos"][:, C.JOINTS]
    print("=== descriptor variants (n_neighbors=28, tgt_stride=1) ===")
    print(f"{'mode':12s} {'edges':>7s} | {'all-up mean/p90':>16s} | {'best-up mean':>12s} | "
          f"{'taken up/leg':>14s}")
    for mode in ["joint_pca", "mm_pose", "mm_pose_vh"]:
        g = MotionGraph(lib, n_neighbors=28, tgt_stride=1, desc_mode=mode)
        allp, bestp = edge_stats(g, q)
        _, aup = gaps(q, allp)
        _, bup = gaps(q, bestp)
        tleg, tup = taken_gaps(g, q)
        print(f"{mode:12s} {len(allp):7d} | {aup.mean():6.3f} / {np.percentile(aup,90):6.3f} | "
              f"{bup.mean():12.3f} | {tup.mean():.3f} / {tleg.mean():.3f}")

    print("\n=== edge density sweep (mm_pose) ===")
    print(f"{'n_neigh / tau':>14s} {'edges':>7s} | {'all-up mean/p90':>16s} | {'best-up mean':>12s} | "
          f"{'taken up':>8s}")
    for nn, tau in [(16, 2.5), (28, 2.5), (60, 2.5), (120, 2.5), (60, 4.0), (120, 6.0)]:
        g = MotionGraph(lib, n_neighbors=nn, tgt_stride=1, tau_factor=tau,
                        desc_mode="mm_pose", cache=False)
        allp, bestp = edge_stats(g, q)
        _, aup = gaps(q, allp)
        _, bup = gaps(q, bestp)
        _, tup = taken_gaps(g, q)
        print(f"{nn:6d} / {tau:4.1f} {len(allp):7d} | {aup.mean():6.3f} / {np.percentile(aup,90):6.3f} | "
              f"{bup.mean():12.3f} | {tup.mean():8.3f}")


if __name__ == "__main__":
    run(load_library(C.JUMP_LIB_PATH))
