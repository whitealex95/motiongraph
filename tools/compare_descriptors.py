"""Side-by-side close-up of the MG transition descriptors (joint_pca | mm_pose | +vel/height).

To make the descriptor's edge quality visible we run a transition STRESS test: force a
transition every `ride` frames (the most-forward one) with a short cross-fade, so the cuts
this descriptor produces are exercised back-to-back. Renders each descriptor close-up with a
tracking camera and a flash on every cut, then stacks them horizontally with titles.

Usage: python -m tools.compare_descriptors
"""
import os
import subprocess
import numpy as np

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.motion_graph import MotionGraph
from motiongraph.render import render_qpos, trace_labels
from motiongraph.kinematics import transform_qpos, alignment_to, blend_qpos

OUT = C.OUT_DIR
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PANELS = [("joint_pca", "joint_pca (full body)"),
          ("mm_pose", "mm_pose (feet only)"),
          ("mm_pose_vh", "mm_pose + vel & height")]


def stress_roll(g, start=1500, n=300, ride=8, blend=5):
    """Force a transition every `ride` frames (best forward match) with a short blend."""
    cur = start
    align = (-g.yaw[cur], g.xy[cur].copy(), np.zeros(2))
    out, tf, frozen, bl, since = [], [], None, 0, 0
    want = np.array([1.0, 0.0])                              # forward, body-local
    for _ in range(n):
        world = transform_qpos(g.qpos[cur], *align)[0]
        cwx, cwy = world[:2].copy(), g.yaw[cur] + align[0]
        if bl > 0:
            world = blend_qpos(frozen, world, 1 - bl / blend); bl -= 1
        out.append(world); tf.append(cur); since += 1
        if since >= ride or g._is_end(cur):
            src = cur if cur in g.trans else g._nearest_src(cur)
            if src >= 0:
                j = min(g.trans[src], key=lambda e: np.linalg.norm(g._seg_vel_local(e[0]) - want))[0]
                align = alignment_to(g.xy[j], g.yaw[j], cwx, cwy)
                frozen, bl, cur, since = world.copy(), blend, j, 0
                continue
        if not g._is_end(cur):
            cur += 1
    return np.asarray(out), np.array(tf)


def main():
    lib = load_library(C.JUMP_LIB_PATH)
    tmp = []
    for mode, _ in PANELS:
        g = MotionGraph(lib, n_neighbors=28, tgt_stride=1, desc_mode=mode)
        seq, tf = stress_roll(g)
        p = os.path.join(OUT, f"_cmp_{mode}.mp4")
        render_qpos(seq, p, trace=trace_labels(tf, lib), cam_dist=2.7, cam_elev=-8,
                    cam_azim=130, width=470, height=640)          # close-up, root-tracking
        tmp.append(p)

    # stack horizontally with a title bar per panel
    fc = "".join(
        f"[{i}:v]drawtext=fontfile={FONT}:text='{lbl}':fontcolor=white:fontsize=22:"
        f"box=1:boxcolor=black@0.6:boxborderw=8:x=(w-tw)/2:y=12[v{i}];"
        for i, (_, lbl) in enumerate(PANELS))
    fc += "".join(f"[v{i}]" for i in range(len(PANELS))) + f"hstack=inputs={len(PANELS)}[v]"
    out = os.path.join(OUT, "compare_descriptors.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *sum([["-i", t] for t in tmp], []),
                    "-filter_complex", fc, "-map", "[v]", "-r", str(C.FPS), out], check=True)
    for t in tmp:
        os.remove(t)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
