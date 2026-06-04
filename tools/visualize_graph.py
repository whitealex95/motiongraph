"""Interactive web visualization of the motion graph -> outputs/motion_graph[_jump].html.

Self-contained HTML (Plotly via CDN), two linked views:
  1. A 3-D animated G1 skeleton of a graph-generated motion (play / scrub / orbit).
  2. The FULL motion graph in 2-D pose space (PCA of the transition descriptor):
       - every frame is a node, coloured by skill (walk vs jump);
       - walk<->walk transition edges in faint grey (subsampled for legibility);
       - skill-transition edges highlighted: walk->jump (orange), jump->walk (green);
       - the top-5 walk->jump edges drawn extra thick (the best take-off points);
       - the generated traversal overlaid (blue walk, orange jump).

Usage: python -m tools.visualize_graph [walk|jump]   (default: jump)
"""
import sys
import numpy as np
import mujoco
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from motiongraph import config as C
from motiongraph.data import load_library
from motiongraph.motion_graph import MotionGraph, _descriptors
from motiongraph.commands import SpeedCommand
from motiongraph.cleanup import cleanup
from motiongraph.g1_model import G1Model

ANIM_STRIDE = 3
WW_EDGES = 3000          # subsample of walk<->walk edges drawn faintly


def _seg(model, data, qpos, center):
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    xs, ys, zs = [], [], []
    for b in range(1, model.nbody):
        p = model.body_parentid[b]
        if p == 0:
            continue
        for pt in (data.xpos[p], data.xpos[b]):
            xs.append(pt[0] - center[0]); ys.append(pt[1] - center[1]); zs.append(pt[2])
        xs += [None]; ys += [None]; zs += [None]
    return xs, ys, zs


def _edge_xy(P, pairs):
    ex, ey = [], []
    for i, j in pairs:
        ex += [P[i, 0], P[j, 0], None]; ey += [P[i, 1], P[j, 1], None]
    return ex, ey


def build(which="jump"):
    jump = which == "jump"
    lib = load_library(C.JUMP_LIB_PATH if jump else C.LIB_PATH)
    g = MotionGraph(lib)
    gm = G1Model()

    if jump:
        out, tframe, tphase = g.follow_with_jump(
            SpeedCommand([(0.0, 1.0, 0.0)]), 8.0, start_frame=1500, jump_at=3.5, return_trace=True)
        title = "Motion graph with a JUMP skill (walk1_subject2 + CAMDM walk->jump->walk)"
    else:
        out, tframe, tphase = g.follow_command(
            SpeedCommand([(0.0, 1.0, 0.0), (4.0, 1.1, np.deg2rad(40))]),
            12.0, start_frame=1500, return_trace=True)
        title = "Motion graph on a single LAFAN1 walk sequence (walk1_subject2)"
    out = cleanup(out, gm)

    # 2-D pose-space embedding (PCA of the transition descriptor)
    desc = _descriptors(lib)
    X = desc - desc.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:2].T
    skill = g.skill

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.46, 0.54],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("Generated motion (play / orbit)", "Full motion graph in pose space"))

    # ---- right: full graph ----
    # walk<->walk edges, faint
    ww = [(i, j) for (a, b), es in g.skill_edges.items() if a == 0 and b == 0 for i, j, _ in es]
    ww = [ww[k] for k in np.linspace(0, len(ww) - 1, min(WW_EDGES, len(ww))).astype(int)]
    ex, ey = _edge_xy(P, ww)
    fig.add_trace(go.Scattergl(x=ex, y=ey, mode="lines", hoverinfo="skip",
                  line=dict(color="rgba(160,160,160,0.18)", width=1), name="walk-walk edges"), 1, 2)
    # skill-transition edges, highlighted
    for pair, col, nm in [((0, 1), "darkorange", "walk->jump"), ((1, 0), "green", "jump->walk"),
                          ((1, 1), "red", "jump->jump")]:
        es = g.skill_edges.get(pair, [])
        ex, ey = _edge_xy(P, [(i, j) for i, j, _ in es])
        if ex:
            fig.add_trace(go.Scattergl(x=ex, y=ey, mode="lines",
                          line=dict(color=col, width=1.6), name=nm), 1, 2)
    # top-5 walk->jump edges extra thick (best take-off points)
    ex, ey = _edge_xy(P, [(i, j) for i, j, _ in g.skill_edges.get((0, 1), [])[:5]])
    if ex:
        fig.add_trace(go.Scattergl(x=ex, y=ey, mode="lines",
                      line=dict(color="darkorange", width=4), name="top-5 walk->jump"), 1, 2)
    # generated traversal: walk (blue) vs jump (orange) segments -- thin, under the nodes
    wx, wy, jx, jy = [], [], [], []
    for k in range(1, len(tframe)):
        a, b = tframe[k - 1], tframe[k]
        seg = (jx, jy) if tphase[k] else (wx, wy)
        seg[0].extend([P[a, 0], P[b, 0], None]); seg[1].extend([P[a, 1], P[b, 1], None])
    fig.add_trace(go.Scattergl(x=wx, y=wy, mode="lines", line=dict(color="rgba(65,105,225,0.7)", width=1.3),
                  name="traversal (walk)"), 1, 2)
    if jx:
        fig.add_trace(go.Scattergl(x=jx, y=jy, mode="lines", line=dict(color="rgba(255,140,0,0.9)", width=2.2),
                      name="traversal (jump)"), 1, 2)
    # nodes coloured by skill -- drawn last so they sit on top
    for s, col, nm, sz in [(0, "steelblue", "walk frames", 3), (1, "orangered", "jump frames", 7)]:
        m = skill == s
        fig.add_trace(go.Scattergl(x=P[m, 0], y=P[m, 1], mode="markers",
                      marker=dict(size=sz, color=col, opacity=0.45 if s == 0 else 0.95,
                                  line=dict(width=0.5, color="black") if s == 1 else None),
                      name=nm, hoverinfo="skip"), 1, 2)
    n_static = len(fig.data)                              # 3-D skeleton is appended next

    # ---- left: 3-D animation ----
    centers = out[:, 0:2]
    s0 = _seg(gm.model, gm.data, out[0], centers[0])
    fig.add_trace(go.Scatter3d(x=s0[0], y=s0[1], z=s0[2], mode="lines",
                  line=dict(color="black", width=5), name="G1"), 1, 1)
    frames = []
    for k in range(0, len(out), ANIM_STRIDE):
        xs, ys, zs = _seg(gm.model, gm.data, out[k], centers[k])
        frames.append(go.Frame(name=str(k), data=[go.Scatter3d(x=xs, y=ys, z=zs)], traces=[n_static]))
    fig.frames = frames

    fig.update_layout(
        title=title,
        scene=dict(xaxis=dict(range=[-0.7, 0.7], title=""), yaxis=dict(range=[-0.7, 0.7], title=""),
                   zaxis=dict(range=[0, 1.8], title=""), aspectmode="data",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))),
        updatemenus=[dict(type="buttons", x=0.02, y=0.05, showactive=False, buttons=[
            dict(label="play", method="animate",
                 args=[None, dict(frame=dict(duration=33, redraw=True), fromcurrent=True)]),
            dict(label="pause", method="animate",
                 args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(active=0, y=0, x=0.12, len=0.3, steps=[
            dict(method="animate", label=f"{int(f.name)/C.FPS:.1f}s",
                 args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
            for f in frames])])
    fig.update_xaxes(title="PCA 1", row=1, col=2)
    fig.update_yaxes(title="PCA 2", row=1, col=2)

    out_path = f"{C.OUT_DIR}/motion_graph{'_jump' if jump else ''}.html"
    fig.write_html(out_path, include_plotlyjs="cdn", auto_play=False)
    print(f"Wrote {out_path}  (nodes {len(P)}, walk->jump edges {len(g.skill_edges.get((0,1),[]))}, "
          f"jump->walk {len(g.skill_edges.get((1,0),[]))})")
    try:
        fig.write_image(out_path.replace(".html", "_preview.png"), width=1600, height=780)
    except Exception as e:
        print("preview skipped:", e)
    return out_path


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "jump")
