"""Interactive web visualization of the motion graph -> outputs/motion_graph.html.

Two linked views (self-contained HTML, Plotly via CDN):
  1. A 3D animated G1 skeleton of a graph-generated walk (play / scrub / orbit).
  2. The motion-graph structure: every frame embedded in 2-D pose space (PCA of the
     transition descriptor), transition edges drawn in light grey, and the path the
     greedy walk actually took -- blue where it plays a clip, red where it jumps a
     transition edge.

Usage: python -m tools.visualize_graph
"""
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

ANIM_STRIDE = 3          # keep every Nth frame in the 3-D animation (HTML size)
NODE_STRIDE = 4          # subsample graph nodes for the 2-D map
MAX_EDGES = 700          # subsample transition edges for clarity


def _skeleton_segments(model, data, qpos, center):
    """Bone line segments (x,y,z with None separators) for one pose, centred on root xy."""
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    xs, ys, zs = [], [], []
    for b in range(1, model.nbody):
        p = model.body_parentid[b]
        if p == 0:
            continue
        for pt in (data.xpos[p], data.xpos[b]):
            xs.append(pt[0] - center[0]); ys.append(pt[1] - center[1]); zs.append(pt[2])
        xs.append(None); ys.append(None); zs.append(None)
    return xs, ys, zs


def build():
    lib = load_library()
    g = MotionGraph(lib)
    gm = G1Model()

    # --- a graph-generated walk, with the traversal trace ---
    cmd = SpeedCommand([(0.0, 1.0, 0.0), (3.0, 1.1, np.deg2rad(40)),
                        (7.0, 1.1, np.deg2rad(-20)), (11.0, 0.9, np.deg2rad(-20))])
    out, tframe, tjump = g.follow_command(cmd, 13.0, start_frame=1500, return_trace=True)
    out = cleanup(out, gm)

    # --- 2-D pose-space embedding (PCA of the transition descriptor) ---
    desc = _descriptors(lib)
    X = desc - desc.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:2].T                                        # (N, 2)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.5, 0.5],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("Graph output: G1 walking (play / orbit)",
                        "Motion graph in pose space (blue=play, red=transition)"))

    # ---- right panel: graph structure (static) ----
    ns = np.arange(0, len(P), NODE_STRIDE)
    fig.add_trace(go.Scattergl(x=P[ns, 0], y=P[ns, 1], mode="markers",
                  marker=dict(size=3, color=ns, colorscale="Viridis", opacity=0.45),
                  name="frames", hovertext=[f"frame {i}" for i in ns], hoverinfo="text"),
                  row=1, col=2)
    edges = [(i, j) for i, es in g.trans.items() for j, _ in es]
    edges = [edges[k] for k in np.linspace(0, len(edges) - 1, min(MAX_EDGES, len(edges))).astype(int)]
    ex, ey = [], []
    for i, j in edges:
        ex += [P[i, 0], P[j, 0], None]; ey += [P[i, 1], P[j, 1], None]
    fig.add_trace(go.Scattergl(x=ex, y=ey, mode="lines",
                  line=dict(color="rgba(150,150,150,0.25)", width=1),
                  name="transition edges", hoverinfo="skip"), row=1, col=2)
    # traversal: continuation segments (blue) and transition jumps (red)
    cx, cy, jx, jy = [], [], [], []
    for k in range(1, len(tframe)):
        a, b = tframe[k - 1], tframe[k]
        seg = (jx, jy) if tjump[k] else (cx, cy)
        seg[0].extend([P[a, 0], P[b, 0], None]); seg[1].extend([P[a, 1], P[b, 1], None])
    fig.add_trace(go.Scattergl(x=cx, y=cy, mode="lines", line=dict(color="royalblue", width=2),
                  name="walk (play)"), row=1, col=2)
    fig.add_trace(go.Scattergl(x=jx, y=jy, mode="lines", line=dict(color="crimson", width=2),
                  name="walk (jump)"), row=1, col=2)

    # ---- left panel: 3-D animated skeleton ----
    centers = out[:, 0:2]
    seg0 = _skeleton_segments(gm.model, gm.data, out[0], centers[0])
    bones = go.Scatter3d(x=seg0[0], y=seg0[1], z=seg0[2], mode="lines",
                         line=dict(color="black", width=5), name="G1")
    fig.add_trace(bones, row=1, col=1)

    frames = []
    for k in range(0, len(out), ANIM_STRIDE):
        xs, ys, zs = _skeleton_segments(gm.model, gm.data, out[k], centers[k])
        frames.append(go.Frame(name=str(k), data=[go.Scatter3d(x=xs, y=ys, z=zs)],
                               traces=[4]))                  # trace index 4 = the skeleton
    fig.frames = frames

    fig.update_layout(
        title="Motion graph on a single LAFAN1 walk sequence (walk1_subject2)",
        scene=dict(xaxis=dict(range=[-0.7, 0.7], title=""), yaxis=dict(range=[-0.7, 0.7], title=""),
                   zaxis=dict(range=[0, 1.8], title=""), aspectmode="data",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))),
        updatemenus=[dict(type="buttons", x=0.05, y=0.05, showactive=False, buttons=[
            dict(label="play", method="animate",
                 args=[None, dict(frame=dict(duration=33, redraw=True), fromcurrent=True)]),
            dict(label="pause", method="animate",
                 args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(active=0, y=0, x=0.15, len=0.35, steps=[
            dict(method="animate", label=f"{int(f.name)/C.FPS:.1f}s",
                 args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
            for f in frames])])
    fig.update_xaxes(title="PCA 1", row=1, col=2)
    fig.update_yaxes(title="PCA 2", row=1, col=2)

    out_path = f"{C.OUT_DIR}/motion_graph.html"
    fig.write_html(out_path, include_plotlyjs="cdn", auto_play=False)
    print(f"Wrote {out_path}  ({len(frames)} animation frames, {len(edges)} edges shown)")
    try:
        fig.write_image(f"{C.OUT_DIR}/motion_graph_preview.png", width=1500, height=750, scale=1)
    except Exception as e:
        print("preview png skipped:", e)
    return out_path


if __name__ == "__main__":
    build()
