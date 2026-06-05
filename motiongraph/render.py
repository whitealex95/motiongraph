"""Offline MuJoCo renderer: qpos sequence -> mp4 (headless, MUJOCO_GL=glfw).

Optionally overlays, per frame, which clip / frame index is playing and flashes a
border + banner whenever the motion *jumps* to a non-consecutive frame (a graph
transition), so transitions are easy to see in the video.
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")
import glob
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from . import config as C

_FONT = next(iter(glob.glob("/usr/share/fonts/**/DejaVuSansMono-Bold.ttf", recursive=True)), None)
_FADE = 14   # frames a transition banner/border stays lit


def _font(sz):
    try:
        return ImageFont.truetype(_FONT, sz)
    except Exception:
        return ImageFont.load_default()


def trace_labels(tframe, lib):
    """Per-frame {clip, idx, skill, trans, from} from a library-frame index trace."""
    names, cid, fic = lib["clip_names"], lib["clip_id"], lib["frame_in_clip"]
    skill = lib["skill"] if "skill" in lib else None
    out = []
    for i, f in enumerate(tframe):
        f = int(f)
        trans = i > 0 and int(tframe[i]) != int(tframe[i - 1]) + 1     # non-consecutive -> jumped
        p = int(tframe[i - 1]) if i > 0 else f
        out.append(dict(clip=str(names[cid[f]]), idx=int(fic[f]),
                        skill=(C.SKILLS[int(skill[f])] if skill is not None else ""),
                        trans=trans, from_clip=str(names[cid[p]]), from_idx=int(fic[p])))
    return out


def _flash(trace):
    """Per-frame (flash intensity 0..1, transition label) decaying after each jump."""
    n = len(trace)
    fl = np.zeros(n)
    info = [None] * n
    for i, l in enumerate(trace):
        if l["trans"]:
            for j in range(i, min(n, i + _FADE)):
                v = 1.0 - (j - i) / _FADE
                if v > fl[j]:
                    fl[j], info[j] = v, l
    return fl, info


def _overlay(img, label, flash, tinfo, t, fps):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    W, H = im.size
    big, mid, sml = _font(34), _font(26), _font(22)

    # top-left status panel: clip / frame index / time / skill
    lines = [f"clip : {label['clip']}", f"frame: {label['idx']:04d}", f"time : {t/fps:5.2f}s"]
    d.rectangle([14, 14, 14 + 360, 14 + 30 * len(lines) + 14], fill=(0, 0, 0, 140))
    for k, ln in enumerate(lines):
        d.text((26, 22 + 30 * k), ln, font=mid, fill=(235, 235, 235, 255))
    if label["skill"] == "jump":
        d.text((26, 22 + 30 * len(lines)), "JUMP", font=big, fill=(255, 150, 0, 255))

    # transition flash: border + banner naming the from->to frames
    if flash > 0.02 and tinfo is not None:
        col = (0, 220, 255)                                  # cyan
        bw = int(4 + 22 * flash)
        for w in range(bw):
            a = int(220 * flash * (1 - w / max(bw, 1)))
            d.rectangle([w, w, W - 1 - w, H - 1 - w], outline=col + (a,))
        banner = f"↪ TRANSITION   {tinfo['from_clip']} #{tinfo['from_idx']}  →  {tinfo['clip']} #{tinfo['idx']}"
        tw = d.textlength(banner, font=sml)
        by = H - 60                                          # bottom centre, clear of the HUD
        d.rectangle([(W - tw) / 2 - 16, by - 6, (W + tw) / 2 + 16, by + 30], fill=(0, 0, 0, int(190 * flash)))
        d.text(((W - tw) / 2, by), banner, font=sml, fill=col + (int(255 * min(1, flash + 0.3)),))
    return np.asarray(im)


def render_qpos(qpos_seq, out_path, markers_fn=None, trace=None, box=None, boxes=None,
                fps=C.FPS, width=1280, height=720, cam_dist=3.5, cam_elev=-18, cam_azim=120,
                cam_fixed=None):
    """Render frames with a root-tracking camera (or a fixed camera when cam_fixed is set).

    markers_fn(frame_idx) -> [(world_pos, size, rgba)] spheres to overlay.
    trace: optional list of trace_labels() dicts (len == frames) for the HUD/flash.
    box / boxes: a static box dict(pos, half, rgba=, label=) -- or a list of them.
    cam_fixed: (cx, cy, cz) look-at for a stationary wide camera (e.g. a whole path).
    """
    boxlist = (boxes or []) + ([box] if box else [])
    model = mujoco.MjModel.from_xml_path(C.SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = cam_dist, cam_elev, cam_azim
    flash, info = (_flash(trace) if trace else (None, None))
    labels = [b["label"] for b in boxlist if b.get("label")]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, quality=8, macro_block_size=None)
    for t, q in enumerate(qpos_seq):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        cam.lookat[:] = cam_fixed if cam_fixed is not None else [q[0], q[1], 0.8]
        renderer.update_scene(data, cam)
        for b in boxlist:                                 # predefined obstacles, always present
            _add_box(renderer.scene, b["pos"], b["half"],
                     b.get("rgba", [0.6, 0.4, 0.2, 0.9]), b.get("mat"))
        if markers_fn is not None:
            for pos, size, rgba in markers_fn(t):
                _add_sphere(renderer.scene, pos, size, rgba)
        img = renderer.render()
        if trace is not None:
            img = _overlay(img, trace[min(t, len(trace) - 1)], flash[t], info[t], t, fps)
        if labels:
            img = _box_labels(img, labels)
        writer.append_data(img)
    writer.close()
    renderer.close()
    print(f"Wrote {out_path} ({len(qpos_seq)} frames @ {fps} fps)")


def _box_labels(img, texts):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    W = im.size[0]
    f = _font(24)
    for k, text in enumerate(texts):
        tw = d.textlength(text, font=f)
        y = 16 + k * 40
        d.rectangle([W - tw - 36, y, W - 12, y + 34], fill=(90, 55, 20, 200))
        d.text((W - tw - 24, y + 4), text, font=f, fill=(255, 215, 150, 255))
    return np.asarray(im)


def _add_box(scene, pos, half, rgba, mat=None):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    m = np.eye(3).ravel() if mat is None else np.asarray(mat, np.float64).ravel()
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.asarray(half, np.float64),
                        np.asarray(pos, np.float64), m, np.asarray(rgba, np.float32))
    scene.ngeom += 1


def _add_sphere(scene, pos, size, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, size, size]), np.asarray(pos, np.float64),
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    scene.ngeom += 1
