"""Offline MuJoCo renderer: qpos sequence -> mp4 (headless, MUJOCO_GL=glfw)."""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
import mujoco
import imageio.v2 as imageio

from . import config as C


def _add_sphere(scene, pos, size, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, size, size]), np.asarray(pos, np.float64),
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def render_qpos(qpos_seq, out_path, markers_fn=None, fps=C.FPS,
                width=1280, height=720, cam_dist=3.5, cam_elev=-18, cam_azim=120):
    """Render frames with a camera tracking the root.

    markers_fn(frame_idx) -> list of (world_pos, size, rgba) spheres to overlay.
    """
    model = mujoco.MjModel.from_xml_path(C.SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = cam_dist, cam_elev, cam_azim

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, quality=8, macro_block_size=None)
    for t, q in enumerate(qpos_seq):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        cam.lookat[:] = [q[0], q[1], 0.8]                 # follow root
        renderer.update_scene(data, cam)
        if markers_fn is not None:
            for pos, size, rgba in markers_fn(t):
                _add_sphere(renderer.scene, pos, size, rgba)
        writer.append_data(renderer.render())
    writer.close()
    renderer.close()
    print(f"Wrote {out_path} ({len(qpos_seq)} frames @ {fps} fps)")
