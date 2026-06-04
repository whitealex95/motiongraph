"""Motion matching: per-interval nearest-neighbour search over a feature database.

The query is [ trajectory-from-command | pose-of-current-frame ]. Searching for the
nearest database frame and jumping there continues the motion while following the
command. Root motion is stitched continuously and pose pops are cross-faded.
"""
import numpy as np
from scipy.spatial import cKDTree

from . import config as C
from . import features as F
from .kinematics import transform_qpos, alignment_to, blend_qpos


class MotionMatcher:
    def __init__(self, lib, traj_w=1.0, pose_w=1.0, min_z=0.6, jump_margin=0.35):
        self.lib = lib
        self.raw = F.compute_features(lib)
        self.feat, self.mean, self.std, self.w = F.standardize(self.raw, traj_w, pose_w)
        self.xy, self.yaw, self.qpos = lib["qpos"][:, 0:2], lib["yaw"], lib["qpos"]
        self.fic, self.lengths = lib["frame_in_clip"], lib["lengths"]
        self.clip_id = lib["clip_id"]
        self.jump_margin = jump_margin
        # search only upright frames so jumps never land in crouch/getup poses
        self.valid = np.where(lib["qpos"][:, 2] >= min_z)[0]
        self.tree = cKDTree(self.feat[self.valid])

    def _is_clip_end(self, i):
        return self.fic[i] >= self.lengths[self.clip_id[i]] - 1

    def _qstd(self, traj_block, cur):
        """Standardized query: command trajectory + current frame's pose features."""
        query = np.concatenate([traj_block, self.raw[cur, F.TRAJ_DIM:]])
        return ((query - self.mean) / self.std) * self.w

    def generate(self, command, seconds, start_frame=0, traj_fn=None):
        """Roll out for `seconds`; traj_fn(t,xy,yaw)->block overrides the command query.

        Hysteresis: at a search we only jump to the nearest neighbour if it is
        clearly better (by jump_margin) than simply continuing the current clip.
        This keeps the character on long continuous fragments -> less jitter/skating.
        """
        n = int(seconds * C.FPS)
        cur = start_frame
        dyaw, pivot, offset = -self.yaw[cur], self.xy[cur].copy(), np.zeros(2)  # start at origin, +x
        out, frozen, blend_left = [], None, 0
        for step in range(n):
            t = step * C.DT
            world = transform_qpos(self.qpos[cur], dyaw, pivot, offset)[0]
            cwx, cwy = world[0:2].copy(), self.yaw[cur] + dyaw
            if blend_left > 0:
                world = blend_qpos(frozen, world, 1 - blend_left / C.BLEND_FRAMES)
                blend_left -= 1
            out.append(world)

            if step > 0 and (step % C.MM_SEARCH_INTERVAL == 0 or self._is_clip_end(cur)):
                block = traj_fn(t, cwx, cwy) if traj_fn else command.trajectory(cwx, cwy, t)
                qstd = self._qstd(block, cur)
                dist_best, vi = self.tree.query(qstd)
                best = int(self.valid[vi])
                end = self._is_clip_end(cur)
                # continuing costs the query's distance to the next frame's features
                cont = 1e9 if end else float(np.linalg.norm(qstd - self.feat[cur + 1]))
                if end or dist_best < cont * (1 - self.jump_margin):
                    jump = not (self.clip_id[best] == self.clip_id[cur] and 0 <= best - cur <= 2)
                    dyaw, pivot, offset = alignment_to(self.xy[best], self.yaw[best], cwx, cwy)
                    if jump:
                        frozen, blend_left = world.copy(), C.BLEND_FRAMES
                    cur = best
                elif not end:
                    cur += 1
            elif not self._is_clip_end(cur):
                cur += 1
        return np.asarray(out)
