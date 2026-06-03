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
    def __init__(self, lib, traj_w=1.0, pose_w=1.0, min_z=0.6):
        self.lib = lib
        self.raw = F.compute_features(lib)
        std, self.mean, self.std, self.w = F.standardize(self.raw, traj_w, pose_w)
        self.xy, self.yaw, self.qpos = lib["qpos"][:, 0:2], lib["yaw"], lib["qpos"]
        self.fic, self.lengths = lib["frame_in_clip"], lib["lengths"]
        self.clip_id = lib["clip_id"]
        # search only upright frames so jumps never land in crouch/getup poses
        self.valid = np.where(lib["qpos"][:, 2] >= min_z)[0]
        self.tree = cKDTree(std[self.valid])

    def _is_clip_end(self, i):
        return self.fic[i] >= self.lengths[self.clip_id[i]] - 1

    def search(self, traj_block, cur):
        """Nearest frame to [command trajectory | current pose]."""
        query = np.concatenate([traj_block, self.raw[cur, F.TRAJ_DIM:]])
        qstd = ((query - self.mean) / self.std) * self.w
        return int(self.valid[self.tree.query(qstd)[1]])

    def generate(self, command, seconds, start_frame=0, traj_fn=None):
        """Roll out for `seconds`; traj_fn(t,xy,yaw)->block overrides the command query."""
        n = int(seconds * C.FPS)
        cur = start_frame
        dyaw, pivot, offset = -self.yaw[cur], self.xy[cur].copy(), np.zeros(2)  # start at origin, +x
        out, frozen, blend_left = [], None, 0
        for step in range(n):
            t = step * C.DT
            world = transform_qpos(self.qpos[cur], dyaw, pivot, offset)[0]
            cwx, cwy = world[0:2].copy(), self.yaw[cur] + dyaw
            if blend_left > 0:
                w = 1 - blend_left / C.BLEND_FRAMES
                world = blend_qpos(frozen, world, w)
                blend_left -= 1
            out.append(world)

            if step > 0 and (step % C.SEARCH_INTERVAL == 0 or self._is_clip_end(cur)):
                block = traj_fn(t, cwx, cwy) if traj_fn else command.trajectory(cwx, cwy, t)
                best = self.search(block, cur)
                jump = not (self.clip_id[best] == self.clip_id[cur] and 0 <= best - cur <= 2)
                dyaw, pivot, offset = alignment_to(self.xy[best], self.yaw[best], cwx, cwy)
                if jump:
                    frozen, blend_left = world.copy(), C.BLEND_FRAMES
                cur = best
            elif not self._is_clip_end(cur):
                cur += 1
        return np.asarray(out)
