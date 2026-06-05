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
        self.skill = lib["skill"] if "skill" in lib else np.zeros(len(self.qpos), np.int32)
        # normal search: upright WALK frames only, so locomotion never produces a jump
        # (a jump happens only when triggered, via best_jump_entry).
        self.valid = np.where((lib["qpos"][:, 2] >= min_z) & (self.skill == 0))[0]
        self.tree = cKDTree(self.feat[self.valid])
        from .jumps import jump_entries
        self.jump_enter, self.jump_land_of, self.jump_apex_of = jump_entries(lib)   # pre-take-off run-up frames

    def _is_clip_end(self, i):
        return self.fic[i] >= self.lengths[self.clip_id[i]] - 1

    def best_jump_entry(self, cur):
        """Pre-take-off run-up frame whose features best match the current frame, so the
        match jumps into the run-up (not mid-air) with a smooth take-off."""
        if len(self.jump_enter) == 0:
            return None
        d = np.linalg.norm(self.feat[self.jump_enter] - self.feat[cur], axis=1)
        f = int(self.jump_enter[d.argmin()])
        return f, self.jump_land_of[f]

    def _qstd(self, traj_block, cur):
        """Standardized query: command trajectory + current frame's pose features."""
        query = np.concatenate([traj_block, self.raw[cur, F.TRAJ_DIM:]])
        return ((query - self.mean) / self.std) * self.w

    def generate(self, command, seconds, start_frame=0, traj_fn=None,
                 jump_at=None, return_trace=False):
        """Roll out for `seconds`; traj_fn(t,xy,yaw)->block overrides the command query.

        Hysteresis: at a search we only jump to the nearest neighbour if it is
        clearly better (by jump_margin) than simply continuing the current clip.
        This keeps the character on long continuous fragments -> less jitter/skating.
        If jump_at is given, at that time the match is forced into the best pre-take-off
        jump run-up and the clip is then ridden through landing (a JUMP on command).
        With return_trace, also return the per-frame library index sequence.
        """
        n = int(seconds * C.FPS)
        cur = start_frame
        dyaw, pivot, offset = -self.yaw[cur], self.xy[cur].copy(), np.zeros(2)  # start at origin, +x
        out, frozen, blend_left, tframe = [], None, 0, []
        locked, did = 0, False
        for step in range(n):
            t = step * C.DT
            world = transform_qpos(self.qpos[cur], dyaw, pivot, offset)[0]
            cwx, cwy = world[0:2].copy(), self.yaw[cur] + dyaw
            if blend_left > 0:
                world = blend_qpos(frozen, world, 1 - blend_left / C.BLEND_FRAMES)
                blend_left -= 1
            out.append(world); tframe.append(cur)

            if jump_at is not None and not did and step >= int(jump_at * C.FPS) and locked == 0:
                je = self.best_jump_entry(cur)              # enter via the `ready` run-up
                if je:
                    entry, land = je
                    dyaw, pivot, offset = alignment_to(self.xy[entry], self.yaw[entry], cwx, cwy)
                    frozen, blend_left = world.copy(), C.BLEND_FRAMES
                    after_end = land + 1 + C.PHASE_TOUCHDOWN + C.PHASE_AFTER   # ready..after, then walk
                    cur, locked, did = entry, min(after_end - entry, n - step), True
                    continue
            if locked > 0:                                  # ride the jump clip through landing
                if not self._is_clip_end(cur):
                    cur += 1
                locked -= 1
                continue

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
        out = np.asarray(out)
        return (out, np.array(tframe)) if return_trace else out

    def walk_path(self, waypoints, box=None, start_frame=0, speed=1.0, reach=1.2,
                  max_seconds=90, return_trace=False):
        """Reactively walk through `waypoints` by aiming the NN trajectory query at the
        current waypoint (go-to-point). If `box` is given, jump over it on each +x crossing
        (entered from the `ready` run-up). No planner -> corners/returns are approximate."""
        from .commands import SpeedCommand
        n = int(max_seconds * C.FPS)
        cur = start_frame
        dyaw, pivot, offset = -self.yaw[cur], self.xy[cur].copy(), np.zeros(2)
        out, frozen, blend_left, tframe = [], None, 0, []
        wp, locked, armed, prev = 0, 0, True, np.zeros(2)
        step = 0
        while step < n and wp < len(waypoints):
            world = transform_qpos(self.qpos[cur], dyaw, pivot, offset)[0]
            cwx, cwy = world[0:2].copy(), self.yaw[cur] + dyaw
            if blend_left > 0:
                world = blend_qpos(frozen, world, 1 - blend_left / C.BLEND_FRAMES)
                blend_left -= 1
            out.append(world); tframe.append(cur)

            if locked > 0:                                   # riding a jump
                if not self._is_clip_end(cur):
                    cur += 1
                locked -= 1
                step += 1
                continue
            if box is not None:                              # jump over the box on a +x crossing
                d = np.asarray(box, float) - cwx
                heading = np.array([np.cos(cwy), np.sin(cwy)])
                aligned = abs(((np.arctan2(d[1], d[0]) - cwy + np.pi) % (2 * np.pi)) - np.pi) < 0.5
                je = self.best_jump_entry(cur)
                fwd = float(self.qpos[self.jump_apex_of[je[0]], 0] - self.qpos[je[0], 0]) if je else 0
                if cwx[0] < 1.0:
                    armed = True
                if je and armed and aligned and 0 < (d @ heading) <= fwd:
                    entry, land = je
                    dyaw, pivot, offset = alignment_to(self.xy[entry], self.yaw[entry], cwx, cwy)
                    frozen, blend_left = world.copy(), C.BLEND_FRAMES
                    cur, locked, armed = entry, (land + 1 + C.PHASE_TOUCHDOWN + C.PHASE_AFTER) - entry, False
                    step += 1
                    continue

            tx, ty = waypoints[wp]
            dvec = np.array([tx, ty]) - cwx
            seg = np.array([tx, ty]) - prev
            passed = seg @ (cwx - np.array([tx, ty])) > 0 if seg @ seg > 1e-6 else False
            if np.linalg.norm(dvec) < reach or passed:
                prev = np.array([tx, ty]); wp += 1
                continue

            if step > 0 and (step % C.MM_SEARCH_INTERVAL == 0 or self._is_clip_end(cur)):
                block = SpeedCommand([(0., speed, float(np.arctan2(dvec[1], dvec[0])))]).trajectory(cwx, cwy, 0.)
                qstd = self._qstd(block, cur)
                dist_best, vi = self.tree.query(qstd)
                best = int(self.valid[vi])
                end = self._is_clip_end(cur)
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
            step += 1
        out = np.asarray(out)
        return (out, np.array(tframe)) if return_trace else out
