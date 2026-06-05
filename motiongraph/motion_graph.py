"""Motion graph (Kovar-style) for the G1.

Build: a per-frame pose+velocity descriptor; transition edges connect frames whose
descriptors are close (a good blend point). Every frame also has a continuation edge
to its successor. Walking the graph yields novel motion built from clip fragments.

  task1: greedy command following -- at decision points pick the edge whose motion
         best matches the commanded velocity.
  task2: beam search planning -- find an edge sequence that arrives at a terminal
         pose/position at a fixed time (motion in-betweening).
"""
import os
import pickle
import numpy as np
from scipy.spatial import cKDTree

from . import config as C
from .kinematics import transform_qpos, alignment_to, blend_qpos, ease_to_terminal


def _descriptors(lib):
    """Per-frame [joints, joint_vel, z, root_local_vel(2), yaw_rate] for transitions."""
    qpos, yaw = lib["qpos"], lib["yaw"]
    q = qpos[:, C.JOINTS]
    N = len(qpos)
    qd = np.zeros_like(q)
    rv = np.zeros((N, 2), np.float32)
    yr = np.zeros(N, np.float32)
    for cid in np.unique(lib["clip_id"]):
        idx = np.where(lib["clip_id"] == cid)[0]
        a, b = idx[0], idx[-1]
        nxt = np.minimum(idx + 1, b)
        qd[idx] = (q[nxt] - q[idx]) / C.DT
        c, s = np.cos(-yaw[idx]), np.sin(-yaw[idx])
        d = (qpos[nxt, 0:2] - qpos[idx, 0:2]) / C.DT
        rv[idx, 0] = c * d[:, 0] - s * d[:, 1]
        rv[idx, 1] = s * d[:, 0] + c * d[:, 1]
        yr[idx] = ((yaw[nxt] - yaw[idx] + np.pi) % (2 * np.pi) - np.pi) / C.DT
    desc = np.concatenate([q, 0.15 * qd, qpos[:, 2:3], rv, yr[:, None]], 1).astype(np.float32)
    return (desc - desc.mean(0)) / (desc.std(0) + 1e-6)


class MotionGraph:
    def __init__(self, lib, n_neighbors=16, src_stride=2, tgt_stride=2, pca_dim=16,
                 tau_factor=2.5, min_z=0.6, cache=True):
        self.lib = lib
        self.qpos, self.yaw = lib["qpos"], lib["yaw"]
        self.xy = lib["qpos"][:, 0:2]
        self.fic, self.lengths, self.clip_id = lib["frame_in_clip"], lib["lengths"], lib["clip_id"]
        self.skill = lib["skill"] if "skill" in lib else np.zeros(len(self.qpos), np.int32)

        # PCA-reduce the 62-d descriptor (always available, also used for skill matching).
        desc = _descriptors(lib)
        X = desc - desc.mean(0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        self.P = (X @ Vt[:pca_dim].T).astype(np.float32)

        cpath = os.path.join(C.ROOT, "data", f"motion_graph_{len(self.qpos)}.pkl")
        key = (len(self.qpos), n_neighbors, src_stride, tgt_stride, pca_dim, tau_factor, min_z)
        self.trans = None
        if cache and os.path.exists(cpath):
            with open(cpath, "rb") as f:
                blob = pickle.load(f)
            if blob.get("key") == key:
                self.trans, self._src = blob["trans"], blob["src"]
                print(f"MotionGraph: loaded {len(self.trans)} nodes from cache")
        if self.trans is None:
            self._build_transitions(n_neighbors, src_stride, tgt_stride, tau_factor, min_z)
            if cache:
                with open(cpath, "wb") as f:
                    pickle.dump({"key": key, "trans": self.trans, "src": self._src}, f)
        self._build_skill_index()

    def _build_transitions(self, n_neighbors, src_stride, tgt_stride, tau_factor, min_z):
        P = self.P
        upright = self.qpos[:, 2] >= min_z
        # normal transitions only target WALK frames; a jump is entered only via
        # best_jump_entry (its pre-take-off run-up), never mid-air.
        valid = np.where(upright & ~self._clip_end_mask() & (self.skill == 0))[0][::tgt_stride]
        tree = cKDTree(P[valid])
        src = np.array([i for i in range(0, len(self.qpos), src_stride) if upright[i]])
        D, NN = tree.query(P[src], k=n_neighbors + 1)                         # batched, low-dim
        tau = float(np.median(D[:, 1])) * tau_factor                          # adaptive blend threshold
        self.trans = {}
        for row, i in enumerate(src):
            edges = []
            for dist, k in zip(D[row], NN[row]):
                if dist > tau or k >= len(valid):
                    continue
                j = int(valid[k])
                if self.clip_id[j] == self.clip_id[i] and abs(j - i) < 30:
                    continue                                                 # skip near-self
                edges.append((j, float(dist)))
            if edges:
                self.trans[int(i)] = edges[:n_neighbors]
        self._src = np.array(sorted(self.trans))                             # frames that own edges
        print(f"MotionGraph: {len(self.trans)} transition nodes, "
              f"{sum(len(v) for v in self.trans.values())} edges (tau={tau:.2f})")

    # --- skill awareness -------------------------------------------------
    def _build_skill_index(self):
        """Tag edges by (skill_from, skill_to); index jump entries for triggering."""
        self.skill_edges = {}                            # (sf, st) -> [(i, j, dist), ...]
        for i, es in self.trans.items():
            for j, d in es:
                self.skill_edges.setdefault((int(self.skill[i]), int(self.skill[j])), []).append((i, j, d))
        for k in self.skill_edges:
            self.skill_edges[k].sort(key=lambda e: e[2])
        # pre-take-off run-up frames -- the only valid places to enter a jump
        from .jumps import jump_entries
        self.jump_enter, self.jump_land_of = jump_entries(self.lib)

    def top_skill_edges(self, k=5):
        """Top-k (best-blend) transition edges for each skill pair, e.g. walk->jump."""
        return {C.SKILLS[a] + "->" + C.SKILLS[b]: v[:k] for (a, b), v in self.skill_edges.items()}

    def best_jump_entry(self, cur):
        """Enterable jump run-up frame whose pose best matches `cur` (-> smooth take-off).
        Returns (entry_frame, land_frame) or None if no jumps in the library."""
        if len(self.jump_enter) == 0:
            return None
        d = np.linalg.norm(self.P[self.jump_enter] - self.P[cur], axis=1)
        f = int(self.jump_enter[d.argmin()])
        return f, self.jump_land_of[f]

    def _clip_end_mask(self):
        return self.fic >= self.lengths[self.clip_id] - 1

    def _is_end(self, i):
        return self.fic[i] >= self.lengths[self.clip_id[i]] - 1

    def _nearest_src(self, frame):
        """Closest frame <= search horizon that owns transition edges."""
        k = self._src.searchsorted(frame)
        for cand in (self._src[k] if k < len(self._src) else -1,
                     self._src[k - 1] if k > 0 else -1):
            if cand >= 0 and self.clip_id[cand] == self.clip_id[frame] and abs(cand - frame) <= C.SEARCH_INTERVAL:
                return int(cand)
        return -1

    # --- playback helper -------------------------------------------------
    def _play(self, frame, count, align):
        dyaw, pivot, offset = align
        end = min(frame + count, frame + (self.lengths[self.clip_id[frame]] - 1 - self.fic[frame]) + 1)
        idx = np.arange(frame, max(end, frame + 1))
        world = transform_qpos(self.qpos[idx], dyaw, pivot, offset)
        last = idx[-1]
        return world, world[-1, 0:2].copy(), self.yaw[last] + dyaw, int(last)

    def _seg_vel_local(self, frame, H=15):
        """Average planar velocity over the next H frames, in the frame's own local
        space (intrinsic motion direction, independent of world placement)."""
        b = frame + (self.lengths[self.clip_id[frame]] - 1 - self.fic[frame])
        j = min(frame + H, b)
        if j <= frame:
            return np.zeros(2)
        v = (self.xy[j] - self.xy[frame]) / ((j - frame) * C.DT)
        c, s = np.cos(-self.yaw[frame]), np.sin(-self.yaw[frame])
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

    # --- task1: greedy command following --------------------------------
    def follow_command(self, command, seconds, start_frame=0, return_trace=False):
        """Greedy command following. With return_trace, also return per-frame
        (library_frame, took_transition) so the graph traversal can be visualized."""
        n = int(seconds * C.FPS)
        cur = start_frame
        align = (-self.yaw[cur], self.xy[cur].copy(), np.zeros(2))   # start at origin, +x
        out, frozen, blend_left, step = [], None, 0, 0
        trace_frame, trace_jump = [], []
        while step < n:
            world = transform_qpos(self.qpos[cur], *align)[0]
            cwx, cwy = world[0:2].copy(), self.yaw[cur] + align[0]
            if blend_left > 0:
                world = blend_qpos(frozen, world, 1 - blend_left / C.BLEND_FRAMES)
                blend_left -= 1
            out.append(world)
            trace_frame.append(cur)
            trace_jump.append(False)

            decide = (step % C.SEARCH_INTERVAL == 0 and step > 0) or self._is_end(cur)
            if decide:
                best, best_tr = self._greedy_choose(cur, cwy, command.desired_velocity(step * C.DT))
                if best_tr:
                    align = alignment_to(self.xy[best], self.yaw[best], cwx, cwy)
                    frozen, blend_left = world.copy(), C.BLEND_FRAMES
                    trace_jump[-1] = True                       # this frame took a transition edge
                cur = best
            elif not self._is_end(cur):
                cur += 1
            step += 1
        out = np.asarray(out)
        if return_trace:
            return out, np.array(trace_frame), np.array(trace_jump)
        return out

    def _greedy_choose(self, cur, cwy, want):
        """Pick {continue} ∪ {transitions} whose intrinsic local velocity best matches
        the desired world velocity `want`. Returns (best_frame, took_transition)."""
        c, s = np.cos(-cwy), np.sin(-cwy)
        wl = np.array([c * want[0] - s * want[1], s * want[0] + c * want[1]])   # local desired vel
        src = cur if cur in self.trans else self._nearest_src(cur)
        cand = []
        if not self._is_end(cur):
            cand.append((cur + 1, 0.0, False))
        if src >= 0:
            for j, dist in self.trans[src]:
                cand.append((j, 0.1 + 0.05 * dist, True))       # switch + blend penalty
        if not cand:                                            # dead end: jump to nearest source
            g = int(self._src[np.argmin(np.abs(self._src - cur))])
            cand = [(self.trans[g][0][0], 0.0, True)]
        best, best_cost, best_tr = None, 1e9, False
        for f, pen, is_tr in cand:
            cost = np.linalg.norm(self._seg_vel_local(f) - wl) + pen
            if cost < best_cost:
                best, best_cost, best_tr = f, cost, is_tr
        return best, best_tr

    # --- jump skill: walk, then perform a jump on a trigger ----------------
    def follow_with_jump(self, command, seconds, start_frame=0, jump_at=4.0,
                         land_pad=35, straighten=0.6, return_trace=False):
        """Walk forward; at t=`jump_at` s transition to the best-matching jump run-up
        and play the jump through landing, then resume walking. The walk heading is
        kept near the start heading (+x) by `straighten` so the path stays roughly
        straight while the jump clip carries the take-off, flight and forward travel."""
        n = int(seconds * C.FPS)
        cur = start_frame
        align = (-self.yaw[cur], self.xy[cur].copy(), np.zeros(2))
        ref = 0.0                                                # reference (start) heading = +x
        out, frozen, blend_left, step, locked, did = [], None, 0, 0, 0, False
        tframe, tphase = [], []                                  # phase: 0 walk, 1 jump
        while step < n:
            world = transform_qpos(self.qpos[cur], *align)[0]
            cwx, cwy = world[0:2].copy(), self.yaw[cur] + align[0]
            if blend_left > 0:
                world = blend_qpos(frozen, world, 1 - blend_left / C.BLEND_FRAMES)
                blend_left -= 1
            out.append(world); tframe.append(cur); tphase.append(int(self.skill[cur]))

            if not did and step >= int(jump_at * C.FPS) and locked == 0:
                je = self.best_jump_entry(cur)                  # transition into the jump run-up
                if je:
                    entry, land = je
                    align = alignment_to(self.xy[entry], self.yaw[entry], cwx, cwy)
                    frozen, blend_left = world.copy(), C.BLEND_FRAMES
                    # ride the (straight) walk->jump->walk clip from the run-up to its end,
                    # so the post-landing walk stays straight instead of returning to the
                    # curvier base clip; cap at the remaining demo length.
                    to_end = self.lengths[self.clip_id[entry]] - 1 - self.fic[entry]
                    cur, locked, did = entry, min(to_end, n - step), True
                    step += 1
                    continue
            if locked > 0:                                       # play the jump straight through
                if not self._is_end(cur):
                    cur += 1
                locked -= 1
            else:
                decide = (step % C.SEARCH_INTERVAL == 0 and step > 0) or self._is_end(cur)
                if decide:
                    spd = command.state(step * C.DT)[0]             # forward, heading nudged
                    h = cwy + straighten * ((ref - cwy + np.pi) % (2 * np.pi) - np.pi)  # toward +x
                    want = spd * np.array([np.cos(h), np.sin(h)])
                    best, best_tr = self._greedy_choose(cur, cwy, want)
                    if best_tr:
                        align = alignment_to(self.xy[best], self.yaw[best], cwx, cwy)
                        frozen, blend_left = world.copy(), C.BLEND_FRAMES
                    cur = best
                elif not self._is_end(cur):
                    cur += 1
            step += 1
        out = np.asarray(out)
        if return_trace:
            return out, np.array(tframe), np.array(tphase)
        return out

    # --- task2: beam-search planning to a terminal state (in-betweening) -----
    def _pose_dist(self, a, b):
        return float(np.linalg.norm(self.qpos[a, C.JOINTS] - self.qpos[b, C.JOINTS]))

    def _options(self, cur, xy, yaw, align):
        """Candidate macro-moves from a state: (start_frame, align, is_jump, penalty)."""
        opts = []
        if not self._is_end(cur):
            opts.append((cur + 1, align, False, 0.0))
        src = cur if cur in self.trans else self._nearest_src(cur)
        if src >= 0:
            for j, d in self.trans[src]:
                opts.append((j, alignment_to(self.xy[j], self.yaw[j], xy, yaw), True, 0.3 + 0.05 * d))
        return opts

    def plan_to(self, command, seconds, start_frame, target_xy, target_yaw, term_frame,
                K=None, beam=64, w_pos=1.5, w_yaw=0.4, w_pose=0.15, cmd_w=0.4,
                max_speed=3.0, tail=2.5):
        """Beam search for an edge sequence that reaches the terminal at time `seconds`.

        Desired velocity follows the speed command while cruising, then switches to
        reach-target pacing (toward the target at remaining_distance / remaining_time)
        for the final `tail` seconds, so the path actually arrives. Cost = velocity
        tracking + transition penalties; goal cost adds endpoint pose/position error.
        """
        K = K or C.SEARCH_INTERVAL
        N = int(seconds * C.FPS)
        target_xy = np.asarray(target_xy, float)

        def desired_vel(xy_end, t_end):
            tsec = t_end * C.DT
            if tsec < seconds - tail:
                return command.desired_velocity(tsec)
            d = target_xy - xy_end                          # steer onto the target
            rem_t = max(C.DT, seconds - tsec)
            n = np.linalg.norm(d)
            return (min(max_speed, n / rem_t) * d / n) if n > 1e-6 else np.zeros(2)
        # node = [cur, align, xy, yaw, t, cost, parent, start_frame, is_jump]
        a0 = (-self.yaw[start_frame], self.xy[start_frame].copy(), np.zeros(2))
        x0 = transform_qpos(self.qpos[start_frame], *a0)[0]
        nodes = [[start_frame, a0, x0[:2], self.yaw[start_frame] + a0[0], 0, 0.0, -1, start_frame, False]]
        frontier, best, best_cost = [0], None, 1e9

        while frontier:
            nxt = []
            for nid in frontier:
                cur, align, xy, yaw, t, cost, _, _, _ = nodes[nid]
                if t >= N:
                    gc = (cost + w_pos * np.linalg.norm(xy - target_xy)
                          + w_yaw * abs(_angdiff(yaw, target_yaw)) + w_pose * self._pose_dist(cur, term_frame))
                    if gc < best_cost:
                        best_cost, best = gc, nid
                    continue
                for start, al, jump, pen in self._options(cur, xy, yaw, align):
                    world, exy, eyaw, last = self._play(start, K, al)
                    avgv = (exy - world[0, :2]) / (max(len(world), 1) * C.DT)
                    want = desired_vel(exy, t + len(world))
                    sc = cost + cmd_w * np.linalg.norm(avgv - want) + pen
                    nodes.append([last, al, exy, eyaw, t + len(world), sc, nid, start, jump])
                    nxt.append(len(nodes) - 1)
            # prune by cost + admissible distance-to-go heuristic
            nxt.sort(key=lambda n: nodes[n][5] + w_pos * max(
                0.0, np.linalg.norm(nodes[n][2] - target_xy) - max_speed * (N - nodes[n][4]) * C.DT))
            frontier = nxt[:beam]

        return self._reconstruct(best, nodes, K, target_xy, target_yaw, term_frame)

    def _reconstruct(self, leaf, nodes, K, target_xy, target_yaw, term_frame):
        """Replay the winning node chain with blends, then ease onto the terminal."""
        chain = []
        nid = leaf
        while nid > 0:
            chain.append(nid)
            nid = nodes[nid][6]
        chain.reverse()

        out, frozen, blend_left = [], None, 0
        for nid in chain:
            _, align, _, _, _, _, _, start, jump = nodes[nid]
            world, _, _, _ = self._play(start, K, align)
            if jump:
                frozen, blend_left = (out[-1] if out else world[0]).copy(), C.BLEND_FRAMES
            for w in world:
                if blend_left > 0:
                    w = blend_qpos(frozen, w, 1 - blend_left / C.BLEND_FRAMES)
                    blend_left -= 1
                out.append(w)
        out = np.asarray(out)

        # terminal pose placed at the target world pose, then ease the tail onto it
        dy, pv, of = alignment_to(self.xy[term_frame], self.yaw[term_frame], target_xy, target_yaw)
        term = transform_qpos(self.qpos[term_frame], dy, pv, of)[0]
        return ease_to_terminal(out, term, int(0.7 * C.FPS))


def _angdiff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi
