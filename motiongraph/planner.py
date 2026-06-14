"""Shared A* planner for both MotionGraph and MotionMatcher.

The cost model, goal-distance heuristic, closed set and reconstruction live here so the
two controllers plan *identically* over their different transition sets. A controller `s`
only has to expose:
  s.qpos, s.xy, s.yaw                                   -- library arrays
  s._options(cur, xy, yaw, align) -> [(start, align, is_transition, penalty), ...]
  s._play(frame, count, align)    -> (world_seq, end_xy, end_yaw, last_frame)
  s._pose_dist(a, b)              -> float
MG's transitions come from its precomputed edges; MM's come from its feature KD-tree.
"""
import heapq
import numpy as np

from . import config as C
from .kinematics import transform_qpos, alignment_to, blend_qpos, ease_to_terminal


def angdiff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def astar_plan(s, command, seconds, start_frame, target_xy, target_yaw, term_frame,
               K=None, max_expansions=20000, w_pos=1.5, w_yaw=0.4, w_pose=0.15, cmd_w=0.4,
               turn_w=0.6, face_w=0.5, cruise=1.0, reach=0.7, ease="full"):
    """A* for a least-cost edge sequence that arrives at the target pose.

    Best-first over a priority queue ordered by f = g + h:
      g = Σ ( cmd_w·‖segment_vel − go-to-target_vel‖ + transition penalty
            + turn_w·|Δheading| + face_w·|heading − travel direction| ), so wandering is
          expensive, progress toward the target is cheap, and the body must face the way it
          walks -- the velocity term alone is the straight-line chord and is blind to a
          fragment spinning in place while drifting forward, which an optimizer like A* will
          otherwise exploit (moonwalking / pirouettes);
      h = w_pos·‖xy − target‖  -- a goal-distance heuristic that pulls the frontier toward
          the target (goal-directed / weighted A*; a zero heuristic degenerates to Dijkstra
          and, given the graph's branching, never arrives on this huge state space).
    A node within `reach` of the target is a GOAL; its g absorbs the goal cost
    w_pos·‖xy−target‖ + w_yaw·|Δyaw| + w_pose·pose-distance, and the first goal popped wins.
    A discretized closed set (frame, world xy/yaw) collapses revisits; `seconds`·1.5 caps the
    horizon and the expansion budget guarantees termination.
    """
    K = K or C.SEARCH_INTERVAL
    Nmax = int(seconds * C.FPS * 1.5)                   # horizon cap (target should arrive first)
    target_xy = np.asarray(target_xy, float)

    # All xy/yaw below are in the PLAN-LOCAL frame: a0 places start_frame at the origin
    # facing +x, so target_xy/target_yaw are given relative to that. The caller re-anchors
    # the returned motion into the world with alignment_to (see run_experiments._plan_walk).
    def want_vel(xy):                                   # go-to-point: cruise toward the target
        d = target_xy - xy
        n = np.linalg.norm(d)
        return (cruise * d / n) if n > 1e-6 else np.zeros(2)

    def goal_cost(xy, yaw, frame):
        return (w_pos * float(np.linalg.norm(xy - target_xy))
                + w_yaw * abs(angdiff(yaw, target_yaw)) + w_pose * s._pose_dist(frame, term_frame))

    # node = [cur, align, xy, yaw, t, g, parent, start_frame, is_transition]; xy,yaw plan-local
    a0 = (-s.yaw[start_frame], s.xy[start_frame].copy(), np.zeros(2))
    x0 = transform_qpos(s.qpos[start_frame], *a0)[0]
    nodes = [[start_frame, a0, x0[:2], s.yaw[start_frame] + a0[0], 0, 0.0, -1, start_frame, False]]
    pq = [(w_pos * float(np.linalg.norm(nodes[0][2] - target_xy)), 0)]   # (f = g + h, node id)
    seen, best, best_g, used = set(), 0, 1e9, 0         # best=0 (start) is the always-valid fallback

    while pq and used < max_expansions:
        _, nid = heapq.heappop(pq)
        cur, align, xy, yaw, t, g, _, _, _ = nodes[nid]
        if float(np.linalg.norm(xy - target_xy)) < reach or t >= Nmax:
            best = nid                                  # first goal popped == lowest f
            break
        key = (cur, round(float(xy[0]), 1), round(float(xy[1]), 1), round(float(yaw), 1))
        if key in seen:
            continue
        seen.add(key); used += 1
        for start, al, is_tr, pen in s._options(cur, xy, yaw, align):
            world, exy, eyaw, last = s._play(start, K, al)    # exy,eyaw = segment end pose (plan-local)
            te = t + len(world)
            avgv = (exy - world[0, :2]) / (max(len(world), 1) * C.DT)   # segment planar velocity
            ng = g + cmd_w * float(np.linalg.norm(avgv - want_vel(world[0, :2]))) + pen
            ng += turn_w * abs(angdiff(eyaw, yaw))                 # discourage spinning
            if float(np.linalg.norm(avgv)) > 0.3:                 # face the way you travel
                ng += face_w * abs(angdiff(eyaw, float(np.arctan2(avgv[1], avgv[0]))))
            if float(np.linalg.norm(exy - target_xy)) < reach:    # absorb goal cost at arrival
                ng += goal_cost(exy, eyaw, last)
                if ng < best_g:
                    best_g, best = ng, len(nodes)
            nodes.append([last, al, exy, eyaw, te, ng, nid, start, is_tr])
            heapq.heappush(pq, (ng + w_pos * float(np.linalg.norm(exy - target_xy)), len(nodes) - 1))

    return _reconstruct(s, best, nodes, K, target_xy, target_yaw, term_frame, ease)


def _reconstruct(s, leaf, nodes, K, target_xy, target_yaw, term_frame, ease="full"):
    """Replay the winning node chain with cross-fades at transitions, then ease onto the
    terminal pose (mode `ease`; "pose" eases joints only -> no root-drag skating)."""
    chain, nid = [], leaf
    while nid > 0:
        chain.append(nid)
        nid = nodes[nid][6]
    chain.reverse()

    out, frozen, blend_left = [], None, 0
    for nid in chain:
        _, align, _, _, _, _, _, start, is_tr = nodes[nid]
        world, _, _, _ = s._play(start, K, align)
        if is_tr:
            frozen, blend_left = (out[-1] if out else world[0]).copy(), C.BLEND_FRAMES
        for w in world:
            if blend_left > 0:
                w = blend_qpos(frozen, w, 1 - blend_left / C.BLEND_FRAMES)
                blend_left -= 1
            out.append(w)
    out = np.asarray(out)

    dy, pv, of = alignment_to(s.xy[term_frame], s.yaw[term_frame], target_xy, target_yaw)
    term = transform_qpos(s.qpos[term_frame], dy, pv, of)[0]
    return ease_to_terminal(out, term, int(0.7 * C.FPS), mode=ease)
