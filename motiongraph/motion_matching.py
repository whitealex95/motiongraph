"""GenoView (Holden "Simple Motion Matching") motion matching for the G1.

A faithful port of ~/Projects/motionmatching-g1 (which in turn ports GenoView's
genoview_g1.py): a per-clip Savitzky-Golay-smoothed "simulation root" (ground position +
facing) is what the matcher tracks and integrates; the pelvis is a local offset of it.
Every SEARCH_TIME seconds we nearest-neighbour search per-clip KD-trees (biased toward
staying put, each clip's last HORIZONS[-1] frames trimmed from the search), inertialize
joints + pelvis-local pos/rot toward the winner, integrate the matched clip's smooth root
velocity through the world, then place the pelvis back on that root. Transitions are
inertialized (decaying spring offsets), NOT cross-faded. The L/R-mirrored database and the
triggered jump skill (jump frames kept out of the search; entered only via a run-up) are
the only extras over genoview_g1.py.

Interface: real-time `step(speed, heading) -> qpos(36)`; `generate(command, seconds, ...)`
wraps it to roll out an offline sequence driven by a SpeedCommand (for rendering).
"""
import numpy as np
from scipy.spatial import cKDTree

from . import config as C
from . import quat
from .mm_features import build_db, yaw_quat, FORWARD, HORIZONS, FPS
from .jumps import jump_entries
from .springs import (DecaySpringDamperPosition, DecaySpringDamperRotation,
                      TrajectorySpringPosition, TrajectorySpringRotation)

DT = C.DT
NDOF = 29


class MotionMatcher:
    def __init__(self, lib, start_frame=None, **_ignored):   # _ignored: legacy traj_w/pose_w
        self.lib = lib
        db = self.db = build_db(lib)
        self.starts, self.stops = db["starts"], db["stops"]
        self.X = db["X"]
        self.dof, self.dofVel = db["dof"], db["dofVel"]
        self.simPosDB, self.simThetaDB = db["simPos"], db["simTheta"]
        self.simVelDB, self.yawRateDB = db["simVel"], db["yawRate"]
        self.plpDB, self.plvDB = db["pelvLocalPos"], db["pelvLocalVel"]
        self.prDB, self.paDB = db["pelvLocalRot"], db["pelvLocalAng"]
        self.clip_id = lib["clip_id"]
        self.skill = lib["skill"] if "skill" in lib else np.zeros(len(self.X), np.int32)
        self.Ttimes = HORIZONS / FPS

        # Per-clip KD-trees over LOCOMOTION clips only (skill==0), each trimming the last
        # HORIZONS[-1] frames so a full future trajectory always exists; jump clips excluded
        # so locomotion never matches into a jump (it only happens on a trigger).
        self.search = []        # (clip_index, tree, range_start)
        searchable = []
        for ci, (rs, re) in enumerate(zip(self.starts, self.stops)):
            if self.skill[rs:re].any() or re - rs <= HORIZONS[-1]:
                continue
            self.search.append((ci, cKDTree(self.X[rs:re - HORIZONS[-1]]), rs))
            searchable.append(re - rs - HORIZONS[-1])
        self.valid = np.empty(int(np.sum(searchable)), int)   # count of searchable frames

        # Pre-take-off run-up frames available to the jump trigger (continuing jumps only).
        self.jump_enter, self.jump_land_of, self.jump_apex_of = jump_entries(lib)
        self.qpos = lib["qpos"]                  # for the reactive box-jump reach (entry->apex)
        self.reset(start_frame)

    # --- state ---------------------------------------------------------------
    def reset(self, start_frame=None):
        if start_frame is None:
            start_frame = min(self.stops[0] - 1, self.starts[0] + 30)
        self.animRange = int(np.searchsorted(self.starts, start_frame, "right") - 1)
        self.animFrame = int(start_frame)
        self.rootPos = self.simPosDB[self.animFrame].copy()    # controller root = smoothed sim root
        self.rootVel = np.zeros(3); self.rootAcc = np.zeros(3); self.rootAng = np.zeros(3)
        self.rootYaw = float(self.simThetaDB[self.animFrame])
        self.rootRot = yaw_quat(self.rootYaw)
        self.desiredDir = quat.mul_vec(self.rootRot, FORWARD)
        self.offDof = np.zeros(NDOF); self.offDofVel = np.zeros(NDOF)   # inertialization offsets
        self.offPP = np.zeros(3); self.offPPVel = np.zeros(3)
        self.offPR = np.array([1.0, 0.0, 0.0, 0.0]); self.offPAng = np.zeros(3)
        self.searchTimer = 0.0
        self.jump_pending = False
        self.jump_locked = 0
        self.Tpos = np.tile(self.rootPos, (len(HORIZONS), 1))
        self.Tdir = np.tile(self.desiredDir, (len(HORIZONS), 1))
        self.gizmo_trace = []    # per-step (Tpos, Tdir) command gizmo, for offline rendering

    # --- jump skill ----------------------------------------------------------
    def trigger_jump(self):
        """Request a jump. Honoured on the next step if not already jumping."""
        if self.jump_locked == 0:
            self.jump_pending = True

    @property
    def jumping(self):
        return self.jump_locked > 0

    @property
    def cur(self):
        return self.animFrame

    def best_jump_entry(self, cur=None):
        """Pre-take-off `ready` run-up frame whose features best match the current frame, so
        the jump is entered from the run-up (not mid-air) with a smooth take-off."""
        cur = self.animFrame if cur is None else cur
        if len(self.jump_enter) == 0:
            return None
        d = np.linalg.norm(self.X[self.jump_enter] - self.X[cur], axis=1)
        f = int(self.jump_enter[d.argmin()])
        return f, self.jump_land_of[f]

    def _inertialize_into(self, b, rng):
        """Capture the pose discontinuity from frame a to b (joints, pelvis-local pos+rot) as
        decaying inertialization offsets, then switch the playhead there (no pop)."""
        a = self.animFrame
        self.offDof = (self.offDof + self.dof[a]) - self.dof[b]
        self.offDofVel = (self.offDofVel + self.dofVel[a]) - self.dofVel[b]
        self.offPP = (self.offPP + self.plpDB[a]) - self.plpDB[b]
        self.offPPVel = (self.offPPVel + self.plvDB[a]) - self.plvDB[b]
        self.offPR = quat.abs(quat.mul_inv(quat.mul(self.offPR, self.prDB[a]), self.prDB[b]))
        self.offPAng = (self.offPAng + self.paDB[a]) - self.paDB[b]
        self.animRange, self.animFrame = rng, b

    # --- one frame -----------------------------------------------------------
    def step(self, speed, heading):
        """Advance one frame. speed [m/s], heading [rad] are the desired locomotion this
        frame; returns the world-space qpos (36,)."""
        starts, stops, X = self.starts, self.stops, self.X

        desiredVel = speed * np.array([np.cos(heading), np.sin(heading), 0.0])
        if speed > 0.01:
            self.desiredDir = np.array([np.cos(heading), np.sin(heading), 0.0])
        desiredRot = yaw_quat(np.arctan2(self.desiredDir[1], self.desiredDir[0]))

        # ---- Predict desired trajectory (critically-damped springs) ----
        dt_col = self.Ttimes[:, None]
        self.Tpos, _, _ = TrajectorySpringPosition(
            self.rootPos, self.rootVel, self.rootAcc, desiredVel, C.VEL_HALFLIFE, dt_col)
        Trot, _ = TrajectorySpringRotation(
            self.rootRot, self.rootAng, desiredRot, C.ROT_HALFLIFE, dt_col)
        self.Tdir = quat.mul_vec(Trot, FORWARD)

        # ---- Jump trigger: inertialize into the best `ready` run-up, then lock ----
        if self.jump_pending and self.jump_locked == 0:
            self.jump_pending = False
            je = self.best_jump_entry()
            if je is not None:
                entry, land = je
                rng = int(np.searchsorted(starts, entry, "right") - 1)
                self._inertialize_into(entry, rng)
                after_end = min(land + 1 + C.PHASE_TOUCHDOWN + C.PHASE_AFTER, stops[rng] - 1)
                self.jump_locked = max(1, after_end - entry)
                self.searchTimer = C.SEARCH_TIME

        # ---- Search (skipped while riding a jump) ----
        if self.jump_locked == 0 and self.searchTimer <= 0.0:
            qh_ctrl = yaw_quat(self.rootYaw)
            Xq = self._runtime_features(qh_ctrl)
            bestRange, bestFrame = self.animRange, self.animFrame
            if bestFrame < stops[bestRange] - HORIZONS[-1]:
                best = float(np.linalg.norm(Xq - X[bestFrame]) - C.CURRENT_BIAS)   # stay-in-clip bias
            else:
                best = np.inf
            for ci, tree, rs in self.search:
                dist, k = tree.query(Xq, eps=C.APPROX_BIAS, distance_upper_bound=best)
                if dist < best:
                    best, bestRange, bestFrame = dist, ci, int(rs + k)
            if bestRange != self.animRange or bestFrame != self.animFrame:
                self._inertialize_into(bestFrame, bestRange)   # seamless inertialized cut
            self.searchTimer = C.SEARCH_TIME

        # ---- Advance the playhead (30 fps data) ----
        self.animFrame = int(np.clip(self.animFrame + 1,
                                     starts[self.animRange], stops[self.animRange] - 1))
        self.searchTimer -= DT
        if self.jump_locked > 0:
            self.jump_locked -= 1
            if self.jump_locked == 0:
                self.searchTimer = 0.0                         # search out of the jump at once
        elif self.animFrame >= stops[self.animRange] - 2:
            self.searchTimer = 0.0
        f = self.animFrame

        # ---- Integrate controller root from the matched clip's smooth root velocity ----
        _, _, self.rootAcc = TrajectorySpringPosition(
            self.rootPos, self.rootVel, self.rootAcc, desiredVel, C.ROT_HALFLIFE, DT)
        qh_clip = yaw_quat(self.simThetaDB[f])
        clipVelLocal = quat.inv_mul_vec(qh_clip, self.simVelDB[f])
        self.rootVel = quat.mul_vec(self.rootRot, clipVelLocal)
        self.rootAng = np.array([0.0, 0.0, self.yawRateDB[f]])
        self.rootPos = self.rootPos + self.rootVel * DT
        self.rootYaw = self.rootYaw + self.yawRateDB[f] * DT
        self.rootRot = yaw_quat(self.rootYaw)

        # ---- Inertialize joints + pelvis-local offset, then reconstruct the pose ----
        self.offDof, self.offDofVel = DecaySpringDamperPosition(
            self.offDof, self.offDofVel, C.INERT_HALFLIFE, DT)
        self.offPP, self.offPPVel = DecaySpringDamperPosition(
            self.offPP, self.offPPVel, C.INERT_HALFLIFE, DT)
        self.offPR, self.offPAng = DecaySpringDamperRotation(
            self.offPR, self.offPAng, C.INERT_HALFLIFE, DT)

        dofOut = self.dof[f] + self.offDof
        pelvLocalPos = self.plpDB[f] + self.offPP
        pelvLocalRot = quat.mul(self.offPR, self.prDB[f])
        pelvWorldPos = self.rootPos + quat.mul_vec(self.rootRot, pelvLocalPos)
        pelvWorldRot = quat.mul(self.rootRot, pelvLocalRot)

        qpos = np.empty(36)
        qpos[0:3] = pelvWorldPos
        qpos[3:7] = pelvWorldRot
        qpos[7:] = dofOut

        # Record the spring-predicted command trajectory (GenoView's DrawTrajectory gizmo)
        # so the offline renderer can draw it, just like the real-time viewer.
        self.gizmo_trace.append((self.Tpos.copy(), self.Tdir.copy()))
        return qpos

    def _runtime_features(self, qh_ctrl):
        """Query = current frame's pose blocks (from X) + the desired trajectory, normalized
        the same way as the database (genoview runtime_features)."""
        Xoffset, Xscale = self.db["Xoffset"], self.db["Xscale"]
        pose = self.X[self.animFrame, 0:15] * Xscale[0:15] + Xoffset[0:15]   # de-normalized pose
        trajPos = quat.inv_mul_vec(qh_ctrl, self.Tpos - self.rootPos)[:, 0:2].ravel()
        trajDir = quat.inv_mul_vec(qh_ctrl, self.Tdir)[:, 0:2].ravel()
        q = np.concatenate([pose, trajPos, trajDir])
        return (q - Xoffset) / Xscale

    # --- offline roll-out (drives step() from a SpeedCommand, for rendering) ---
    def generate(self, command, seconds, start_frame=None, jump_at=None, return_trace=False):
        """Roll out `seconds` of motion: each frame feed the command's (speed, heading) to
        step(). If jump_at is given, trigger one jump at that time. Returns world qpos (T,36)
        (and the per-frame library index if return_trace)."""
        self.reset(start_frame)
        n = int(seconds * C.FPS)
        out, tframe = [], []
        for s in range(n):
            t = s * C.DT
            if jump_at is not None and t >= jump_at and not self.jumping:
                self.trigger_jump(); jump_at = None
            spd, hd = command.state(t)
            out.append(self.step(spd, hd)); tframe.append(self.cur)
        out = np.asarray(out)
        return (out, np.array(tframe)) if return_trace else out
