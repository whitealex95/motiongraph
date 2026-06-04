"""Build / load the unified G1 motion library from the LAFAN1 CSVs."""
import os
import glob
import numpy as np

from . import config as C
from .g1_model import G1Model, csv_to_qpos, quat_wxyz_yaw


def _load_clip(name, data_dir=C.DATA_DIR, trim=C.TRIM):
    rows = np.genfromtxt(os.path.join(data_dir, name + ".csv"), delimiter=",")
    if trim:
        rows = rows[trim:len(rows) - trim]   # drop T-pose blend frames at both ends
    return csv_to_qpos(rows)  # (T, 36) wxyz


def _label_jump(model, q, foot_thr=0.13, takeoff_pad=14, land_pad=12, entry_pad=6):
    """Per-frame skill (0 walk, 1 jump) + (entry, takeoff, land) for each jump.

    Flight = both feet above foot_thr; the jump skill is padded back over the
    takeoff crouch and forward over the landing. The entry frame is a still-walking
    frame just before the crouch -- a good place to commit to a jump from walking.
    """
    footz = model.fk_feet(q)[:, :, 2].min(1)
    air = footz > foot_thr
    idx = np.where(air)[0]
    skill = np.zeros(len(q), np.int32)
    jumps = []                                       # (entry, takeoff, land, continues) local
    if len(idx):
        for s in np.split(idx, np.where(np.diff(idx) > 3)[0] + 1):
            if len(s) < 3:
                continue
            takeoff, land = int(s[0]), int(s[-1])
            a, b = max(0, takeoff - takeoff_pad), min(len(q), land + land_pad)
            skill[a:b] = 1
            w0, w1 = min(land + 40, len(q) - 1), min(land + 60, len(q) - 1)   # settled post-landing
            continues = w1 > w0 and np.linalg.norm(q[w1, 0:2] - q[w0, 0:2]) / ((w1 - w0) * C.DT) > 0.5
            jumps.append((max(0, a - entry_pad), takeoff, land, bool(continues)))
    return skill, jumps


def build_library(clips=None, out=C.LIB_PATH):
    """Concatenate clips into one array; precompute heading and FK foot positions."""
    clips = clips or C.LOCO_CLIPS
    clips = [c for c in clips if os.path.exists(os.path.join(C.DATA_DIR, c + ".csv"))]
    if not clips:
        raise FileNotFoundError(f"No clips in {C.DATA_DIR}; run scripts/download_data.sh")

    model = G1Model()
    qpos, clip_id, frame_in_clip, lengths = [], [], [], []
    for cid, name in enumerate(clips):
        q = _load_clip(name)
        qpos.append(q)
        clip_id.append(np.full(len(q), cid))
        frame_in_clip.append(np.arange(len(q)))
        lengths.append(len(q))
        print(f"  [{cid}] {name}: {len(q)} frames")

    qpos = np.concatenate(qpos)
    feet = model.fk_feet(qpos)                       # (N, 2, 3) world
    yaw = quat_wxyz_yaw(qpos[:, 3:7])                 # (N,)

    np.savez_compressed(
        out,
        qpos=qpos.astype(np.float32),
        feet_world=feet.astype(np.float32),
        yaw=yaw.astype(np.float32),
        clip_id=np.concatenate(clip_id).astype(np.int32),
        frame_in_clip=np.concatenate(frame_in_clip).astype(np.int32),
        lengths=np.array(lengths, np.int32),
        clip_names=np.array(clips),
    )
    print(f"Saved library: {qpos.shape[0]} frames, {len(clips)} clips -> {out}")
    return out


def build_jump_library(out=C.JUMP_LIB_PATH):
    """Walk base clip (skill=walk) + CAMDM walk->jump->walk clips (skill auto-labeled)."""
    model = G1Model()
    specs = [(C.JUMP_BASE_WALK, C.DATA_DIR, C.TRIM)] + \
            [(c, C.JUMP_DATA_DIR, 0) for c in C.JUMP_CLIPS]
    qpos, clip_id, frame_in_clip, lengths, names = [], [], [], [], []
    skill, j_entry, j_takeoff, j_land, j_cont = [], [], [], [], []
    off = 0
    for cid, (name, d, trim) in enumerate(specs):
        if not os.path.exists(os.path.join(d, name + ".csv")):
            continue
        q = _load_clip(name, d, trim)
        sk, jumps = _label_jump(model, q)
        qpos.append(q); skill.append(sk)
        clip_id.append(np.full(len(q), cid)); frame_in_clip.append(np.arange(len(q)))
        lengths.append(len(q)); names.append(name)
        for e, t, l, cont in jumps:                  # store as global frame indices
            j_entry.append(off + e); j_takeoff.append(off + t); j_land.append(off + l)
            j_cont.append(cont)
        off += len(q)
        print(f"  [{cid}] {name}: {len(q)} frames, {len(jumps)} jump(s)"
              f"{' (continues)' if jumps and jumps[0][3] else ' (stops)' if jumps else ''}")

    qpos = np.concatenate(qpos)
    np.savez_compressed(
        out,
        qpos=qpos.astype(np.float32),
        feet_world=model.fk_feet(qpos).astype(np.float32),
        yaw=quat_wxyz_yaw(qpos[:, 3:7]).astype(np.float32),
        clip_id=np.concatenate(clip_id).astype(np.int32),
        frame_in_clip=np.concatenate(frame_in_clip).astype(np.int32),
        lengths=np.array(lengths, np.int32),
        clip_names=np.array(names),
        skill=np.concatenate(skill).astype(np.int32),
        jump_entry=np.array(j_entry, np.int32),
        jump_takeoff=np.array(j_takeoff, np.int32),
        jump_land=np.array(j_land, np.int32),
        jump_continues=np.array(j_cont, bool),
    )
    print(f"Saved jump library: {len(qpos)} frames, {int(np.concatenate(skill).sum())} "
          f"jump frames, {len(j_entry)} jumps -> {out}")
    return out


def load_library(path=C.LIB_PATH):
    if not os.path.exists(path):
        (build_jump_library if path == C.JUMP_LIB_PATH else build_library)(out=path)
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    build_library()
    build_jump_library()
