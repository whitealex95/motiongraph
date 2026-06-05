"""Shared jump-skill indexing, used by both motion matching and motion graph.

A jump is entered ONLY in its `ready` phase (the run-up before take-off) and exited
only after its `after` phase (post-touchdown recovery). `jump_entries` returns the
`ready` frames of each jump and maps each to its landing/apex frame, so a controller
transitions in during `ready` and then rides take-off / flight / landing.
"""
import numpy as np

from . import config as C

READY = 1   # phase code (see config.JUMP_PHASES)


def jump_entries(lib, continuing_only=True):
    """`ready`-phase run-up frames + {frame: land_frame} + {frame: apex_frame}."""
    enter, land_of, apex_of = [], {}, {}
    if "jump_entry" not in lib:
        return np.array([], np.int32), land_of, apex_of
    cont = lib["jump_continues"] if "jump_continues" in lib else np.ones(len(lib["jump_entry"]), bool)
    apex = lib["jump_apex"] if "jump_apex" in lib else lib["jump_takeoff"]
    phase = lib["phase"] if "phase" in lib else None
    for e, t, l, ap, c in zip(lib["jump_entry"], lib["jump_takeoff"], lib["jump_land"], apex, cont):
        if continuing_only and not c:
            continue
        for f in range(int(e), int(t)):
            if phase is None or phase[f] == READY:          # confine entry to the ready phase
                enter.append(f)
                land_of[f] = int(l)
                apex_of[f] = int(ap)
    return np.array(enter, np.int32), land_of, apex_of
