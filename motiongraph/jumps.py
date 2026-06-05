"""Shared jump-skill indexing, used by both motion matching and motion graph.

A jump can only be *entered* from its pre-take-off run-up — never from the airborne
phase. `jump_entries` returns exactly those run-up frames (entry .. take-off) and a map
to each jump's landing frame, so a controller transitions in before take-off and then
rides the clip's own take-off / flight / landing.
"""
import numpy as np


def jump_entries(lib, pre_margin=2, continuing_only=True):
    """Pre-take-off run-up frames + {frame: land_frame}.

    pre_margin: stop this many frames before take-off (leaves room for the blend).
    continuing_only: only jumps that keep walking after landing.
    """
    enter, land_of = [], {}
    if "jump_entry" not in lib:
        return np.array([], np.int32), land_of
    cont = lib["jump_continues"] if "jump_continues" in lib else np.ones(len(lib["jump_entry"]), bool)
    for e, t, l, c in zip(lib["jump_entry"], lib["jump_takeoff"], lib["jump_land"], cont):
        if continuing_only and not c:
            continue
        for f in range(int(e), int(t) - pre_margin):       # strictly before take-off
            enter.append(f)
            land_of[f] = int(l)
    return np.array(enter, np.int32), land_of
