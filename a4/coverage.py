"""Which A4 points (motorway + links, lon 7.07-7.62) lie outside both the old territory mask and the corridor mask,
and whether the old region ever built them (old tiles had no motorway support before 2026-09-27)."""
import json, sys
import numpy as np
sys.path.insert(0, r"A:\blender_work\a4")
from make_corridor import utm  # noqa: E402  (re-running make_corridor is harmless: same output)
