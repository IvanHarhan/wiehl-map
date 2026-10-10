"""Grow the main region's outline (region_data/prepared/territory.npz) along the A4 between the corridor (lon 7.465)
and the Wiehl junction (lon ~7.53) plus its slip roads, so the old tiles get the motorway the corridor chunks lead to.
Prints the old tiles that change (to re-export). Backup: territory_before_a4.npz."""
import json, math
import numpy as np
P = r"A:\blender_work\region_data\prepared"
BUF = 350.0
t = dict(np.load(P + r"\territory_before_a4.npz"))
mask, E0, N1, res = t["mask"].copy(), int(t["E0"]), int(t["N1"]), int(t["res"])
E_A, E_F, K0, LON0 = 6378137.0, 1 / 298.257222101, 0.9996, math.radians(9)
E2 = E_F * (2 - E_F); EP2 = E2 / (1 - E2)


def utm(lat, lon):
    phi, lam = np.radians(lat), np.radians(lon)
    N = E_A / np.sqrt(1 - E2 * np.sin(phi) ** 2)
    T = np.tan(phi) ** 2; C = EP2 * np.cos(phi) ** 2; A = np.cos(phi) * (lam - LON0)
    M = E_A * ((1 - E2/4 - 3*E2**2/64 - 5*E2**3/256) * phi - (3*E2/8 + 3*E2**2/32 + 45*E2**3/1024) * np.sin(2*phi)
               + (15*E2**2/256 + 45*E2**3/1024) * np.sin(4*phi) - (35*E2**3/3072) * np.sin(6*phi))
    x = 500000 + K0 * N * (A + (1 - T + C) * A**3 / 6 + (5 - 18*T + T*T + 72*C - 58*EP2) * A**5 / 120)
    y = K0 * (M + N * np.tan(phi) * (A*A/2 + (5 - T + 9*C + 4*C*C) * A**4 / 24 + (61 - 58*T + T*T + 600*C - 330*EP2) * A**6 / 720))
    return np.c_[x, y]


pts = []
for e in json.load(open(r"A:\blender_work\region_data\osm_roads.json", encoding="utf-8"))["elements"]:
    if e.get("tags", {}).get("highway") not in ("motorway", "motorway_link") or len(e.get("geometry", [])) < 2:
        continue
    g = e["geometry"]
    u = utm(np.array([p["lat"] for p in g]), np.array([p["lon"] for p in g]))
    lon = np.array([p["lon"] for p in g])
    for a, b, la in zip(u[:-1], u[1:], lon[:-1]):
        if 7.465 <= la <= 7.535:
            n = max(1, int(np.hypot(*(b - a)) / 10))
            pts += [a + (b - a) * s for s in np.linspace(0, 1, n + 1)]
pts = np.array(pts)
rr = int(BUF / res)
yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
disc = xx ** 2 + yy ** 2 <= rr ** 2
H, W = mask.shape
before = mask.copy()
for r, c in {(int((N1 * 1000 - y) / res), int((x - E0 * 1000) / res)) for x, y in pts}:
    r0, r1, c0, c1 = max(0, r - rr), min(H, r + rr + 1), max(0, c - rr), min(W, c + rr + 1)
    if r0 < r1 and c0 < c1:
        mask[r0:r1, c0:c1] |= disc[r0 - (r - rr):r1 - (r - rr), c0 - (c - rr):c1 - (c - rr)]
old_tiles = {tuple(map(int, x)) for x in np.load(r"A:\blender_work\region_data\territory.npz")["tiles"]}
changed = sorted({(E0 + c // 100, N1 - 1 - r // 100) for r, c in zip(*np.nonzero(mask & ~before))} & old_tiles)
t["mask"] = mask
np.savez_compressed(P + r"\territory.npz", **t)
print("points", len(pts), "cells added", int((mask & ~before).sum()), "km2", round((mask & ~before).sum() * res * res / 1e6, 2))
print("TILES", ",".join(f"{e}_{n}" for e, n in changed))
