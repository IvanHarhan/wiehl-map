"""Run from build_blend.py (exec) after build_streets.py: all the water as water (2026-09-28). Lakes and ponds
(natural=water, landuse=reservoir/basin), rivers' own areas (waterway=riverbank, water=river multipolygons such
as the Rhine) and rivers, canals and streams as lines with a width, rasterised at 3 m inside the territory; a sheet
at the water level (a lake's low shore, a river's smoothed bed line) and the terrain pressed under it.
Uses build_blend/build_streets globals: extra, tm, ground, in_territory_d, AFF, INV, UX0, UY0, MX0, MY0, MX1, MY1,
terrain, col, REGION, _smooth, densify."""
import numpy as np  # Blender's Python has no PIL: polygons and lines are rasterised by hand

W_RES = 4.0  # 3 m over the region's 17 x 17 km needed gigabytes next to the Blender build
LINE_W = {"river": 12.0, "canal": 10.0, "stream": 2.0}
ww, wh = int((MX1 - MX0) / W_RES) + 1, int((MY1 - MY0) / W_RES) + 1


def w_px(xy):
    return [((x - MX0) / W_RES, (MY1 - y) / W_RES) for x, y in xy]


def fill_rings(rings_px, w, h):
    """even-odd fill of pixel-space rings (outers and holes alike) into a w x h mask, scanline by scanline"""
    m = np.zeros((h, w), bool)
    e = np.vstack([np.c_[r, np.roll(r, -1, axis=0)] for r in rings_px])  # x0 y0 x1 y1
    e = e[e[:, 1] != e[:, 3]]
    for row in range(h):
        yc = row + 0.5
        on = (np.minimum(e[:, 1], e[:, 3]) <= yc) & (np.maximum(e[:, 1], e[:, 3]) > yc)
        if not on.any():
            continue
        s = e[on]
        xs = np.sort(s[:, 0] + (yc - s[:, 1]) * (s[:, 2] - s[:, 0]) / (s[:, 3] - s[:, 1]))
        for a, b in zip(xs[0::2], xs[1::2]):
            c0, c1 = max(0, int(np.ceil(a - 0.5))), min(w - 1, int(np.floor(b - 0.5)))
            if c0 <= c1:
                m[row, c0:c1 + 1] = True
    return m


def ring_xy(geom):
    return tm(np.array([p["lat"] for p in geom]), np.array([p["lon"] for p in geom]))


def rings_of(e):
    """outer and inner rings of a closed way or a multipolygon relation (member ways joined end to end)"""
    if e["type"] == "way":
        g = e.get("geometry") or []
        return ([ring_xy(g)] if len(g) > 3 and g[0] == g[-1] else []), []
    out = {"outer": [], "inner": []}
    for role in out:
        segs = [[(p["lat"], p["lon"]) for p in m["geometry"]] for m in e.get("members", [])
                if m.get("type") == "way" and m.get("role", "outer") == role and m.get("geometry")]
        while segs:
            ring = segs.pop(0)
            grown = True
            while grown and ring[0] != ring[-1]:
                grown = False
                for i, s in enumerate(segs):
                    if s[0] == ring[-1]:
                        ring += s[1:]
                    elif s[-1] == ring[-1]:
                        ring += s[::-1][1:]
                    elif s[-1] == ring[0]:
                        ring = s + ring[1:]
                    elif s[0] == ring[0]:
                        ring = s[::-1] + ring[1:]
                    else:
                        continue
                    segs.pop(i)
                    grown = True
                    break
            if len(ring) > 3:
                out[role].append(tm(np.array([p[0] for p in ring]), np.array([p[1] for p in ring])))
    return out["outer"], out["inner"]


def is_area(tg):
    return (tg.get("natural") == "water" or tg.get("waterway") == "riverbank"
            or tg.get("landuse") in ("reservoir", "basin"))


# ---- areas: each body its own level, the lowest fifth of the ground inside it ----
lev_a = np.full((wh, ww), -1e9, np.float32)
n_area = n_line = 0
for e in extra:
    tg = e.get("tags", {})
    if not is_area(tg):
        continue
    outers, inners = rings_of(e)
    outers = [r for r in outers if r[:, 0].max() > MX0 and r[:, 0].min() < MX1 and r[:, 1].max() > MY0 and r[:, 1].min() < MY1]
    if not outers:
        continue
    allr = np.vstack(outers)
    c0, c1 = max(0, int((allr[:, 0].min() - MX0) / W_RES) - 1), min(ww, int((allr[:, 0].max() - MX0) / W_RES) + 2)
    r0, r1 = max(0, int((MY1 - allr[:, 1].max()) / W_RES) - 1), min(wh, int((MY1 - allr[:, 1].min()) / W_RES) + 2)
    if c0 >= c1 or r0 >= r1:
        continue
    # just the body's own box; holes (islands) by parity
    m = fill_rings([np.array(w_px(r)) - (c0, r0) for r in outers + inners], c1 - c0, r1 - r0)
    rr, cc = np.nonzero(m)
    if len(rr) == 0:
        continue
    pick = np.random.default_rng(3).choice(len(rr), min(len(rr), 3000), replace=False)
    sx = MX0 + (cc[pick] + c0 + 0.5) * W_RES
    sy = MY1 - (rr[pick] + r0 + 0.5) * W_RES
    lv = float(np.percentile(ground(np.c_[sx, sy]), 20))
    sub = lev_a[r0:r1, c0:c1]
    sub[m] = np.maximum(sub[m], lv)
    n_area += 1

# ---- lines: rivers, canals, streams, their level following the smoothed ground along them ----
for e in extra:
    tg = e.get("tags", {})
    kind = tg.get("waterway")
    if e.get("type") != "way" or kind not in LINE_W or tg.get("tunnel") or tg.get("layer", "0").startswith("-") \
            or len(e.get("geometry", [])) < 2:
        continue
    xy = ring_xy(e["geometry"])
    if xy[:, 0].max() < MX0 or xy[:, 0].min() > MX1 or xy[:, 1].max() < MY0 or xy[:, 1].min() > MY1:
        continue
    try:
        wid = float(tg.get("width", "").split()[0])
    except (ValueError, IndexError):
        wid = LINE_W[kind]
    pts = densify(xy, W_RES / 2)
    zs = (_smooth(ground(pts), med=9, avg=5) - 0.05).astype(np.float32)
    px = np.array(w_px(pts))
    # a disc of the river's width stamped at every half cell along it
    rad = max(0.5, wid / W_RES / 2)
    k = np.arange(-int(np.ceil(rad)), int(np.ceil(rad)) + 1)
    dc, dr = np.meshgrid(k, k)
    disc = dc ** 2 + dr ** 2 <= max(rad * rad, 0.5)
    dc, dr = dc[disc], dr[disc]
    cc = (px[:, 0].astype(int)[:, None] + dc[None, :]).ravel()
    rr = (px[:, 1].astype(int)[:, None] + dr[None, :]).ravel()
    zz = np.repeat(zs, len(dc))
    ok = (cc >= 0) & (cc < ww) & (rr >= 0) & (rr < wh)
    np.maximum.at(lev_a, (rr[ok], cc[ok]), zz[ok])
    n_line += 1

lev = lev_a
wet = lev > -1e8
# the territory only
rr, cc = np.nonzero(wet)
cx_, cy_ = MX0 + (cc + 0.5) * W_RES, MY1 - (rr + 0.5) * W_RES
d_ = (np.c_[cx_, cy_] - AFF[2]) @ INV
keep = in_territory_d(d_[:, 0], d_[:, 1]) if REGION else np.ones(len(rr), bool)
wet[:] = False
wet[rr[keep], cc[keep]] = True

# ---- the sheet: a vertex at every cell corner touching water, its height the mean of those cells ----
rr, cc = np.nonzero(wet)
if len(rr):
    vid = -np.ones((wh + 1, ww + 1), np.int32)
    zsum = np.zeros((wh + 1, ww + 1), np.float32)
    zcnt = np.zeros((wh + 1, ww + 1), np.float32)
    for dr in (0, 1):
        for dc in (0, 1):
            np.add.at(zsum, (rr + dr, cc + dc), lev[rr, cc])
            np.add.at(zcnt, (rr + dr, cc + dc), 1.0)
    vr, vc = np.nonzero(zcnt > 0)
    vid[vr, vc] = np.arange(len(vr))
    vx, vy = MX0 + vc * W_RES, MY1 - vr * W_RES
    vz = zsum[vr, vc] / zcnt[vr, vc] + 0.05
    faces = np.c_[vid[rr + 1, cc], vid[rr + 1, cc + 1], vid[rr, cc + 1], vid[rr, cc]]
    wme = bpy.data.meshes.new("NRW_Water")
    wme.from_pydata(np.c_[vx, vy, vz].tolist(), [], faces.tolist())
    wme.uv_layers.new(name="UVMap").data.foreach_set("uv", (np.c_[vx, vy][faces.ravel()] / 10.0).astype(np.float32).ravel())
    wm = bpy.data.materials.new("NRW_Water")
    wm.use_nodes = True
    bsdf = wm.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.03, 0.07, 0.08, 1)
    bsdf.inputs["Roughness"].default_value = 0.05
    wme.materials.append(wm)
    wob = bpy.data.objects.new("NRW_Water", wme)
    col.objects.link(wob)
    # the terrain dips under the water so the sheet never fights it
    tme = terrain.data
    tco = np.empty(len(tme.vertices) * 3, np.float32); tme.vertices.foreach_get("co", tco); tco = tco.reshape(-1, 3)
    tc = ((tco[:, 0] - MX0) / W_RES).astype(int)
    tr = ((MY1 - tco[:, 1]) / W_RES).astype(int)
    ok = (tc >= 0) & (tc < ww) & (tr >= 0) & (tr < wh)
    idx = np.nonzero(ok)[0]
    under = idx[wet[tr[idx], tc[idx]]]
    tco[under, 2] = np.minimum(tco[under, 2], lev[tr[under], tc[under]] - 0.8)
    tme.vertices.foreach_set("co", tco.ravel())
    tme.update()
    print("WATER areas", n_area, "lines", n_line, "cells", len(rr), "km2", round(len(rr) * W_RES * W_RES / 1e6, 2),
          "terrain vertices pressed", len(under))
else:
    print("WATER none")
