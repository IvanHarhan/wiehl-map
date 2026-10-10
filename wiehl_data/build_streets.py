"""Run from build_blend.py (exec) after build_extras.py: asphalt roads with markings and zebra crossings,
curbs + sidewalks, surface parking lots with bays, detailed ground material.
Uses build_blend globals: P, UX0, UY0, UX1, UY1, AFF, INV, dem, offset, col, Nodes, TEX, TEX_SIZE_M, terrain, ortho."""
import math
import numpy as np
from mathutils import Vector
from mathutils import geometry as mgeom

# carriageway widths as measured on German roads (kerb to kerb), for ways OSM gives no width or lanes for
ROAD_W = {"motorway": 11.25, "motorway_link": 6.0, "trunk": 11.25, "trunk_link": 6.0,
          "primary": 8.0, "primary_link": 6.5, "secondary": 7.5, "secondary_link": 6.5, "tertiary": 7.0,
          "tertiary_link": 6.0, "residential": 6.0, "unclassified": 5.5, "living_street": 5.0, "service": 3.5}
LANE_W = 3.5
# Autobahn (the A4 corridor, 2026-09-27): OSM draws each carriageway as its own one-way; 3.75 m lanes, a 0.75 m strip
# on the median side, a 3 m hard shoulder on the right
MOTORWAY = {"motorway", "motorway_link", "trunk", "trunk_link"}
MW_LANE, MW_LEFT, MW_SHOULDER = 3.75, 0.75, 3.0
GUTTER_W = 0.3  # the paved gutter (Rinne) along the kerb
WITH_SIDEWALK = set(ROAD_W) - {"service"} - MOTORWAY
CENTRE_LINE = {"primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link"}
# surfaces float a few cm over the 2 m terrain grid; higher road classes win where carriageways overlap
CLASS_Z = {"motorway": 0.09, "motorway_link": 0.085, "trunk": 0.09, "trunk_link": 0.085,
           "primary": 0.08, "primary_link": 0.08, "secondary": 0.075, "secondary_link": 0.075, "tertiary": 0.07,
           "tertiary_link": 0.07, "unclassified": 0.065, "residential": 0.065, "living_street": 0.065, "service": 0.06}
PARKING_Z, MARK_Z = 0.055, 0.015
CURB_H, CURB_W = 0.2, 0.15  # curb top above ground, i.e. ~12 cm above the road surface
sc = bpy.context.scene
LAT0, LON0 = sc["lat"], sc["lon"]


def tm(lat, lon):
    R = 6378137.0
    la, lo = np.radians(lat), np.radians(lon - LON0)
    Bv = np.sin(lo) * np.cos(la)
    return np.c_[0.5 * R * np.log((1 + Bv) / (1 - Bv)), R * (np.arctan(np.tan(la) / np.cos(lo)) - math.radians(LAT0))]


def ground(xy):
    d = (xy - AFF[2]) @ INV
    c = np.clip(d[:, 0] / DEM_RES - 0.5, 0, dem.shape[1] - 1.001)
    r = np.clip(((UY1 - UY0) - d[:, 1]) / DEM_RES - 0.5, 0, dem.shape[0] - 1.001)
    c0, r0 = c.astype(int), r.astype(int)
    fc, fr = c - c0, r - r0
    z = (dem[r0, c0] * (1 - fc) * (1 - fr) + dem[r0, c0 + 1] * fc * (1 - fr)
         + dem[r0 + 1, c0] * (1 - fc) * fr + dem[r0 + 1, c0 + 1] * fc * fr)
    return z + offset


def inside_terrain(xy, margin=1.0):
    d = (xy - AFF[2]) @ INV
    box = (d[:, 0] > margin) & (d[:, 0] < UX1 - UX0 - margin) & (d[:, 1] > margin) & (d[:, 1] < UY1 - UY0 - margin)
    # the region stops at the user's outline: roads, sidewalks and props with it
    return box & in_territory_d(d[:, 0], d[:, 1]) if REGION else box


def mw_lanes(tags):
    try:
        return max(1, int(tags["lanes"]))
    except (KeyError, ValueError):
        return 1 if tags["highway"].endswith("_link") else 2


def width(tags):
    if tags["highway"] in MOTORWAY:
        link = tags["highway"].endswith("_link")
        return MW_LEFT + mw_lanes(tags) * MW_LANE + (1.0 if link else MW_SHOULDER)
    fast = 7.5 if tags.get("_fast") else 0.0  # the user's through roads: never narrower than a proper Landstraße
    try:
        return max(float(tags["width"].split()[0]), fast)
    except (KeyError, ValueError):
        pass
    try:
        return max(int(tags["lanes"]) * LANE_W, ROAD_W[tags["highway"]], fast)
    except (KeyError, ValueError):
        return max(ROAD_W[tags["highway"]], fast)


def densify(xy, step=1.5):
    out = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.array(out)


def frame(pts):
    """unit tangents, left normals and running distance along a polyline"""
    tan = np.gradient(pts, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-6)
    along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return tan, np.c_[-tan[:, 1], tan[:, 0]], along


def in_poly(pts, poly):
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    hit = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
    return hit.sum(1) % 2 == 1


class MeshBuilder:
    def __init__(self):
        self.v, self.f, self.m, self.uv = [], [], [], []

    def face(self, pts, mat=0, uvs=None):
        b = len(self.v)
        self.v.extend(tuple(p) for p in pts)
        self.f.append(tuple(range(b, b + len(pts))))
        self.m.append(mat)
        self.uv.extend(uvs if uvs is not None else [(p[0], p[1]) for p in pts])

    def strip(self, left, right, zl, zr, mat=0, along=None, keep=None):
        """quads between two polylines (left/right of travel direction), face up"""
        for i in range(len(left) - 1):
            if keep is not None and not (keep[i] and keep[i + 1]):
                continue
            quad = [(*right[i], zr[i]), (*right[i + 1], zr[i + 1]), (*left[i + 1], zl[i + 1]), (*left[i], zl[i])]
            uvs = None
            if along is not None:
                w = float(np.linalg.norm(left[i] - right[i]))
                uvs = [(along[i], 0), (along[i + 1], 0), (along[i + 1], w), (along[i], w)]
            self.face(quad, mat, uvs)

    def build(self, name, uv_name, materials):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        me.polygons.foreach_set("material_index", np.array(self.m, np.int32))
        me.uv_layers.new(name=uv_name).data.foreach_set("uv", np.array(self.uv, np.float32).ravel())
        me.validate()
        for mt in materials:
            me.materials.append(mt)
        ob = bpy.data.objects.new(name, me)
        col.objects.link(ob)
        return ob


# ---------------- OSM input ----------------
ways = []
for e in json.load(open(os.path.join(os.path.dirname(P), "osm_roads.json")))["elements"]:
    t = e.get("tags", {})
    if t.get("highway") not in ROAD_W or t.get("bridge") or t.get("tunnel") or len(e.get("geometry", [])) < 2:
        continue
    g = e["geometry"]
    ways.append((t, tm(np.array([p["lat"] for p in g]), np.array([p["lon"] for p in g]))))
extra = json.load(open(os.path.join(os.path.dirname(P), "osm_extra.json"), encoding="utf-8"))["elements"]

# the roads the user drew in grey on their sketch (territory_from_sketch.py): the OSM ways running along those strokes
# are the through roads between the towns, marked "_fast" and dressed like German country roads
n_fast = 0
sketch = os.path.join(os.path.dirname(P), "territory.npz")
if REGION and os.path.exists(sketch):
    g_utm = np.load(sketch)["grey"].astype(np.float64)
    g_loc = to_local(g_utm[:, 0] - UX0, g_utm[:, 1] - UY0)
    g_cells = set(map(tuple, np.floor(g_loc / 80).astype(int)))

    def near_grey(p):
        c = np.floor(p / 80).astype(int)
        return np.array([any((a + i, b + j) in g_cells for i in (-1, 0, 1) for j in (-1, 0, 1)) for a, b in c])

    for t, xy in ways:
        if t["highway"] in ("primary", "primary_link", "secondary", "secondary_link", "tertiary", "trunk"):
            if near_grey(densify(xy, 20.0)).mean() >= 0.6:
                t["_fast"] = "yes"
                n_fast += 1
for t, xy in ways:
    if t["highway"] in MOTORWAY:
        t["_fast"] = "yes"  # delineator posts, no pavement
DETAIL_CODES = [int(k[:3]) * 10000 + int(k[4:]) for k in meta.get("detail", [])]


def in_town(xy):
    """in one of the towns' detailed tiles (region only)"""
    if not REGION:
        return np.ones(len(xy), bool)
    d = (xy - AFF[2]) @ INV
    return np.isin(np.floor((d[:, 0] + UX0) / 1000).astype(int) * 10000 + np.floor((d[:, 1] + UY0) / 1000).astype(int),
                   DETAIL_CODES)

# carriageway count raster (0.5 m) over the terrain: >=1 on a road, >=2 where carriageways overlap (junctions)
RES = 1.0 if REGION else 0.5  # a 0.5 m raster over 15 x 17 km would be a gigabyte
corners = np.array([[0, 0, 1], [UX1 - UX0, 0, 1], [0, UY1 - UY0, 1], [UX1 - UX0, UY1 - UY0, 1]], float) @ AFF
MX0, MY0 = corners.min(0) - 5
MX1, MY1 = corners.max(0) + 5
mw, mh = int((MX1 - MX0) / RES), int((MY1 - MY0) / RES)
cnt = np.zeros((mh, mw), np.uint8)
for t, xy in ways:
    rr = width(t) / RES / 2
    k = np.arange(-int(rr), int(rr) + 1)
    dc, dr = np.meshgrid(k, k)
    disc = dc ** 2 + dr ** 2 <= rr ** 2
    dc, dr = dc[disc], dr[disc]
    cc, rw = (xy[:, 0] - MX0) / RES, (MY1 - xy[:, 1]) / RES
    c0, c1 = int(max(0, cc.min() - rr - 2)), int(min(mw, cc.max() + rr + 3))
    r0, r1 = int(max(0, rw.min() - rr - 2)), int(min(mh, rw.max() + rr + 3))
    if c0 >= c1 or r0 >= r1:
        continue
    loc = np.zeros((r1 - r0, c1 - c0), bool)
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(1, int(np.linalg.norm(b - a) / (RES / 2)))
        s = a + (b - a) * (np.arange(n + 1) / n)[:, None]
        c = ((s[:, 0] - MX0) / RES).astype(int)[:, None] + dc[None, :] - c0
        r = ((MY1 - s[:, 1]) / RES).astype(int)[:, None] + dr[None, :] - r0
        ok = (c >= 0) & (c < c1 - c0) & (r >= 0) & (r < r1 - r0)
        loc[r[ok], c[ok]] = True
    cnt[r0:r1, c0:c1] += loc


def road_count(xy):
    c = ((xy[:, 0] - MX0) / RES).astype(int)
    r = ((MY1 - xy[:, 1]) / RES).astype(int)
    ok = (c >= 0) & (c < mw) & (r >= 0) & (r < mh)
    out = np.zeros(len(xy), np.uint8)
    out[ok] = cnt[r[ok], c[ok]]
    return out


# ---------------- road surface height: the bare-earth DEM along a road has culvert pits, kerb and hedge bumps;
# a road runs smooth, so each way's centre profile is median-filtered (25 m) then averaged (10 m) and splatted into
# a 4 m raster that every carriageway, marking, gutter and kerb foot samples instead of the raw terrain ----------------
def _smooth(z, med=25, avg=11):
    if len(z) < 5:
        return z
    h = med // 2
    zp = np.pad(z, h, mode="edge")
    zm = np.median(np.lib.stride_tricks.sliding_window_view(zp, med), axis=1)
    k = np.ones(avg) / avg
    return np.convolve(np.pad(zm, avg // 2, mode="edge"), k, mode="valid")


RZ_RES = 4.0
rzw, rzh = int((MX1 - MX0) / RZ_RES) + 2, int((MY1 - MY0) / RZ_RES) + 2
# two layers: the Autobahn's own carriageways, and every other road. In one raster the roads next to the A4 (the
# ramps up to an overpass) were averaged with it: they sagged to its level, the overpass decks with them, and the
# A4 got bumps (2026-09-27). Callers pick the layer through RZ_LAYER (1 while drawing a motorway or trunk).
MW_MAIN = ("motorway", "trunk")
RZ_LAYER = 0
rz_sum = np.zeros((2, rzh, rzw), np.float32)
rz_cnt = np.zeros((2, rzh, rzw), np.float32)
for t, xy in ways:
    pts = densify(xy, 1.0)
    zs = _smooth(ground(pts))
    lay = 1 if t["highway"] in MW_MAIN else 0
    reach = width(t) / 2 + 2.0
    for dx_ in np.arange(-reach, reach + 0.1, RZ_RES / 2):
        for dy_ in np.arange(-reach, reach + 0.1, RZ_RES / 2):
            if dx_ * dx_ + dy_ * dy_ > reach * reach:
                continue
            c = ((pts[:, 0] + dx_ - MX0) / RZ_RES).astype(int)
            r = ((MY1 - pts[:, 1] - dy_) / RZ_RES).astype(int)
            ok = (c >= 0) & (c < rzw) & (r >= 0) & (r < rzh)
            np.add.at(rz_sum[lay], (r[ok], c[ok]), zs[ok])
            np.add.at(rz_cnt[lay], (r[ok], c[ok]), 1.0)


def road_z(xy):
    """smoothed road height (bilinear over the 4 m raster); nan off the roads"""
    fc = (xy[:, 0] - MX0) / RZ_RES - 0.5
    fr = (MY1 - xy[:, 1]) / RZ_RES - 0.5
    c0 = np.clip(np.floor(fc).astype(int), 0, rzw - 2)
    r0 = np.clip(np.floor(fr).astype(int), 0, rzh - 2)
    ax, ay = np.clip(fc - c0, 0, 1), np.clip(fr - r0, 0, 1)
    s = w = 0
    cnt_l, sum_l = rz_cnt[RZ_LAYER], rz_sum[RZ_LAYER]
    for dr, dc, wt in ((0, 0, (1 - ax) * (1 - ay)), (0, 1, ax * (1 - ay)), (1, 0, (1 - ax) * ay), (1, 1, ax * ay)):
        cn = cnt_l[r0 + dr, c0 + dc]
        s = s + wt * np.where(cn > 0, sum_l[r0 + dr, c0 + dc] / np.maximum(cn, 1), 0)
        w = w + wt * (cn > 0)
    return np.where(w > 0.05, s / np.maximum(w, 1e-6), np.nan)


def ground_road(xy):
    """the smoothed ground under a road, else the terrain (callers add their layer offset as they did to ground)"""
    z = road_z(xy)
    return np.where(np.isnan(z), ground(xy), z)


if REGION:
    # the 5 m terrain out of town bulges through roads and their sidewalks: sink it along the carriageways;
    # anywhere, terrain never stands above the smoothed road surface
    tme = terrain.data
    tco = np.empty(len(tme.vertices) * 3, np.float32); tme.vertices.foreach_get("co", tco); tco = tco.reshape(-1, 3)
    coarse = np.nonzero(terrain_step >= 5.0)[0]
    near = np.zeros(len(coarse), bool)
    for off in ((0, 0), (3, 0), (-3, 0), (0, 3), (0, -3)):
        near |= road_count(tco[coarse, :2] + np.array(off, np.float32)) > 0
    tco[coarse[near], 2] -= 0.3
    rzt = road_z(tco[:, :2].astype(np.float64))
    RZ_LAYER = 1  # and under the Autobahn
    rzm = road_z(tco[:, :2].astype(np.float64))
    RZ_LAYER = 0
    rzt = np.where(np.isnan(rzt), rzm, np.where(np.isnan(rzm), rzt, np.minimum(rzt, rzm)))
    # only right under a carriageway: pressing the whole 4 m cell dug a ditch along roads in cuttings
    under = ~np.isnan(rzt) & (road_count(tco[:, :2]) > 0)
    tco[under, 2] = np.minimum(tco[under, 2], rzt[under] - 0.15)
    tme.vertices.foreach_set("co", tco.ravel())
    tme.update()
    print("ROAD_SMOOTH terrain vertices pressed under roads", int(under.sum()))

# trees and bushes standing in a carriageway (the vegetation mask sees the crowns over the road) go
for name in ("NRW_Trees", "NRW_Bushes"):
    ob = bpy.data.objects.get(name)
    if ob is None:
        continue
    me0 = ob.data
    n0 = len(me0.vertices)
    vco = np.empty(n0 * 3, np.float32); me0.vertices.foreach_get("co", vco); vco = vco.reshape(-1, 3)
    on_road = np.zeros(n0, bool)
    for off in ((0, 0), (1.5, 0), (-1.5, 0), (0, 1.5), (0, -1.5)):
        on_road |= road_count(vco[:, :2] + np.array(off, np.float32)) > 0
    keep_v = ~on_road
    me1 = bpy.data.meshes.new(name)
    me1.vertices.add(int(keep_v.sum()))
    me1.vertices.foreach_set("co", vco[keep_v].ravel())
    for a in me0.attributes:
        if a.domain != "POINT" or a.name == "position" or a.name.startswith(".") or \
                a.data_type not in ("INT", "FLOAT", "FLOAT_VECTOR"):
            continue
        prop, width_ = ("vector", 3) if a.data_type == "FLOAT_VECTOR" else ("value", 1)
        vals = np.empty(n0 * width_, np.int32 if a.data_type == "INT" else np.float32)
        a.data.foreach_get(prop, vals)
        me1.attributes.new(a.name, a.data_type, "POINT").data.foreach_set(prop, vals.reshape(n0, -1)[keep_v].ravel())
    ob.data = me1
    bpy.data.meshes.remove(me0)
    print("OFF_ROAD", name, "removed", int(on_road.sum()), "of", n0)


roads, marks, walks, gutters = MeshBuilder(), MeshBuilder(), MeshBuilder(), MeshBuilder()

# ---------------- carriageways + markings ----------------
for t, xy in ways:
    hw = t["highway"]
    RZ_LAYER = 1 if hw in MW_MAIN else 0
    zc = CLASS_Z[hw]
    w2 = width(t) / 2
    pts = densify(xy, 1.0)
    ins = inside_terrain(pts)
    if not ins.any():
        continue
    tan, nrm, along = frame(pts)
    L, R = pts + nrm * w2, pts - nrm * w2
    roads.strip(L, R, ground_road(L) + zc, ground_road(R) + zc, along=along, keep=ins)
    # round caps close the wedge gaps where ways meet at an angle
    for p, ok in ((pts[0], ins[0]), (pts[-1], ins[-1])):
        if ok:
            ring = np.array([p + w2 * np.array((math.cos(a), math.sin(a))) for a in np.linspace(0, 2 * math.pi, 17)[:-1]])
            zr = ground_road(ring) + zc - 0.002
            roads.face([(*q, z) for q, z in zip(ring, zr)], uvs=[tuple(q) for q in ring])

    junction = road_count(pts) >= 2
    line_ok = ins & ~junction
    fast = bool(t.get("_fast"))
    out_town = ~in_town(pts)
    rural = fast & out_town
    if hw in MOTORWAY:
        # left edge line by the median, dashed lane lines (6 m / 12 m), a broad solid line to the hard shoulder
        n_mw = mw_lanes(t)
        offs = [(w2 - MW_LEFT, 0.075, None)] + [(w2 - MW_LEFT - k * MW_LANE, 0.075, 18.0) for k in range(1, n_mw)]
        offs.append((w2 - MW_LEFT - n_mw * MW_LANE, 0.125, None))
        mw_ok = ins & (road_count(pts) < 3)  # through junction areas of the slip roads too
        for off, half, period in offs:
            c = pts + nrm * off
            hl = nrm * half
            keep = mw_ok if period is None else mw_ok & ((along % period) < 6.0)
            marks.strip(c + hl, c - hl, ground_road(c + hl) + zc + MARK_Z, ground_road(c - hl) + zc + MARK_Z, along=along, keep=keep)
    if hw in CENTRE_LINE and t.get("oneway") not in ("yes", "-1") and t.get("junction") != "roundabout" and t.get("lanes") != "1":
        # guide line: 3 m dash / 6 m gap in town, 6 m / 12 m on a country road
        dash = np.where(rural, (along % 18.0) < 6.0, (along % 9.0) < 3.0)
        hl = nrm * 0.06
        marks.strip(pts + hl, pts - hl, ground_road(pts + hl) + zc + MARK_Z, ground_road(pts - hl) + zc + MARK_Z,
                    along=along, keep=line_ok & dash)
    # lane lines on multi-lane roads: dashed between lanes of one direction, solid down the middle of four
    try:
        n_lanes = int(t.get("lanes", "0"))
    except ValueError:
        n_lanes = 0
    two_way = t.get("oneway") not in ("yes", "-1") and t.get("junction") != "roundabout"
    if hw not in MOTORWAY and (n_lanes >= 3 or (n_lanes == 2 and not two_way)):
        lw = 2 * w2 / n_lanes
        for k in range(1, n_lanes):
            off = -w2 + k * lw
            centre = two_way and 2 * k == n_lanes
            c = pts + nrm * off
            hl = nrm * (0.075 if centre else 0.06)
            keep = line_ok if centre else line_ok & ((along % 9.0) < 3.0)
            marks.strip(c + hl, c - hl, ground_road(c + hl) + zc + MARK_Z, ground_road(c - hl) + zc + MARK_Z, along=along, keep=keep)
    # edge lines: main roads everywhere, secondary and tertiary roads out of town
    if hw not in MOTORWAY and (hw.startswith("primary") or fast or (hw in ("secondary", "tertiary") and out_town.any())):
        edge_ok = line_ok if (hw.startswith("primary") or fast) else line_ok & out_town
        for s in (-1, 1):
            c = pts + s * nrm * (w2 - 0.3)
            hl = nrm * 0.075
            marks.strip(c + hl, c - hl, ground_road(c + hl) + zc + MARK_Z, ground_road(c - hl) + zc + MARK_Z, along=along, keep=edge_ok)
    if fast:
        # white delineator posts every 50 m either side out of town (Leitpfosten), in the markings' white paint
        for s0 in np.arange(25.0, along[-1], 50.0):
            i = int(np.searchsorted(along, s0))
            if i >= len(pts) or not (rural[i] and ins[i]):
                continue
            for s in (-1, 1):
                c = pts[i] + s * nrm[i] * (w2 + 1.0)
                if road_count(c[None])[0]:
                    continue
                z0 = ground(c[None])[0]
                ax, ay = tan[i] * 0.06, nrm[i] * 0.05
                corners = [c + sx * ax + sy * ay for sy in (-1, 1) for sx in (-1, 1)]
                for z_lo, z_hi in ((z0 - 0.1, z0 + 1.0),):
                    b, tp = [(*q, z_lo) for q in corners], [(*q, z_hi) for q in corners]
                    for q in ((0, 1, 3, 2),):
                        marks.face([tp[j] for j in q])
                    for a_, b_ in ((0, 1), (1, 3), (3, 2), (2, 0)):
                        marks.face([b[a_], b[b_], tp[b_], tp[a_]])

RZ_LAYER = 0

# ---------------- zebra crossings ----------------
seg_a = np.vstack([xy[:-1] for _, xy in ways])
seg_b = np.vstack([xy[1:] for _, xy in ways])
seg_w = np.concatenate([[width(t) / 2] * (len(xy) - 1) for t, xy in ways])
seg_z = np.concatenate([[CLASS_Z[t["highway"]]] * (len(xy) - 1) for t, xy in ways])
n_zebra = 0
for e in extra:
    tg = e.get("tags", {})
    if e["type"] != "node" or tg.get("highway") != "crossing" or tg.get("crossing") not in ("uncontrolled", "marked", "zebra"):
        continue
    if tg.get("crossing:island") == "yes" and tg.get("crossing_ref") != "zebra":
        continue
    p = tm(np.array([e["lat"]]), np.array([e["lon"]]))[0]
    ab = seg_b - seg_a
    tpar = np.clip(((p - seg_a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    proj = seg_a + ab * tpar[:, None]
    i = int(np.argmin(np.linalg.norm(proj - p, axis=1)))
    if np.linalg.norm(proj[i] - p) > 6:
        continue
    d = ab[i] / np.linalg.norm(ab[i])
    n = np.array((-d[1], d[0]))
    for k in np.arange(-seg_w[i] + 0.5, seg_w[i] - 0.25, 1.0):
        c = proj[i] + n * k
        quad = np.array([c - d * 1.5 - n * 0.25, c + d * 1.5 - n * 0.25, c + d * 1.5 + n * 0.25, c - d * 1.5 + n * 0.25])
        if not inside_terrain(quad).all():
            continue
        zq = ground_road(quad) + seg_z[i] + MARK_Z
        marks.face([(*q, z) for q, z in zip(quad, zq)])
    n_zebra += 1

# ---------------- curbs + sidewalks ----------------
n_side = 0
for t, xy in ways:
    if t["highway"] not in WITH_SIDEWALK:
        continue
    sides = {"both": (-1, 1), "left": (1,), "right": (-1,), "yes": (-1, 1)}.get(t.get("sidewalk", "both"), ())
    if not sides:
        continue
    pts = densify(xy)
    tan, nrm, along = frame(pts)
    w2 = width(t) / 2
    sw_w = 2.5 if t["highway"].startswith(("primary", "secondary")) else 2.0
    ds = (w2, w2 + CURB_W, w2 + CURB_W + sw_w)
    zc = CLASS_Z[t["highway"]]
    for s in sides:
        rows = [pts + s * nrm * d for d in ds]
        zs = [ground_road(pts)] * 3  # kerb and pavement ride on the smoothed road, level across
        g_in = pts + s * nrm * (w2 - GUTTER_W)
        zg_in, zg_out = ground_road(g_in) + zc + 0.004, zs[0] + zc + 0.004
        ok = inside_terrain(rows[2]) & (road_count(pts + s * nrm * (w2 + CURB_W + sw_w / 2)) == 0)
        if t.get("_fast") and "sidewalk" not in t:
            ok &= in_town(pts)  # country roads have a verge, not a pavement
        for i in range(len(pts) - 1):
            if not (ok[i] and ok[i + 1]):
                continue
            a0, a1 = along[i], along[i + 1]
            pt = lambda k, q, h: (*rows[k][q], zs[k][q] + h)
            # curb face, curb top, sidewalk, outer face (winding flipped for the left side)
            qs = [
                ((pt(0, i, 0), pt(0, i + 1, 0), pt(0, i + 1, CURB_H), pt(0, i, CURB_H)), 0, ((a0, 0), (a1, 0), (a1, CURB_H), (a0, CURB_H))),
                ((pt(0, i, CURB_H), pt(0, i + 1, CURB_H), pt(1, i + 1, CURB_H), pt(1, i, CURB_H)), 0, ((a0, 0), (a1, 0), (a1, CURB_W), (a0, CURB_W))),
                ((pt(1, i, CURB_H), pt(1, i + 1, CURB_H), pt(2, i + 1, CURB_H), pt(2, i, CURB_H)), 1, None),
                ((pt(2, i, CURB_H), pt(2, i + 1, CURB_H), pt(2, i + 1, -0.1), pt(2, i, -0.1)), 0, ((a0, 0), (a1, 0), (a1, CURB_H), (a0, CURB_H))),
            ]
            for quad, mat, uv4 in qs:
                if s > 0:
                    quad = quad[::-1]
                    uv4 = uv4[::-1] if uv4 else None
                walks.face(quad, mat, list(uv4) if uv4 else None)
            # the paved gutter on the road side of the kerb, face up
            gi = [(*g_in[q], zg_in[q]) for q in (i, i + 1)]
            go = [(*rows[0][q], zg_out[q]) for q in (i, i + 1)]
            inner, outer = (gi, go) if s > 0 else (go, gi)
            gutters.face([inner[0], inner[1], outer[1], outer[0]], 0,
                         [(a0, 0), (a1, 0), (a1, GUTTER_W), (a0, GUTTER_W)])
        n_side += 1

# ---------------- surface parking lots ----------------
lots = MeshBuilder()
n_lots = n_bays = 0
park_spots, rng_park = [], np.random.default_rng(17)  # x, y, z, heading (rad, local frame), 0 lot / 1 kerb
for e in extra:
    tg = e.get("tags", {})
    if e["type"] != "way" or tg.get("amenity") != "parking" or tg.get("parking") in ("underground", "multi-storey", "rooftop"):
        continue
    poly = tm(np.array([p["lat"] for p in e["geometry"]]), np.array([p["lon"] for p in e["geometry"]]))
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    if len(poly) < 3 or not inside_terrain(poly.mean(0)[None])[0]:
        continue
    stack = [tuple(poly[i] for i in tri) for tri in mgeom.tessellate_polygon([[Vector((x, y, 0)) for x, y in poly]])]
    while stack:
        a, b, c = stack.pop()
        if max(np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a)) > 3.0:
            ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
            stack += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
            continue
        tri = np.array([a, b, c])
        if np.cross(b - a, c - a) < 0:
            tri = tri[::-1]
        zt = ground(tri) + PARKING_Z
        lots.face([(*q, z) for q, z in zip(tri, zt)])
    # bays: rows of 5 m stalls either side of a 6 m aisle, aligned with the lot's longest side
    edges = np.roll(poly, -1, axis=0) - poly
    U = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    U /= np.linalg.norm(U)
    V = np.array((-U[1], U[0]))
    pu, pv = poly @ U, poly @ V
    for b0 in np.arange(pv.min(), pv.max(), 16.0):
        for s0 in (b0, b0 + 11.0):
            if s0 + 5.0 > pv.max():
                continue
            for u in np.arange(pu.min() + 0.3, pu.max(), 2.5):
                p0, p1 = U * u + V * s0, U * u + V * (s0 + 5.0)
                probe = np.array([p0 + V * 0.3, (p0 + p1) / 2, p1 - V * 0.3])
                if not in_poly(probe, poly).all() or road_count(probe).any():
                    continue
                hw_ = U * 0.06
                quad = np.array([p0 - hw_, p0 + hw_, p1 + hw_, p1 - hw_])
                if np.cross(quad[1] - quad[0], quad[2] - quad[0]) < 0:
                    quad = quad[::-1]
                zq = ground(quad) + PARKING_Z + MARK_Z
                marks.face([(*q, z) for q, z in zip(quad, zq)])
                n_bays += 1
                # a car in about a third of the stalls (the stall lies between this line and the next)
                c = (p0 + p1) / 2 + U * 1.25
                if rng_park.random() < 0.35 and in_poly((c + U * 1.2)[None], poly)[0]:
                    park_spots.append((*c, float(ground(c[None])[0]) + PARKING_Z, math.atan2(*(V * rng_park.choice((-1, 1)))[::-1]), 0))
    n_lots += 1

# cars parked at the kerb of residential streets in town, now and then, clear of junctions and way ends
for t, xy in ways:
    if t["highway"] not in ("residential", "living_street", "unclassified") or t.get("_fast"):
        continue
    pts = densify(xy, 1.0)
    if len(pts) < 30 or not in_town(pts[len(pts) // 2:len(pts) // 2 + 1])[0]:
        continue
    tan, nrm, along = frame(pts)
    w2 = width(t) / 2
    for s in (-1, 1):
        for i in range(12, len(pts) - 12, 7):
            if rng_park.random() > 0.03:  # rare: the user found the streets blocked with 0.14
                continue
            c = pts[i] + s * nrm[i] * (w2 - 1.05)
            if road_count(c[None])[0] >= 2 or road_count(pts[i][None])[0] >= 2 or not inside_terrain(c[None])[0]:
                continue
            d = tan[i] * s  # driving direction of that side
            park_spots.append((*c, float(ground_road(c[None])[0]) + CLASS_Z[t["highway"]], math.atan2(d[1], d[0]), 1))


# ---------------- materials ----------------
def paving_material(name, asset, uv_name, saturation=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nn = Nodes(m)
    uv = nn.uv(uv_name)
    hs = nn.new("ShaderNodeHueSaturation")
    hs.inputs["Saturation"].default_value = saturation
    nn.link(nn.tex(asset, "Diffuse", uv), hs.inputs["Color"])
    nn.link(hs.outputs["Color"], nn.bsdf.inputs["Base Color"])
    nn.link(nn.tex(asset, "Rough", uv, False), nn.bsdf.inputs["Roughness"])
    nn.normal(nn.tex(asset, "nor_gl", uv, False))
    return m


def paint_material():
    m = bpy.data.materials.new("NRW_RoadPaint")
    m.use_nodes = True
    nn = Nodes(m)
    noise = nn.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.5
    nn.link(nn.new("ShaderNodeTexCoord").outputs["Object"], noise.inputs["Vector"])
    nn.link(nn.mix(noise.outputs["Fac"], (0.85, 0.85, 0.82, 1), (0.55, 0.55, 0.53, 1)), nn.bsdf.inputs["Base Color"])
    nn.bsdf.inputs["Roughness"].default_value = 0.55
    return m


# ---------------- bridges: the terrain (bare earth) has the stream or road under them, so the deck runs straight
# from one abutment to the other, on a concrete slab with parapets ----------------
def oface(mb, quad, want, mat):
    """a quad wound so its normal points along `want`"""
    q = [np.asarray(p, float) for p in quad]
    if np.cross(q[1] - q[0], q[3] - q[0]) @ want < 0:
        q = q[::-1]
    mb.face(q, mat)


# the Autobahn's carriageways (centre points, half widths, road height) for the clearance of the decks over it
_mw = [(densify(xy, 3.0), width(t) / 2) for t, xy in ways if t["highway"] in MW_MAIN]
if _mw:
    RZ_LAYER = 1
    MW_P = np.vstack([p for p, _ in _mw])
    MW_W = np.concatenate([np.full(len(p), w) for p, w in _mw])
    MW_Z = ground_road(MW_P) + CLASS_Z["motorway"]
    RZ_LAYER = 0
MW_CLEAR, MW_RAMP = 5.0, 0.08  # clear height over the A4 (German standard 4.7 m), deck falling 8 % away from it
n_bridges = n_lifted = 0
for e in json.load(open(os.path.join(os.path.dirname(P), "osm_roads.json")))["elements"]:
    t = e.get("tags", {})
    if t.get("highway") not in ROAD_W or t.get("bridge") in (None, "no") or len(e.get("geometry", [])) < 2:
        continue
    g = e["geometry"]
    pts = densify(tm(np.array([p["lat"] for p in g]), np.array([p["lon"] for p in g])), 1.0)
    ins = inside_terrain(pts)
    if not ins.any():
        continue
    tan, nrm, along = frame(pts)
    RZ_LAYER = 1 if t["highway"] in MW_MAIN else 0
    z0, z1 = ground_road(pts[[0, -1]])
    RZ_LAYER = 0
    zd = np.maximum(z0 + (z1 - z0) * along / max(along[-1], 1e-6), ground(pts)) + CLASS_Z[t["highway"]]
    if _mw and t["highway"] not in MW_MAIN:
        # a deck crossing the Autobahn stands clear of it, whatever its ends say
        box_ = (MW_P[:, 0] > pts[:, 0].min() - 30) & (MW_P[:, 0] < pts[:, 0].max() + 30) & \
               (MW_P[:, 1] > pts[:, 1].min() - 30) & (MW_P[:, 1] < pts[:, 1].max() + 30)
        if box_.any():
            mp, mwid, mz = MW_P[box_], MW_W[box_], MW_Z[box_]
            dd = np.linalg.norm(pts[:, None, :] - mp[None, :, :], axis=2)
            k = dd.argmin(1)
            over = dd[np.arange(len(pts)), k] < mwid[k] + 1.5
            if over.any():
                need = mz[k][over] + MW_CLEAR + CLASS_Z[t["highway"]]
                dist_over = np.abs(along[:, None] - along[over][None, :])
                req = (need[None, :] - MW_RAMP * dist_over).max(1)
                if (req > zd).any():
                    zd = np.maximum(zd, req)
                    n_lifted += 1
    w2 = width(t) / 2 + 0.6
    roads.strip(pts + nrm * w2, pts - nrm * w2, zd, zd, along=along, keep=ins)
    for i in range(len(pts) - 1):
        if not (ins[i] and ins[i + 1]):
            continue
        for sgn in (1, -1):
            # a low concrete edge beam, and on it a railing: posts every 2 m, a top and a middle rail
            n3 = np.r_[nrm[i] * sgn, 0.0]
            a_in, b_in = pts[i] + nrm[i] * sgn * w2, pts[i + 1] + nrm[i + 1] * sgn * w2
            a_out, b_out = pts[i] + nrm[i] * sgn * (w2 + 0.35), pts[i + 1] + nrm[i + 1] * sgn * (w2 + 0.35)
            za, zb = zd[i], zd[i + 1]
            oface(walks, [(*a_in, za + 0.3), (*b_in, zb + 0.3), (*b_out, zb + 0.3), (*a_out, za + 0.3)], np.array((0, 0, 1.0)), 0)
            oface(walks, [(*a_in, za), (*b_in, zb), (*b_in, zb + 0.3), (*a_in, za + 0.3)], -n3, 0)
            oface(walks, [(*a_out, za - 0.9), (*b_out, zb - 0.9), (*b_out, zb + 0.3), (*a_out, za + 0.3)], n3, 0)
            m_in, m_out = a_in + (a_out - a_in) * 0.4, a_in + (a_out - a_in) * 0.6
            n_in, n_out = b_in + (b_out - b_in) * 0.4, b_in + (b_out - b_in) * 0.6
            for zr, hr in ((1.05, 0.06), (0.65, 0.04)):  # rails
                oface(walks, [(*m_in, za + zr), (*n_in, zb + zr), (*n_out, zb + zr), (*m_out, za + zr)], np.array((0, 0, 1.0)), 0)
                oface(walks, [(*m_out, za + zr - hr), (*n_out, zb + zr - hr), (*n_out, zb + zr), (*m_out, za + zr)], n3, 0)
                oface(walks, [(*m_in, za + zr - hr), (*n_in, zb + zr - hr), (*n_in, zb + zr), (*m_in, za + zr)], -n3, 0)
            if int(along[i] / 2.0) != int(along[i + 1] / 2.0):  # a post
                t3 = np.r_[tan[i], 0.0] * 0.04
                c = (m_in + m_out) / 2
                corners = [np.r_[c + nrm[i] * s1 * 0.04, 0.0] + t3 * s2 for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                for k in range(4):
                    p0, p1 = corners[k], corners[(k + 1) % 4]
                    out = (p0 + p1) / 2 - np.r_[c, 0.0]
                    oface(walks, [(*p0[:2], za + 0.3), (*p1[:2], za + 0.3), (*p1[:2], za + 1.05), (*p0[:2], za + 1.05)], out, 0)
        # a pier every 15 m where the deck stands high over the ground
        if int(along[i] / 15.0) != int(along[i + 1] / 15.0):
            gz_ = float(ground(pts[i][None])[0])
            if zd[i] - gz_ > 2.0:
                t2, n2_ = tan[i] * 0.45, nrm[i] * w2 * 0.6
                cs = [pts[i] + t2 * s1 + n2_ * s2 for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                for k in range(4):
                    p0, p1 = cs[k], cs[(k + 1) % 4]
                    out = np.r_[(p0 + p1) / 2 - pts[i], 0.0]
                    oface(walks, [(*p0, gz_ - 0.5), (*p1, gz_ - 0.5), (*p1, zd[i] - 0.9), (*p0, zd[i] - 0.9)], out, 0)
        l_, r_ = pts + nrm * (w2 + 0.3), pts - nrm * (w2 + 0.3)
        oface(walks, [(*l_[i], zd[i] - 0.9), (*l_[i + 1], zd[i + 1] - 0.9), (*r_[i + 1], zd[i + 1] - 0.9), (*r_[i], zd[i] - 0.9)],
              np.array((0, 0, -1.0)), 0)
    n_bridges += 1
print("BRIDGES", n_bridges, "lifted over the Autobahn", n_lifted)

# parked cars for Unity (MapParkedCars): a point per car with its heading, kept as the NRW_ParkSpots object
if park_spots:
    ps = np.array(park_spots, np.float32)
    pme = bpy.data.meshes.new("NRW_ParkSpots")
    pme.vertices.add(len(ps))
    pme.vertices.foreach_set("co", ps[:, :3].ravel())
    pme.attributes.new("heading", "FLOAT", "POINT").data.foreach_set("value", ps[:, 3])
    pme.attributes.new("kerb", "INT", "POINT").data.foreach_set("value", ps[:, 4].astype(np.int32))
    col.objects.link(bpy.data.objects.new("NRW_ParkSpots", pme))
print("PARKED cars", len(park_spots))

asphalt = paving_material("NRW_Asphalt", "asphalt_02", "RoadUV")
roads.build("NRW_Roads", "RoadUV", [asphalt])
lots.build("NRW_Parking", "RoadUV", [asphalt])
marks.build("NRW_RoadMarkings", "RoadUV", [paint_material()])
walks.build("NRW_Sidewalks", "WallUV", [paving_material("NRW_Curb", "brushed_concrete", "WallUV", 0.15),
                                         paving_material("NRW_Pavers", "concrete_pavers_02", "WallUV", 0.15)])
gutters.build("NRW_Gutters", "RoadUV", [paving_material("NRW_Gutter", "concrete_pavers_02", "RoadUV", 0.0)])


def ground_material(photo_img=None, veg_img=None, name="NRW_GroundDetail"):
    """orthophoto colour, with grass or concrete detail texture on top so it is not a flat picture up close"""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nn = Nodes(m)
    uv = nn.uv("UVMap")
    photo = nn.new("ShaderNodeTexImage", interpolation="Cubic", image=photo_img or ortho, extension="EXTEND")
    nn.link(uv, photo.inputs["Vector"])
    gamma = nn.new("ShaderNodeGamma")
    gamma.inputs["Gamma"].default_value = 1.2
    nn.link(photo.outputs["Color"], gamma.inputs["Color"])
    veg = nn.new("ShaderNodeTexImage", extension="EXTEND",
                 image=veg_img or bpy.data.images.load(os.path.join(P, "vegmask.png")))
    veg.image.colorspace_settings.name = "Non-Color"
    nn.link(uv, veg.inputs["Vector"])
    obj = nn.new("ShaderNodeTexCoord").outputs["Object"]

    def detail(asset):
        img = bpy.data.images.load(os.path.join(TEX, asset + "_Diffuse.jpg"), check_existing=True)
        px = np.empty(img.size[0] * img.size[1] * 4, np.float32)
        img.pixels.foreach_get(px)
        rgb = px.reshape(-1, 4)[::97, :3]
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        mean = float((lin @ np.array([0.2126, 0.7152, 0.0722])).mean())
        bw = nn.new("ShaderNodeRGBToBW")
        nn.link(nn.tex(asset, "Diffuse", obj), bw.inputs[0])
        return nn.math("DIVIDE", bw.outputs[0], max(mean, 1e-3))

    d = nn.mix(veg.outputs["Color"], nn.math("POWER", detail("brushed_concrete"), 0.5), detail("leafy_grass"))
    d = nn.mix(0.65, (1, 1, 1, 1), d)
    col_ = nn.new("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    col_.inputs[0].default_value = 1.0
    nn.link(gamma.outputs["Color"], col_.inputs[6])
    nn.link(d, col_.inputs[7])
    nn.link(col_.outputs[2], nn.bsdf.inputs["Base Color"])
    nn.bsdf.inputs["Roughness"].default_value = 0.9
    nn.normal(nn.mix(veg.outputs["Color"], (0.5, 0.5, 1.0, 1), nn.tex("leafy_grass", "nor_gl", obj, False)), 0.6)
    return m


if REGION:
    for i, (e, n) in enumerate(TILES):
        terrain.data.materials[i] = ground_material(tile_image("ortho", f"{e}_{n}"), tile_image("vegmask", f"{e}_{n}", False),
                                                    f"NRW_GroundDetail_{e}_{n}")
else:
    terrain.data.materials[0] = ground_material()
print("STREETS ways", len(ways), "through roads", n_fast, "road_faces", len(roads.f), "marks", len(marks.f), "zebras", n_zebra,
      "sidewalk_sides", n_side, "lots", n_lots, "bays", n_bays)
