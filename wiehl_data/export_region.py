"""Export the region (wiehl_region.blend) for Unity in 500 m cells, so Unity only draws what is near and in view.
Per cell: Ground (terrain) + Roofs on the cell's own orthophoto, Walls with every facade detail packed into UVs and
vertex colours for the Dexstand/Facade shader (no baking: 17 000 buildings would need a thousand atlases), Roads,
Marks, Walks, Barriers, Rail, Props (lamps, bus stops, signs, the hand-made bus station and Rewe), Eaves (roof
overhangs), Details (door steps, porch roofs, balconies; styles and all of these by architecture zone, zones.py) and low-poly Trees,
plus hidden *_col collision. One FBX per 1 km tile. The origin is moved to the bus station on the ground.
Run: blender -b wiehl_region.blend --python export_region.py -- OUT_DIR PREPARED_DIR [detail]
(OUT_DIR must already hold Textures/ and cells.json from region_data/unity_textures_region.py)"""
import bpy, json, math, os, re, sys
import numpy as np
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:]
OUT, P = ARGS[0], ARGS[1]
MODE = ARGS[2] if len(ARGS) > 2 else "all"
ONLY_DETAIL = MODE == "detail"
# "corridor": the towns in full, and between them only the road from the bus station to Brüchermühle
# with CORRIDOR_M on either side
CORRIDOR = MODE == "corridor"
CORRIDOR_M = 100.0
ROUTE_TO = (6236.5, -2276.7)  # Kölner Straße (L 336) in Brüchermühle, local metres
TEXDIR = os.path.join(OUT, "Textures")
SRC_TEX = r"A:\blender_work\textures"
meta = json.load(open(os.path.join(P, "meta.json")))
UX0, UY0, UX1, UY1 = meta["utm_bbox"]
AFF = np.array(meta["affine"])
INV = np.linalg.inv(AFF[:2])
CELLS = json.load(open(os.path.join(OUT, "cells.json")))
TEX_M = {k: v[0] / 1000 for k, v in json.load(open(os.path.join(SRC_TEX, "dims.json"))).items()}
WANT_TILES = set(meta["detail"]) if ONLY_DETAIL else set(meta["ortho_m_per_px"])
# "part=K/N" after the mode: every N-th tile from the K-th, so a big export fits in memory next to a running Unity
PART = next((a[5:] for a in ARGS[3:] if a.startswith("part=")), None)
if PART:
    k_, n_ = map(int, PART.split("/"))
    WANT_TILES = set(sorted(WANT_TILES)[k_::n_])
# only some tiles again (env ONLY_TILES="393_5647,394_5648"): the old region's A4 tiles, 2026-09-27
if os.environ.get("ONLY_TILES"):
    WANT_TILES &= set(os.environ["ONLY_TILES"].split(","))
sc = bpy.context.scene
dg = bpy.context.evaluated_depsgraph_get()

# ---------------- origin: the bus station, on the ground ----------------
R = 6378137.0


def tm(lat, lon):
    la, lo = math.radians(lat), math.radians(lon - sc["lon"])
    b = math.sin(lo) * math.cos(la)
    return 0.5 * R * math.log((1 + b) / (1 - b)), R * (math.atan(math.tan(la) / math.cos(lo)) - math.radians(sc["lat"]))


ter = bpy.data.objects["NRW_Terrain"]
if meta.get("chunk"):
    # an A4 corridor chunk: Wiehl's bus station is not in it, so the main region's origin comes with the chunk
    ORIGIN = np.array(meta["chunk"]["origin"], np.float32)
else:
    osm = json.load(open(os.path.join(os.path.dirname(P), "osm_extra.json"), encoding="utf-8"))["elements"]
    bus = [e for e in osm if e.get("tags", {}).get("amenity") == "bus_station" and "geometry" in e]
    bus = min(bus, key=lambda e: np.hypot(*np.mean([tm(p["lat"], p["lon"]) for p in e["geometry"]], axis=0)))
    sx, sy = np.mean([tm(p["lat"], p["lon"]) for p in bus["geometry"]], axis=0)
    ok, hit, *_ = ter.ray_cast((sx, sy, 5000.0), (0, 0, -1), depsgraph=dg)
    ORIGIN = np.array((sx, sy, hit.z if ok else 0.0), np.float32)
print("ORIGIN (bus station)", ORIGIN.round(2), flush=True)


def to_d(xy):
    return (np.asarray(xy, np.float64) - AFF[2]) @ INV


def cell_keys(xy):
    d = to_d(xy)
    ci, cj = np.floor(d[:, 0] / 500).astype(int), np.floor(d[:, 1] / 500).astype(int)
    e, n = UX0 // 1000 + ci // 2, UY0 // 1000 + cj // 2
    return np.array([f"{a}_{b}_{c}" for a, b, c in zip(e, n, ci % 2 + 2 * (cj % 2))])


def tile_of_cell(ck):
    return "_".join(ck.split("_")[:2])


# ---------------- the road between the towns (corridor mode) ----------------
def route(start, goal):
    """shortest drive over OSM's bigger roads, as a local-metre polyline"""
    import heapq
    cost = {"motorway": 1.0, "trunk": 1.0, "primary": 1.0, "secondary": 1.0, "tertiary": 1.2, "primary_link": 1.0,
            "secondary_link": 1.0, "tertiary_link": 1.2, "unclassified": 2.0, "residential": 2.5}
    pts, index, adj = [], {}, {}

    def node(p):
        k = (round(p["lat"], 7), round(p["lon"], 7))
        if k not in index:
            index[k] = len(pts)
            pts.append(tm(p["lat"], p["lon"]))
        return index[k]

    for e in json.load(open(os.path.join(os.path.dirname(P), "osm_roads.json"), encoding="utf-8"))["elements"]:
        k = cost.get(e.get("tags", {}).get("highway"))
        if k is None or len(e.get("geometry", [])) < 2:
            continue
        ids = [node(p) for p in e["geometry"]]
        for a, b in zip(ids[:-1], ids[1:]):
            w = k * float(np.hypot(*(np.array(pts[a]) - np.array(pts[b]))))
            adj.setdefault(a, []).append((b, w))
            adj.setdefault(b, []).append((a, w))
    P2 = np.array(pts)
    s = int(np.argmin(np.hypot(*(P2 - start).T)))
    g = int(np.argmin(np.hypot(*(P2 - goal).T)))
    dist, prev, heap = {s: 0.0}, {}, [(0.0, s)]
    while heap:
        d, a = heapq.heappop(heap)
        if a == g:
            break
        if d > dist.get(a, 1e18):
            continue
        for b, w in adj.get(a, ()):
            if d + w < dist.get(b, 1e18):
                dist[b], prev[b] = d + w, a
                heapq.heappush(heap, (d + w, b))
    path = [g]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return P2[path[::-1]]


corridor_mask = None
if CORRIDOR:
    line = route(ORIGIN[:2].astype(np.float64), np.array(ROUTE_TO))
    seg = np.diff(line, axis=0)
    dense = np.vstack([a + (b - a) * t[:, None] for a, b in zip(line[:-1], line[1:])
                       for t in [np.linspace(0, 1, max(2, int(np.hypot(*(b - a)) / 5)))]])
    RES = 5.0
    corridor_mask = np.zeros((int((UY1 - UY0) / RES), int((UX1 - UX0) / RES)), bool)
    rr = int(CORRIDOR_M / RES)
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    disc = xx ** 2 + yy ** 2 <= rr ** 2
    for dx_, dy_ in ((np.asarray(dense) - AFF[2]) @ INV):
        c, r = int(dx_ / RES), int(corridor_mask.shape[0] - dy_ / RES)
        r0, r1, c0, c1 = max(0, r - rr), min(corridor_mask.shape[0], r + rr + 1), max(0, c - rr), min(corridor_mask.shape[1], c + rr + 1)
        if r0 < r1 and c0 < c1:
            corridor_mask[r0:r1, c0:c1] |= disc[r0 - (r - rr):r1 - (r - rr), c0 - (c - rr):c1 - (c - rr)]
    print("ROUTE", len(line), "points,", round(float(np.hypot(*seg.T).sum()) / 1000, 1), "km", flush=True)


def keep(xy):
    """which points of a tile to export: everything in the towns; out of town only the corridor"""
    xy = np.asarray(xy)[:, :2]
    if not CORRIDOR:
        return np.ones(len(xy), bool)
    d = (xy.astype(np.float64) - AFF[2]) @ INV
    e = np.floor((d[:, 0] + UX0) / 1000).astype(int)
    n = np.floor((d[:, 1] + UY0) / 1000).astype(int)
    town = np.isin(e * 10000 + n, [int(k[:3]) * 10000 + int(k[4:]) for k in meta["detail"]])
    c = np.clip((d[:, 0] / 5.0).astype(int), 0, corridor_mask.shape[1] - 1)
    r = np.clip((corridor_mask.shape[0] - d[:, 1] / 5.0).astype(int), 0, corridor_mask.shape[0] - 1)
    return town | corridor_mask[r, c]


# ---------------- materials ----------------
def image(path, non_color=False):
    img = bpy.data.images.load(path, check_existing=True)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def export_material(name, color=None, rgb=(0.8, 0.8, 0.8), normal=None, rough=0.85, alpha=False, alpha_value=1.0, metal=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in [n for n in nt.nodes if n.type not in {"BSDF_PRINCIPLED", "OUTPUT_MATERIAL"}]:
        nt.nodes.remove(n)
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Alpha"].default_value = alpha_value
    if color:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = image(color)
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
        if alpha:
            nt.links.new(t.outputs["Alpha"], bsdf.inputs["Alpha"])
    if normal:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = image(normal, True)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return m


_plain = {}


def plain_copy(m):
    """a Blender material of the props, signs and hand-made buildings as a plain export material"""
    if m is None:
        return export_material("Default")
    if m.name not in _plain:
        _plain[m.name] = _make_plain(m)
    return _plain[m.name]


def _make_plain(m):
    name = re.sub(r"^NRW_", "", m.name)
    b = m.node_tree.nodes.get("Principled BSDF") if m.use_nodes and m.node_tree else None
    rgb = tuple(b.inputs["Base Color"].default_value)[:3] if b else tuple(m.diffuse_color)[:3]
    tex = next((n.image for n in (m.node_tree.nodes if m.node_tree else []) if n.type == "TEX_IMAGE" and n.image), None)
    path = None
    if tex is not None and tex.filepath:
        path = os.path.join(TEXDIR, os.path.basename(bpy.path.abspath(tex.filepath)))
        if not os.path.exists(path):
            path = None
    alpha = b.inputs["Alpha"].default_value if b else 1.0
    cutout = path is not None and path.endswith(".png")
    return export_material(("Cutout_" if cutout else "Glass_" if alpha < 0.99 else "") + name, path, rgb,
                           rough=b.inputs["Roughness"].default_value if b else 0.6,
                           metal=b.inputs["Metallic"].default_value if b else 0.0, alpha=cutout, alpha_value=alpha)


def ortho_mat(ck):
    return export_material("Ortho_" + ck, os.path.join(TEXDIR, f"ortho_{ck}.jpg"), rough=0.95)


def roof_ortho_mat(ck):
    """the same orthophoto, named apart so Unity lays roof tiles over it instead of grass (Dexstand/OrthoDetail)"""
    return export_material("RoofOrtho_" + ck, os.path.join(TEXDIR, f"ortho_{ck}.jpg"), rough=0.95)


# ---------------- mesh splitting (numpy, no bmesh: the terrain alone has 4.7 M vertices) ----------------
def arrays(me):
    n = len(me.polygons)
    A = {}
    for name, dtype in (("loop_start", np.int32), ("loop_total", np.int32), ("material_index", np.int32)):
        a = np.empty(n, dtype); me.polygons.foreach_get(name, a); A[name] = a
    c = np.empty(n * 3, np.float32); me.polygons.foreach_get("center", c); A["center"] = c.reshape(-1, 3)
    lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv); A["lv"] = lv
    co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co); A["co"] = co.reshape(-1, 3)
    sm = np.empty(n, bool); me.polygons.foreach_get("use_smooth", sm); A["smooth"] = sm
    return A


def loop_uv(me, name):
    a = np.empty(len(me.loops) * 2, np.float32); me.uv_layers[name].data.foreach_get("uv", a)
    return a.reshape(-1, 2)


def face_attr(me, name, width):
    a = np.empty(len(me.polygons) * width, np.float32)
    me.attributes[name].data.foreach_get("vector" if width == 3 else "color" if width == 4 else "value", a)
    return a.reshape(-1, width) if width > 1 else a


def loops_of(A, faces):
    ls, lt = A["loop_start"][faces], A["loop_total"][faces]
    return np.repeat(ls - np.r_[0, np.cumsum(lt)[:-1]], lt) + np.arange(lt.sum()), lt


exp = bpy.data.collections.new("EXPORT")
sc.collection.children.link(exp)
by_tile = {}


def emit(name, ck, A, faces, mats, mat_idx=None, uvs=(), colors=None, smooth=None):
    """a new object from `faces` of the mesh described by A; uvs = [(name, per-loop array of the source)]"""
    faces = np.asarray(faces)
    if not len(faces):
        return None
    li, lt = loops_of(A, faces)
    uniq, inv = np.unique(A["lv"][li], return_inverse=True)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(uniq)); me.vertices.foreach_set("co", (A["co"][uniq] - ORIGIN).ravel())
    me.loops.add(len(li)); me.loops.foreach_set("vertex_index", inv.astype(np.int32))
    me.polygons.add(len(faces)); me.polygons.foreach_set("loop_start", np.r_[0, np.cumsum(lt)[:-1]].astype(np.int32))
    me.update(calc_edges=True)
    for uname, arr in uvs:
        me.uv_layers.new(name=uname).data.foreach_set("uv", arr[li].astype(np.float32).ravel())
    for m in mats:
        me.materials.append(m)
    if mat_idx is not None:
        me.polygons.foreach_set("material_index", mat_idx[faces].astype(np.int32))
    if colors is not None:
        me.color_attributes.new("Col", "FLOAT_COLOR", "CORNER").data.foreach_set("color", colors[li].astype(np.float32).ravel())
    me.polygons.foreach_set("use_smooth", (A["smooth"][faces] if smooth is None else np.full(len(faces), smooth)))
    ob = bpy.data.objects.new(name, me)
    exp.objects.link(ob)
    by_tile.setdefault(tile_of_cell(ck), []).append(ob)
    return ob


def wanted(ck):
    return tile_of_cell(ck) in WANT_TILES


def cell_uv(tile_uv, ck):
    """tile-relative UV (0..1 over the km) -> UV on the cell's cropped orthophoto"""
    c = CELLS[ck]
    x, y = tile_uv[:, 0] * 1000, tile_uv[:, 1] * 1000
    return np.c_[(x - c["x0"]) / (c["x1"] - c["x0"]), (y - c["y0"]) / (c["y1"] - c["y0"])]


def quadrant(tile, pts):
    e, n = map(int, tile.split("_"))
    d = to_d(pts[:, :2])
    qx = np.clip(np.floor((d[:, 0] - (e * 1000 - UX0)) / 500), 0, 1).astype(int)
    qy = np.clip(np.floor((d[:, 1] - (n * 1000 - UY0)) / 500), 0, 1).astype(int)
    return qx + 2 * qy


def ortho_split(ob, label, collision_label=None):
    """terrain or roofs: faces grouped by the orthophoto tile of their material, then by quarter"""
    me = ob.data
    A = arrays(me)
    uv = loop_uv(me, "UVMap")
    for slot, m in enumerate(me.materials):
        mt = re.search(r"_(\d{3}_\d{4})$", m.name) if m else None
        if not mt or mt.group(1) not in WANT_TILES:
            continue
        tile = mt.group(1)
        faces = np.nonzero(A["material_index"] == slot)[0]
        faces = faces[keep(A["center"][faces])]
        q = quadrant(tile, A["center"][faces])
        for qi in range(4):
            ck = f"{tile}_{qi}"
            f = faces[q == qi]
            if not len(f):
                continue
            li, _ = loops_of(A, f)
            cuv = np.zeros_like(uv)
            cuv[li] = cell_uv(uv[li], ck)
            emit(f"{label}_{ck}", ck, A, f, [ortho_mat(ck)], uvs=[("UVMap", cuv)])
            if collision_label:
                emit(f"{collision_label}_{ck}_col", ck, A, f, [])
    print("SPLIT", label, flush=True)


# ---------------- ground ----------------
ortho_split(ter, "Ground", "GroundCol")


def ground_z(xy):
    """terrain height under each point (nan off the terrain); the terrain lives until the facade extras are placed"""
    out = np.full(len(xy), np.nan)
    for i, (x, y) in enumerate(np.asarray(xy, np.float64)):
        ok, hit, *_ = ter.ray_cast((x, y, 5000.0), (0, 0, -1), depsgraph=dg)
        if ok:
            out[i] = hit.z
    return out

# ---------------- buildings: roofs on the orthophoto, walls for the facade shader ----------------
bld = bpy.data.objects["NRW_Buildings"]
bme = bld.data
BA = arrays(bme)
bid = np.empty(len(bme.polygons), np.int32); bme.attributes["bid"].data.foreach_get("value", bid)
n_b = int(bid.max()) + 1
bcen = np.zeros((n_b, 2)); np.add.at(bcen, bid, BA["center"][:, :2]); bcen /= np.maximum(np.bincount(bid, minlength=n_b), 1)[:, None]
bcell = cell_keys(bcen)
roof_slots = [i for i, m in enumerate(bme.materials) if m and m.name.startswith("NRW_Roof_")]
ruv = loop_uv(bme, "UVMap")
for slot in roof_slots:
    tile = re.search(r"_(\d{3}_\d{4})$", bme.materials[slot].name).group(1)
    if tile not in WANT_TILES:
        continue
    faces = np.nonzero(BA["material_index"] == slot)[0]
    faces = faces[keep(BA["center"][faces])]
    q = quadrant(tile, BA["center"][faces])
    for qi in range(4):
        ck = f"{tile}_{qi}"
        f = faces[q == qi]
        if len(f):
            li, _ = loops_of(BA, f)
            cuv = np.zeros_like(ruv)
            cuv[li] = cell_uv(ruv[li], ck)
            emit(f"Roofs_{ck}", ck, BA, f, [roof_ortho_mat(ck)], uvs=[("UVMap", cuv)], smooth=False)

# walls: facade data per face -> UV channels + vertex colour (see Dexstand/Facade in Unity)
geo, door, kind, look = (face_attr(bme, n, 3) for n in ("w_geo", "w_door", "w_kind", "w_look"))
gcol = face_attr(bme, "gcol", 4)
scol = face_attr(bme, "scol", 1) if "scol" in bme.attributes else np.zeros(len(bme.polygons), np.float32)

# house styles by architecture zone (zones.py): 0 plaster, 1 Bergisch slate (white frames, green shutters),
# 2 half-timber, 3 brick; "modern" is plaster in greys with anthracite frames. The old town keeps its Google colour
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "zones.py"), encoding="utf-8").read())
FALLBACK = np.array([0.80, 0.76, 0.68])  # google_colors.py stores this one as is, unconverted
bk = np.full(n_b, -1.0); bk[bid] = kind[:, 0]
bcol = np.zeros((n_b, 3)); bcol[bid] = gcol[:, :3]
bbrick = np.zeros(n_b); bbrick[bid] = look[:, 1]
has_google = np.abs(bcol - FALLBACK).sum(1) > 0.02
house = (bk == 0) | (bk == 8)
fresh = house & ~has_google
bh = np.zeros(n_b); bh[bid] = door[:, 2]
zone = assign_zones(bcen)
rng = np.random.default_rng(11)
style = np.zeros(n_b)
newcol = bcol.copy()
win = np.tile((1.5, 1.6), (n_b, 1))           # window width, height (m); business buildings: paired office windows
spacing = np.full(n_b, 3.8)                    # window bay (m)
smode = np.zeros(n_b)                          # slate: 0 all over 1 gable 2 upper floors 3 ground floor 4 west side
plinth = np.full(n_b, 0.5)
frame = np.zeros(n_b)                          # 0 white 1 light grey 2 anthracite 3 grey
shut = np.zeros(n_b, bool)
roll = np.zeros(n_b, bool)
overhang = np.zeros(n_b)
for zi, zid in enumerate(ZONE_IDS):
    Z = ZONES[zid]
    m = np.nonzero(zone == zi)[0]
    n = len(m)
    if not n:
        continue
    st = rng.choice(5, n, p=np.array(Z["styles"]) / sum(Z["styles"]))
    st[(st == 2) & (bh[m] >= 10.0)] = 0  # timber belongs to one- and two-storey houses, not to blocks of flats
    sm = rng.choice(5, n, p=np.array(SLATE_MODES) / sum(SLATE_MODES))
    sm[(st == 1) & (bh[m] >= 11.0) & (sm == 0)] = 2  # blocks of flats: slate on the upper floors, not all over
    plaster_c = pick(rng, Z["plaster"], n)
    zc = plaster_c.copy()                # slate houses keep a plaster colour for their white parts
    zc[st == 4] = pick(rng, MODERN, n)[st == 4]
    zc[st == 2] = lin(TIMBER_FILL)
    f_ = fresh[m]
    style[m[f_]] = np.where(st == 4, 0, st)[f_]
    newcol[m[f_]] = zc[f_]
    o_ = ~house[m] & ~has_google[m]
    newcol[m[o_]] = plaster_c[o_]
    h_ = house[m]
    old = (st == 2) | ((st == 1) & (sm == 0))  # half-timber and all-slate houses: small upright windows
    wt = rng.choice(4, n, p=np.array(Z["wt"]) / sum(Z["wt"]))
    wt[old] = 0
    w, h = WTYPES[wt].T * rng.uniform(0.92, 1.08, (2, n))
    sp = np.maximum(rng.uniform(*Z["bay"], n), w + 1.4)
    sp[old] = np.minimum(sp[old], 3.4)
    smode[m[f_]] = sm[f_]
    fr = rng.choice(4, n, p=np.array(Z["frames"]) / sum(Z["frames"]))
    fr[st == 4] = np.where(rng.random(n) < 0.7, 2, fr)[st == 4]
    fr[old] = 0
    win[m[h_]] = np.c_[w, h][h_]
    spacing[m[h_]] = sp[h_]
    frame[m[h_]] = fr[h_]
    shut[m[h_]] = ((rng.random(n) < Z["shutters"]) & ~((st == 1) & (sm == 0)))[h_]  # all-slate ones always have them
    roll[m[h_]] = ((rng.random(n) < Z["roller"]) & ~old)[h_]
    plinth[m[h_]] = rng.uniform(*Z["plinth"], n)[h_]
    overhang[m] = rng.uniform(*Z["overhang"], n)
lum = bcol @ np.array([0.2126, 0.7152, 0.0722])
style[house & has_google & (lum < 0.05)] = 1
style[house & has_google & (bbrick > 0.5)] = 3
# landmarks.py fixes the style and colour of the buildings modelled from photos
landmark = np.zeros(n_b, bool)
if "fstyle" in bme.attributes:
    fsty = face_attr(bme, "fstyle", 1)
    fb = bid[fsty >= 0]
    style[fb] = fsty[fsty >= 0]
    newcol[fb] = bcol[fb]
    landmark[fb] = True
# no pink houses (the user, 2026-09-25: "красный максимум"): pink to purple hues turn a muted red
cmax, cmin = newcol.max(1), newcol.min(1)
sat = np.where(cmax > 1e-4, (cmax - cmin) / np.maximum(cmax, 1e-4), 0)
pink = (newcol[:, 0] >= cmax - 1e-6) & (newcol[:, 2] > newcol[:, 1] + 0.1 * (cmax - cmin)) & (sat > 0.12) & ~landmark
lum_p = newcol[pink] @ np.array([0.2126, 0.7152, 0.0722])
newcol[pink] = np.c_[lum_p * 2.2, lum_p * 0.55, lum_p * 0.45].clip(0, 0.7)
print("PINK to red", int(pink.sum()), flush=True)
print("STYLES slate", int((style == 1).sum()), "timber", int((style == 2).sum()), "brick", int((style == 3).sum()),
      "plaster", int(((style == 0) & house).sum()), flush=True)
print("ZONES", {z: int((zone[house] == i).sum()) for i, z in enumerate(ZONE_IDS)}, flush=True)

# ---------------- facades: shop fronts in the centres, raised doors with steps, porch roofs, balconies ----------------
bkeep = keep(bcen)
bwant = bkeep & np.array([wanted(c) for c in bcell])
fnrm = np.empty(len(bme.polygons) * 3, np.float32); bme.polygons.foreach_get("normal", fnrm); fnrm = fnrm.reshape(-1, 3)
wuv = loop_uv(bme, "WallUV")
wallf = np.nonzero((BA["material_index"] == 1) & (np.abs(fnrm[:, 2]) < 0.3))[0]
wn = fnrm[wallf, :2] / np.maximum(np.linalg.norm(fnrm[wallf, :2], axis=1), 1e-6)[:, None]
fkey = np.c_[bid[wallf], np.round(np.arctan2(wn[:, 1], wn[:, 0]) * 30), np.round((BA["center"][wallf, :2] * wn).sum(1) * 2)]
_, first, fac = np.unique(fkey, axis=0, return_index=True, return_inverse=True)
fac = fac.reshape(-1)
F0 = wallf[first]                                   # one face per facade
fb_ = bid[F0]
fn_ = wn[first]
ftan = np.c_[-fn_[:, 1], fn_[:, 0]]                 # +u along the facade (Z x normal, as build_blend's WallUV)
l0 = BA["loop_start"][F0]
fp0 = BA["co"][BA["lv"][l0]]
fu0, fv0 = wuv[l0, 0], wuv[l0, 1]
fzmin = fp0[:, 2] - fv0                             # the building's base
fwlen, fwbot, fweave = geo[F0, 0], geo[F0, 1], geo[F0, 2]
fdoor_u, fdoor_t = door[F0, 0], np.round(door[F0, 1]).astype(int)


def fworld(i, u, w=0.0):
    """xy on facade i at u along it, w out of the wall"""
    return fp0[i, :2] + ftan[i] * np.asarray(u - fu0[i])[..., None] + fn_[i] * np.asarray(w)[..., None]


fmid = fworld(np.arange(len(F0)), fwlen / 2)
ridx, rdist = np.full(len(F0), -1, np.int64), np.full(len(F0), np.inf)
fw_ = np.nonzero(bwant[fb_])[0]  # only this part's tiles
ridx[fw_], rdist[fw_] = ROADS.nearest(fmid[fw_], reach=1)
to_road = np.where(ridx[:, None] >= 0, ROAD_PTS[np.maximum(ridx, 0)] - fmid, 0)
facing = (to_road * fn_).sum(1) / np.maximum(np.hypot(*to_road.T), 1e-6)
facing[ridx < 0] = 0
rdist[ridx < 0] = 1e9

# shop fronts: the facade of a 2+ storey house in a centre that faces its street
shop_b = np.nonzero(bwant & house & ~landmark & (bh >= 6.5) & (rng.random(n_b) < np.array([ZONES[z]["shop"] for z in ZONE_IDS])[zone]))[0]
ok_f = (facing > 0.4) & (rdist < 30) & (fwlen >= 3) & (fwbot < 1.5)
shop_fac = []
for b in shop_b:
    cand = np.nonzero((fb_ == b) & ok_f)[0]
    if len(cand):
        shop_fac.append(cand[np.argmax(facing[cand])])
shop_face = np.isin(fac, shop_fac)
kind[wallf[shop_face], 1] = 1.0
door[wallf[shop_face & (np.round(door[wallf, 1]) == 1)], 1] = 0.0
fdoor_t[shop_fac] = 0

# doors stand on the ground in front of them; a house door a few steps above it
sbox = {}  # cell -> [verts, faces, material]


def add_box(ck, i, u0, u1, w0, w1, z0, z1, mat):
    """a box against facade i: u along it, w out of the wall, z absolute"""
    t3, n3 = np.r_[ftan[i], 0.0], np.r_[fn_[i], 0.0]
    o = np.r_[fp0[i, :2] - ftan[i] * fu0[i], 0.0]
    c = np.array([o + t3 * u + n3 * w + (0, 0, z) for z in (z0, z1) for w in (w0, w1) for u in (u0, u1)])
    v, f, m = sbox.setdefault(ck, ([], [], []))
    base_i = 8 * len(v)
    v.append(c)
    for q in ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)):
        f.append(tuple(base_i + j for j in q))
        m.append(mat)


S_STONE, S_SLAB, S_RAIL, S_CANOPY, S_BIN, S_LID0 = range(6)  # S_LID0..+3: black, blue, brown, yellow lids
n_bins = 0
drive_cars = []  # x, y, z, heading of cars parked on drives (this part's tiles), for Unity's MapParkedCars
fbase = np.zeros(len(F0))
dfac = np.nonzero((fdoor_t > 0) & bwant[fb_])[0]
dpt = fworld(dfac, fdoor_u[dfac], 0.8)
gz = ground_z(dpt)
zone_f = zone[fb_]
n_steps = n_canopy = 0
for i, g in zip(dfac, gz):
    b = fb_[i]
    bottom = fzmin[i] + fwbot[i]
    if np.isnan(g):
        continue
    top_room = fweave[i] - fwbot[i] - 2.5
    rise = max(0.0, g - bottom)
    Z = ZONES[ZONE_IDS[zone_f[i]]]
    k = 0
    if fdoor_t[i] == 1 and house[b] and not landmark[b]:
        k = int(rng.integers(Z["steps"][0], Z["steps"][1] + 1))
        k = max(0, min(k, int((top_room - rise) / 0.17)))
    fbase[i] = float(np.clip(rise + 0.17 * k, 0.0, max(top_room, 0.0)))
    if fdoor_t[i] == 2 and rng.random() < 0.55:  # a car on the drive, nose to the garage door
        c2 = fworld(i, fdoor_u[i], 3.0)
        drive_cars.append((c2[0], c2[1], g, math.atan2(-fn_[i][1], -fn_[i][0])))
    if fdoor_t[i] != 1 or not house[b] or landmark[b]:
        continue
    ck, u = bcell[b], fdoor_u[i]
    zt = bottom + fbase[i]
    if k:
        add_box(ck, i, u - 0.8, u + 0.8, 0.0, 1.0, g - 0.3, zt, S_STONE)      # landing at the door
        for s in range(1, k):
            add_box(ck, i, u - 0.7, u + 0.7, 1.0 + 0.3 * (s - 1), 1.0 + 0.3 * s, g - 0.3, zt - 0.17 * s, S_STONE)
        n_steps += 1
    if rng.random() < Z["canopy"]:
        add_box(ck, i, u - 0.95, u + 0.95, 0.0, 1.1, zt + 2.45, zt + 2.55, S_CANOPY)
        n_canopy += 1
    # wheelie bins beside the door: grey bodies, lids for rest waste, paper, bio and packaging
    if rng.random() < 0.65:
        side = rng.choice((-1, 1))
        u0 = u + side * 1.7  # clear of the door and its steps
        for kb, lid in enumerate(rng.choice(4, int(rng.integers(2, 4)), replace=False)):
            ub = u0 + side * kb * 0.66
            if ub < 0.4 or ub > fwlen[i] - 0.4:
                break
            add_box(ck, i, ub - 0.29, ub + 0.29, 0.25, 0.97, g - 0.05, g + 1.0, S_BIN)
            add_box(ck, i, ub - 0.31, ub + 0.31, 0.23, 0.99, g + 1.0, g + 1.07, S_LID0 + int(lid))
            n_bins += 1
fcode = np.round(fbase / 0.05)
door[wallf, 1] += 4 * fcode[fac]

# balconies: a stack per bay on the chosen side of flats, one on the first floor of houses
n_balc = 0
bal_share = np.array([[ZONES[z]["balc"][0], ZONES[z]["balc"][1]] for z in ZONE_IDS])
flat = bh >= 8.5
bal_b = np.nonzero(bwant & house & ~landmark & (rng.random(n_b) < bal_share[zone, flat.astype(int)]))[0]
cand_f = np.nonzero(np.isin(fb_, bal_b) & (fwlen >= 5) & (fweave - fwbot >= 5.4))[0]
valley = np.full(len(F0), np.nan)
vf = cand_f[np.array([ZONES[ZONE_IDS[zone_f[i]]]["side"] == "valley" for i in cand_f], bool)] if len(cand_f) else cand_f
if len(vf):
    valley[vf] = ground_z(fworld(vf, fwlen[vf] / 2, 4.0))
for b in bal_b:
    fs = cand_f[fb_[cand_f] == b]
    if not len(fs):
        continue
    side = ZONES[ZONE_IDS[zone[b]]]["side"]
    score = facing[fs] if side == "street" else -np.nan_to_num(valley[fs], nan=1e4) if side == "valley" else -facing[fs]
    i = fs[int(np.argmax(score))]
    wl = fwlen[i]
    nwin = max(int(wl // spacing[b]), 1)
    bay = wl / nwin
    stacks = sorted({nwin // 4, nwin - 1 - nwin // 4}) if flat[b] and wl >= 12 else [nwin // 2]
    k0 = int(np.ceil((fwbot[i] + 2.4) / 3.0))
    floors = [k for k in range(k0, k0 + 12) if 3 * k + 3 <= fweave[i] + 0.3]
    if not flat[b]:
        floors = floors[:1]
    wide = float(np.clip(bay * 0.85 + 0.9, 2.0, 3.6))
    for j in stacks:
        uc = float(np.clip((j + 0.5) * bay, wide / 2 + 0.2, wl - wide / 2 - 0.2))
        for k in floors:
            z = fzmin[i] + 3 * k
            add_box(bcell[b], i, uc - wide / 2, uc + wide / 2, 0.0, 1.3, z - 0.15, z, S_SLAB)
            add_box(bcell[b], i, uc - wide / 2, uc + wide / 2, 1.25, 1.3, z, z + 1.0, S_RAIL)
            for s in (-1, 1):
                add_box(bcell[b], i, uc + s * wide / 2 - 0.025, uc + s * wide / 2 + 0.025, 0.0, 1.25, z, z + 1.0, S_RAIL)
            n_balc += 1
if drive_cars:
    dc = np.array(drive_cars, np.float64)
    dc[:, :3] -= ORIGIN
    json.dump({"cars": np.round(dc, 2).ravel().tolist()},
              open(os.path.join(OUT, f"parked_drive_{PART.split('/')[0] if PART else 0}.json"), "w"))
print("FACADE EXTRAS shops", len(shop_fac), "steps", n_steps, "porch roofs", n_canopy, "balconies", n_balc,
      "bins", n_bins, flush=True)
detail_mats = [export_material("StepStone", rgb=tuple(lin("A8A49C")), rough=0.8),
               export_material("BalconySlab", rgb=(0.45, 0.44, 0.42), rough=0.9),
               export_material("Railing", rgb=tuple(lin("303030")), rough=0.5, metal=0.6),
               export_material("PorchCanopy", rgb=(0.05, 0.05, 0.055), rough=0.4, metal=0.5),
               export_material("BinBody", rgb=tuple(lin("4A4D4F")), rough=0.6),
               export_material("BinLidBlack", rgb=tuple(lin("1E1F20")), rough=0.6),
               export_material("BinLidBlue", rgb=tuple(lin("1F4E9A")), rough=0.6),
               export_material("BinLidBrown", rgb=tuple(lin("5A3A22")), rough=0.6),
               export_material("BinLidYellow", rgb=tuple(lin("E8C21C")), rough=0.6)]


def box_uv(co, faces):
    """planar metre UVs per quad: along its first edge and its second"""
    uv = np.zeros((len(faces) * 4, 2), np.float32)
    for k, q in enumerate(faces):
        p = co[list(q)]
        e1 = p[1] - p[0]; e1 /= max(np.linalg.norm(e1), 1e-6)
        e2 = p[3] - p[0]; e2 /= max(np.linalg.norm(e2), 1e-6)
        uv[4 * k:4 * k + 4] = np.c_[(p - p[0]) @ e1, (p - p[0]) @ e2]
    return uv


def mesh_from(name, ck, co, faces, mats, mat_idx, uv=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata((co - ORIGIN).tolist(), [], [tuple(int(x) for x in f) for f in faces])
    for m in mats:
        me.materials.append(m)
    me.polygons.foreach_set("material_index", np.asarray(mat_idx, np.int32))
    me.uv_layers.new(name="UVMap").data.foreach_set("uv", (box_uv(co, faces) if uv is None else uv).astype(np.float32).ravel())
    ob = bpy.data.objects.new(name, me)
    exp.objects.link(ob)
    by_tile.setdefault(tile_of_cell(ck), []).append(ob)


for ck, (v, f, m) in sbox.items():
    if wanted(ck):
        mesh_from(f"Details_{ck}", ck, np.vstack(v), f, detail_mats, m)

# ---------------- roof overhangs: every free edge of a pitched roof carried on in its own plane ----------------
# top face smears the roof edge's orthophoto pixels outward, a light soffit and fascia underneath
soffit = export_material("Soffit", rgb=(0.72, 0.71, 0.68), rough=0.9)
slot_tile = {s: re.search(r"_(\d{3}_\d{4})$", bme.materials[s].name).group(1) for s in roof_slots}
rf = np.nonzero(np.isin(BA["material_index"], roof_slots))[0]
rf = rf[bwant[bid[rf]] & (bk[bid[rf]] != 2) & (bh[bid[rf]] >= 3.0)]
li, lt = loops_of(BA, rf)
start = np.repeat(np.r_[0, np.cumsum(lt)[:-1]], lt)
pos_in = np.arange(len(li)) - start
nxt = start + (pos_in + 1) % np.repeat(lt, lt)
va, vb = BA["lv"][li], BA["lv"][li[nxt]]
ekey = np.minimum(va, vb).astype(np.int64) * len(BA["co"]) + np.maximum(va, vb)
_, einv, ecnt = np.unique(ekey, return_inverse=True, return_counts=True)
face_of = np.repeat(rf, lt)
nf = fnrm[face_of]
pa, pb = BA["co"][va].astype(np.float64), BA["co"][vb].astype(np.float64)
e = pb - pa
elen = np.linalg.norm(e, axis=1)
sel = (ecnt[einv.reshape(-1)] == 1) & (nf[:, 2] > 0.2) & (nf[:, 2] < 0.97) & (elen > 0.4)
pa, pb, e, nf, face_of = pa[sel], pb[sel], e[sel], nf[sel].astype(np.float64), face_of[sel]
la, lb = li[sel], li[nxt[sel]]
o = np.cross(e, nf)
o /= np.maximum(np.linalg.norm(o, axis=1), 1e-9)[:, None]
o[(((pa + pb) / 2 - BA["center"][face_of]) * o).sum(1) < 0] *= -1  # away from its own face
O = o * overhang[bid[face_of]][:, None]
T = nf * 0.12
drop = np.array((0.0, 0.0, 0.16))
top = np.stack([pa, pb, pb + O, pa + O], 1)
bot = np.stack([pa - T, pb - T, pb + O - T, pa + O - T], 1)
fas = np.stack([pa + O, pb + O, pb + O - T - drop, pa + O - T - drop], 1)
top_uv = np.stack([ruv[la], ruv[lb], ruv[lb], ruv[la]], 1)


def orient(q, want, uv=None):
    """wind each quad so its normal points along `want`"""
    flip = (np.cross(q[:, 1] - q[:, 0], q[:, 3] - q[:, 0]) * want).sum(1) < 0
    q[flip] = q[flip][:, ::-1]
    if uv is not None:
        uv[flip] = uv[flip][:, ::-1]


orient(top, nf, top_uv)
orient(bot, -nf)
orient(fas, o)
tile_f = np.array([slot_tile[s] for s in BA["material_index"][face_of]], dtype="U16")
cell_f = np.empty(len(face_of), dtype="U20")
for t in np.unique(tile_f):
    k = tile_f == t
    cell_f[k] = [f"{t}_{q}" for q in quadrant(t, BA["center"][face_of[k]])]
n_eave = 0
for ck in np.unique(cell_f):
    if not wanted(ck):
        continue
    k = np.nonzero(cell_f == ck)[0]
    n = len(k)
    co = np.concatenate([top[k], bot[k], fas[k]]).reshape(-1, 3)
    faces = np.arange(3 * n * 4).reshape(-1, 4)
    uv = np.zeros((3 * n * 4, 2))
    uv[:n * 4] = cell_uv(top_uv[k].reshape(-1, 2), ck)
    mesh_from(f"Eaves_{ck}", ck, co, faces, [roof_ortho_mat(ck), soffit], np.r_[np.zeros(n), np.ones(2 * n)], uv)
    n_eave += n
print("EAVES", n_eave, "roof edges", flush=True)

_ter_me = ter.data
bpy.data.objects.remove(ter)
bpy.data.meshes.remove(_ter_me)  # 4.7 M vertices: gone before the walls are split

owner = np.repeat(np.arange(len(bme.polygons)), BA["loop_total"])


def per_loop(face_vals):
    return face_vals[owner]


flags = (kind[:, 1] + 2 * kind[:, 2] + 4 * look[:, 0] + 8 * (style[bid] == 3) + 16 * frame[bid] + 64 * shut[bid]
         + 128 * roll[bid] + 256 * smode[bid])
uv_sets = [("WallUV", wuv),
           ("Geo", per_loop(np.c_[geo[:, 0], geo[:, 2]])),                       # wall length, eave height
           ("Geo2", per_loop(np.c_[geo[:, 1], door[:, 2]])),                      # wall bottom, building height
           ("Door", per_loop(np.c_[door[:, 0], door[:, 1]])),                     # door position, door type + 4 * base/5 cm
           ("Kind", per_loop(np.c_[kind[:, 0], style[bid]])),                     # kind, house style
           ("Flags", per_loop(np.c_[flags, look[:, 2] + 8 * scol])),              # shop/band/white/brick/frame/shutters/roller, colours
           ("Win", per_loop(np.c_[win[bid, 0], win[bid, 1]])),                    # window width, height
           ("Win2", per_loop(np.c_[spacing[bid], plinth[bid]]))]                  # window bay, plinth height
wall_col = per_loop(np.c_[newcol[bid], np.ones(len(bid))])
facade = export_material("Facade", rgb=(0.8, 0.8, 0.8), rough=0.8)
walls = np.nonzero(BA["material_index"] == 1)[0]
walls = walls[bkeep[bid[walls]]]
# LoD2 has walls twice where buildings overlap (a canopy's wall on a house wall, a part inside another): two
# coplanar faces facing the same way flicker as they fight for the depth — the windows blink. One of each stays.
wkey = np.c_[np.round(BA["center"][walls] / 0.05), np.round(fnrm[walls] * 20)].astype(np.int64)
_, first_w = np.unique(wkey, axis=0, return_index=True)
n_dup = len(walls) - len(first_w)
walls = walls[np.sort(first_w)]
print("WALLS duplicates dropped", n_dup, flush=True)
wcell = bcell[bid[walls]]
for ck in np.unique(wcell):
    if wanted(ck):
        emit(f"Walls_{ck}", ck, BA, walls[wcell == ck], [facade], uvs=uv_sets, colors=wall_col, smooth=False)
allb = np.arange(len(bme.polygons))
allb = allb[bkeep[bid]]
acell = bcell[bid[allb]]
for ck in np.unique(acell):
    if wanted(ck):
        emit(f"BuildCol_{ck}_col", ck, BA, allb[acell == ck], [])
print("BUILDINGS done", flush=True)
del BA, geo, door, kind, look, gcol, wuv, uv_sets, wall_col, owner, ruv, fnrm, top, bot, fas, sbox
bpy.data.objects.remove(bld)
bpy.data.meshes.remove(bme)


# ---------------- streets ----------------
def scale_uv(me, uv_name, per_material):
    uvl = me.uv_layers[uv_name]
    uv = np.empty(len(me.loops) * 2, np.float32); uvl.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    mat = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("material_index", mat)
    total = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_total", total)
    k = np.array([per_material.get(i, 1.0) for i in range(max(len(me.materials), 1))], np.float32)
    uvl.data.foreach_set("uv", (uv / np.repeat(k[mat], total)[:, None]).ravel())


def split_by_cell(ob_name, label, uv_name, mats, per_material=None, collision=False):
    ob = bpy.data.objects.get(ob_name)
    if ob is None:
        return
    if per_material:
        scale_uv(ob.data, uv_name, per_material)
    A = arrays(ob.data)
    uv = loop_uv(ob.data, uv_name)
    cells = cell_keys(A["center"][:, :2])
    sel = keep(A["center"])
    for ck in np.unique(cells[sel]):
        if not wanted(ck):
            continue
        f = np.nonzero((cells == ck) & sel)[0]
        emit(f"{label}_{ck}", ck, A, f, mats, mat_idx=A["material_index"], uvs=[("UVMap", uv)])
        if collision:
            emit(f"{label}Col_{ck}_col", ck, A, f, [])
    me = ob.data
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    print("SPLIT", label, flush=True)


UTEX = TEXDIR
asphalt = export_material("Asphalt", os.path.join(UTEX, "asphalt_02_Diffuse.jpg"), normal=os.path.join(UTEX, "asphalt_02_nor_gl.jpg"), rough=0.9)
# roads collide themselves: the terrain under them is sunk, and a car on it sinks into the asphalt
split_by_cell("NRW_Roads", "Roads", "RoadUV", [asphalt], {0: TEX_M["asphalt_02"]}, collision=True)
split_by_cell("NRW_Parking", "Parking", "RoadUV", [asphalt], {0: TEX_M["asphalt_02"]}, collision=True)
split_by_cell("NRW_RoadMarkings", "Marks", "RoadUV", [export_material("RoadPaint", rgb=(0.8, 0.8, 0.78), rough=0.6)])
split_by_cell("NRW_Sidewalks", "Walks", "WallUV",
              [export_material("Curb", os.path.join(UTEX, "curb_grey.jpg"), normal=os.path.join(UTEX, "brushed_concrete_nor_gl.jpg")),
               export_material("Pavers", os.path.join(UTEX, "pavers_grey.jpg"), normal=os.path.join(UTEX, "concrete_pavers_02_nor_gl.jpg"))],
              {0: TEX_M["brushed_concrete"], 1: TEX_M["concrete_pavers_02"]}, collision=True)
split_by_cell("NRW_Gutters", "Gutters", "RoadUV",
              [export_material("Gutter", os.path.join(UTEX, "pavers_grey.jpg"), normal=os.path.join(UTEX, "concrete_pavers_02_nor_gl.jpg"))],
              {0: TEX_M["concrete_pavers_02"] * 0.5})
metal = export_material("Metal", rgb=(0.07, 0.075, 0.08), rough=0.4, metal=0.8)
split_by_cell("NRW_Barriers", "Barriers", "RoadUV",
              [export_material("Hedge", os.path.join(UTEX, "hedge.jpg"), normal=os.path.join(UTEX, "leafy_grass_nor_gl.jpg")),
               metal, export_material("Cutout_Fence", os.path.join(UTEX, "fence.png"), alpha=True),
               export_material("GardenWall", os.path.join(UTEX, "curb_grey.jpg")),
               export_material("GuardRail", rgb=(0.55, 0.56, 0.57), rough=0.35, metal=0.8)],
              {0: 2.0, 2: 1.0, 3: TEX_M["brushed_concrete"]})
def collision_by_cell(ob_name, label):
    """collision only, nothing drawn: the Autobahn's solid outer guard rails"""
    ob = bpy.data.objects.get(ob_name)
    if ob is None:
        return
    A = arrays(ob.data)
    cells = cell_keys(A["center"][:, :2])
    sel = keep(A["center"])
    for ck in np.unique(cells[sel]):
        if wanted(ck):
            emit(f"{label}Col_{ck}_col", ck, A, np.nonzero((cells == ck) & sel)[0], [])
    me = ob.data
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    print("SPLIT", label, "(collision only)", flush=True)


collision_by_cell("NRW_RailWall", "RailWall")
# rivers, lakes, the Rhine (build_water.py): Unity puts the Dexstand/Water shader on the "Water" material
split_by_cell("NRW_Water", "Water", "UVMap", [export_material("Water", rgb=(0.03, 0.07, 0.08), rough=0.05)])
split_by_cell("NRW_Railway", "Rail", "RoadUV",
              [export_material("RailSteel", rgb=(0.32, 0.3, 0.28), rough=0.35, metal=0.9), export_material("Sleeper", rgb=(0.42, 0.41, 0.39)),
               export_material("PlatformTop", os.path.join(UTEX, "pavers_grey.jpg")), export_material("PlatformEdge", os.path.join(UTEX, "curb_grey.jpg"))],
              {2: TEX_M["concrete_pavers_02"], 3: TEX_M["brushed_concrete"]}, collision=True)


# ---------------- props: every small object merged per cell ----------------
def combine(name, ck, items):
    """one mesh from (mesh, 4x4 world matrix) pairs; materials merged by name"""
    mats, co_l, lv_l, lt_l, mi_l, uv_l, base = [], [], [], [], [], [], 0
    for me, mw in items:
        n = len(me.polygons)
        if not n:
            continue
        co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        M = np.array(mw)
        co_l.append(co @ M[:3, :3].T + M[:3, 3] - ORIGIN)
        lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv); lv_l.append(lv + base)
        lt = np.empty(n, np.int32); me.polygons.foreach_get("loop_total", lt); lt_l.append(lt)
        mi = np.empty(n, np.int32); me.polygons.foreach_get("material_index", mi)
        remap = []
        for m in (me.materials or [None]):
            em = plain_copy(m)
            if em not in mats:
                mats.append(em)
            remap.append(mats.index(em))
        mi_l.append(np.array(remap, np.int32)[np.clip(mi, 0, len(remap) - 1)])
        if me.uv_layers:
            u = np.empty(len(me.loops) * 2, np.float32); me.uv_layers[0].data.foreach_get("uv", u); uv_l.append(u.reshape(-1, 2))
        else:
            uv_l.append(np.zeros((len(me.loops), 2), np.float32))
        base += len(co)
    if not co_l:
        return None
    lt = np.concatenate(lt_l)
    me = bpy.data.meshes.new(name)
    co = np.vstack(co_l)
    me.vertices.add(len(co)); me.vertices.foreach_set("co", co.astype(np.float32).ravel())
    lv = np.concatenate(lv_l)
    me.loops.add(len(lv)); me.loops.foreach_set("vertex_index", lv)
    me.polygons.add(len(lt)); me.polygons.foreach_set("loop_start", np.r_[0, np.cumsum(lt)[:-1]].astype(np.int32))
    me.update(calc_edges=True)
    me.uv_layers.new(name="UVMap").data.foreach_set("uv", np.vstack(uv_l).astype(np.float32).ravel())
    for m in mats:
        me.materials.append(m)
    me.polygons.foreach_set("material_index", np.concatenate(mi_l))
    ob = bpy.data.objects.new(name, me)
    exp.objects.link(ob)
    by_tile.setdefault(tile_of_cell(ck), []).append(ob)
    return ob


prop_items = {}
temp_meshes = []
sources = list(bpy.data.collections["NRW_Props"].objects) + list(bpy.data.collections["NRW_Signs"].objects)
sources += [o for o in (bpy.data.objects.get("NRW_Busbf"), bpy.data.objects.get("NRW_Rewe")) if o is not None]
if "NRW_Landmarks" in bpy.data.collections:
    sources += list(bpy.data.collections["NRW_Landmarks"].objects)
for o in sources:
    if o.type == "FONT":
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
        temp_meshes.append(me)
    elif o.type == "MESH":
        me = o.data
    else:
        continue
    c = np.array([o.matrix_world @ Vector(v) for v in o.bound_box]).mean(0)
    ck = cell_keys(c[None, :2])[0]
    if wanted(ck) and keep(c[None, :2])[0]:
        prop_items.setdefault(ck, []).append((me, o.matrix_world))
for ck, items in prop_items.items():
    combine(f"Props_{ck}", ck, items)
print("PROPS cells", len(prop_items), flush=True)

# ---------------- trees: a crown of leaf-cluster cards (each tree its own shape) on a 6-sided trunk, from the points ----
def trunk(r=0.035, h=0.62, sides=6):
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    v = np.r_[np.c_[r * np.cos(a), r * np.sin(a), np.zeros(sides)], np.c_[0.7 * r * np.cos(a), 0.7 * r * np.sin(a), np.full(sides, h)]]
    f = np.array([(i, (i + 1) % sides, sides + (i + 1) % sides, sides + i) for i in range(sides)], np.int32)
    return v.astype(np.float32), f


CROWN_C = np.array([0, 0, 0.64], np.float32)
CROWN_R = np.array([0.8, 0.8, 0.38], np.float32)


def leaf_cards(n=14):
    """a crown of n leaf-cluster cards round an ellipsoid, each facing out: reads as foliage, fewer faces than a ball"""
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    d = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)].astype(np.float32)
    V, F, UV = [], [], []
    for di in d:
        u = np.cross(di, (0, 0, 1)); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-3 else np.array((1, 0, 0), np.float32)
        v = np.cross(di, u)
        c = CROWN_C + di * CROWN_R * 0.55
        s = 0.62
        b = len(V)
        V += [c - u * s - v * s, c + u * s - v * s, c + u * s + v * s, c - u * s + v * s]
        UV += [(0, 0), (1, 0), (1, 1), (0, 1)]
        F += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    return np.array(V, np.float32), np.array(F, np.int32), np.array(UV, np.float32)


def fir_cards():
    """a spruce as two tiers of three crossed needle cards"""
    V, F, UV = [], [], []
    for (z0, z1, r) in ((0.1, 0.66, 0.34), (0.46, 1.02, 0.24)):
        for a in (0, np.pi / 3, 2 * np.pi / 3):
            u = np.array((np.cos(a), np.sin(a), 0), np.float32) * r
            b = len(V)
            V += [-u + (0, 0, z0), u + (0, 0, z0), u + (0, 0, z1), -u + (0, 0, z1)]
            UV += [(0, 0), (1, 0), (1, 1), (0, 1)]
            F += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    return np.array(V, np.float32), np.array(F, np.int32), np.array(UV, np.float32)


LV, LF, LUV = leaf_cards()
FV, FF, FUV = fir_cards()
TV, TF = trunk()
tp = bpy.data.objects["NRW_Trees"].data
npts = len(tp.vertices)
pos = np.empty(npts * 3, np.float32); tp.vertices.foreach_get("co", pos); pos = pos.reshape(-1, 3)
variant = np.empty(npts, np.int32); tp.attributes["variant"].data.foreach_get("value", variant)
tscale = np.empty(npts * 3, np.float32); tp.attributes["scale"].data.foreach_get("vector", tscale); tscale = tscale.reshape(-1, 3)
trot = np.empty(npts, np.float32); tp.attributes["rot"].data.foreach_get("value", trot)
ttint = np.empty(npts, np.float32); tp.attributes["tint"].data.foreach_get("value", ttint)
# Trees are not baked into meshes any more (400 k of them were ~1.3 GB of the APK): Unity's MapTrees draws two
# template trees GPU-instanced from these points. trees.bytes: float32 x, y, z (Blender metres around the bus
# station), sx, sy, sz, rotation about z, conifer (0/1) per tree; tree_templates.json: the templates' geometry in
# Blender axes (Unity = (-x, z, -y)), trunk + leaf cards and trunk + fir cards.
tkeep = keep(pos)
if not PART or PART.startswith("0/"):
    rel = pos[tkeep] - ORIGIN
    np.c_[rel, tscale[tkeep], trot[tkeep], (variant[tkeep] == 3)].astype(np.float32).tofile(os.path.join(OUT, "trees.bytes"))

    def part(V, F, UV=None):
        return {"v": np.round(V, 4).ravel().tolist(), "t": F.ravel().tolist(),
                "uv": (UV if UV is not None else V[:, :2] * 2.0).ravel().tolist()}
    trunk_tris = np.c_[TF[:, [0, 1, 2]], TF[:, [0, 2, 3]]].reshape(-1, 3)
    json.dump({"trunk": part(TV, trunk_tris), "leaves": part(LV, LF, LUV), "fir": part(FV, FF, FUV)},
              open(os.path.join(OUT, "tree_templates.json"), "w"))
print("TREES", int(tkeep.sum()), flush=True)

# ---------------- parked cars: points for Unity's MapParkedCars (whole map, written once) ----------------
spots = bpy.data.objects.get("NRW_ParkSpots")
if spots is not None and (not PART or PART.startswith("0/")):
    sme = spots.data
    sp = np.empty(len(sme.vertices) * 3, np.float32); sme.vertices.foreach_get("co", sp); sp = sp.reshape(-1, 3)
    hd = np.empty(len(sme.vertices), np.float32); sme.attributes["heading"].data.foreach_get("value", hd)
    ok = keep(sp)
    rel = sp[ok] - ORIGIN  # Blender metres around the bus station; Unity = (-x, z, -y) * map scale
    json.dump({"cars": np.round(np.c_[rel, hd[ok]], 2).ravel().tolist()}, open(os.path.join(OUT, "parked.json"), "w"))
    print("PARKED", int(ok.sum()), flush=True)

# ---------------- one FBX per 1 km tile ----------------
layer = bpy.context.view_layer
for tile, objs in sorted(by_tile.items()):
    layer.update()
    for o in layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    layer.objects.active = objs[0]
    path = os.path.join(OUT, f"Tile_{tile}.fbx")
    bpy.ops.export_scene.fbx(filepath=path, use_selection=True, object_types={"MESH"}, apply_unit_scale=True,
                             apply_scale_options="FBX_SCALE_ALL", bake_space_transform=True, axis_forward="-Z", axis_up="Y",
                             use_mesh_modifiers=False, mesh_smooth_type="OFF", colors_type="LINEAR", path_mode="STRIP",
                             embed_textures=False, add_leaf_bones=False, bake_anim=False)
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in objs if not o.name.endswith("_col"))
    print("TILE", tile, len(objs), "objects", tris, "triangles", round(os.path.getsize(path) / 2 ** 20, 1), "MB", flush=True)
# every part adds its tiles to the list the Unity side reads
rj = os.path.join(OUT, "region.json")
done = set(json.load(open(rj))["tiles"]) if PART and os.path.exists(rj) and not PART.startswith("0/") else set()
json.dump({"origin_local": ORIGIN.tolist(), "cell_m": 500, "tiles": sorted(done | set(by_tile))}, open(rj, "w"))
print("EXPORT_DONE", len(by_tile), "tiles")
