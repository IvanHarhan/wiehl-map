"""Open карта.blend and add NRW terrain + LoD2 buildings aligned to the Google 3D Tiles mesh.
blender -b карта.blend --python build_blend.py -- OUT.blend [PREPARED_DIR]
PREPARED_DIR defaults to the old town (wiehl_data/prepared); region_data/prepared is the whole region, whose
orthophoto comes as 1 km tiles, so terrain and roofs there get one material per tile."""
import bpy, bmesh, json, math, os, sys
import numpy as np
from mathutils.bvhtree import BVHTree
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:]
OUT = ARGS[0]
P = ARGS[1] if len(ARGS) > 1 else r"A:\blender_work\wiehl_data\prepared"
meta = json.load(open(os.path.join(P, "meta.json")))
UX0, UY0, UX1, UY1 = meta["utm_bbox"]
AFF = np.array(meta["affine"])  # region: 4.5 cm off at worst over 10 km, fine for placing things on the ground
REGION = "ortho_m_per_px" in meta
# an A4 corridor chunk (A:\blender_work\a4\chunk_K): no Google mesh under it and none of Wiehl's hand-made places;
# its height offset must be the main region's, or the tiles would not meet
CHUNK = bool(meta.get("chunk"))
DEM_RES = meta.get("dem_res", 1)  # metres per DEM cell
TERRAIN_STEP = 2


def to_local(dx, dy):
    return np.c_[dx, dy, np.ones(len(dx))] @ AFF


INV = np.linalg.inv(AFF[:2])


def local_to_uv(xy):
    d = (xy - AFF[2]) @ INV
    return np.c_[d[:, 0] / (UX1 - UX0), d[:, 1] / (UY1 - UY0)]


# ---- height offset: compare DEM with the Google mesh on open ground ----
dem = np.load(os.path.join(P, "dem.npy"))
google = bpy.data.objects.get("Google 3D Tiles")
offset = None
if google:
    dg = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(google, dg)
    mw = google.matrix_world
    diffs = []
    rng = np.random.default_rng(0)
    for _ in range(4000):
        r, c = rng.integers(0, dem.shape[0]), rng.integers(0, dem.shape[1])
        x, y = to_local(np.array([(c + 0.5) * DEM_RES]), np.array([UY1 - UY0 - (r + 0.5) * DEM_RES]))[0]
        hit = bvh.ray_cast(mw.inverted() @ Vector((x, y, 5000)), Vector((0, 0, -1)))
        if hit[0] is not None:
            diffs.append((mw @ hit[0]).z - dem[r, c])
    if diffs:
        offset = float(np.percentile(diffs, 25))
if CHUNK:
    offset = float(meta["chunk"]["height_offset"])
elif offset is None:
    offset = 47.0 - bpy.context.scene.get("height_offset", 0.0)
print("HEIGHT_OFFSET", offset, "samples", len(diffs) if google else 0)

col = bpy.data.collections.new("NRW")
bpy.context.scene.collection.children.link(col)

if REGION:
    TILES = sorted(tuple(map(int, k.split("_"))) for k in meta["ortho_m_per_px"])
    ortho = None
else:
    ortho = bpy.data.images.load(os.path.join(P, "ortho.jpg"))


def tile_image(folder, key, colour=True):
    img = bpy.data.images.load(os.path.join(P, folder, key + (".jpg" if folder == "ortho" else ".png")), check_existing=True)
    if not colour:
        img.colorspace_settings.name = "Non-Color"
    return img


def local_to_d(xy):
    """local x, y -> metres east/north of the region's south-west corner"""
    return (xy - AFF[2]) @ INV


def ortho_material(name, image=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.9
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image or ortho
    tex.extension = "EXTEND"
    tex.interpolation = "Cubic"
    # the summer orthophoto is pale; deepen midtones so asphalt and lawns don't read as white
    gamma = nt.nodes.new("ShaderNodeGamma")
    gamma.inputs["Gamma"].default_value = 1.2
    nt.links.new(tex.outputs["Color"], gamma.inputs["Color"])
    nt.links.new(gamma.outputs["Color"], bsdf.inputs["Base Color"])
    return m


TEX = r"A:\blender_work\textures"
TEX_SIZE_M = {k: v[0] / 1000 for k, v in json.load(open(os.path.join(TEX, "dims.json"))).items()}


class Nodes:
    def __init__(self, mat):
        self.nt = mat.node_tree
        self.bsdf = self.nt.nodes["Principled BSDF"]

    def new(self, kind, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, a, b):
        self.nt.links.new(a, b)

    def math(self, op, a, b=None):
        n = self.new("ShaderNodeMath", operation=op)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                self.link(v, n.inputs[i])
        return n.outputs[0]

    def attr(self, name):
        return self.new("ShaderNodeAttribute", attribute_name=name).outputs["Fac"]

    def uv(self, name):
        return self.new("ShaderNodeUVMap", uv_map=name).outputs["UV"]

    def tex(self, asset, kind, uv, colour=True):
        m = self.new("ShaderNodeMapping")
        m.inputs["Scale"].default_value = (1 / TEX_SIZE_M[asset],) * 3
        self.link(uv, m.inputs["Vector"])
        t = self.new("ShaderNodeTexImage")
        t.image = bpy.data.images.load(os.path.join(TEX, "%s_%s.jpg" % (asset, kind)), check_existing=True)
        if not colour:
            t.image.colorspace_settings.name = "Non-Color"
        self.link(m.outputs["Vector"], t.inputs["Vector"])
        return t.outputs["Color"]

    def mix(self, fac, a, b):
        n = self.new("ShaderNodeMix", data_type="RGBA")
        for sock, v in ((n.inputs[0], fac), (n.inputs[6], a), (n.inputs[7], b)):
            if isinstance(v, tuple):
                sock.default_value = v
            elif isinstance(v, (int, float)):
                sock.default_value = v
            else:
                self.link(v, sock)
        return n.outputs[2]

    def normal(self, colour, strength=1.0):
        n = self.new("ShaderNodeNormalMap")
        n.inputs["Strength"].default_value = strength
        self.link(colour, n.inputs["Color"])
        self.link(n.outputs["Normal"], self.bsdf.inputs["Normal"])


def band(nn, x, lo, hi):
    return nn.math("MULTIPLY", nn.math("GREATER_THAN", x, lo), nn.math("LESS_THAN", x, hi))


exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wall_shader.py"), encoding="utf-8").read())


def roof_material(image=None, name="NRW_Roof"):
    """Orthophoto colour; sloped roofs get clay-tile relief on top of it."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nn = Nodes(m)
    photo = nn.new("ShaderNodeTexImage", interpolation="Cubic", extension="EXTEND")
    photo.image = image or ortho
    nn.link(nn.uv("UVMap"), photo.inputs["Vector"])
    tuv = nn.uv("WallUV")
    sloped = nn.attr("sloped")
    detail = nn.new("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    nn.link(sloped, detail.inputs[0])
    nn.link(photo.outputs["Color"], detail.inputs[6])
    bright = nn.new("ShaderNodeBrightContrast")
    bright.inputs["Bright"].default_value = 0.35
    bw = nn.new("ShaderNodeRGBToBW")
    nn.link(nn.tex("clay_roof_tiles_02", "Diffuse", tuv), bw.inputs[0])
    nn.link(bw.outputs[0], bright.inputs["Color"])
    nn.link(bright.outputs[0], detail.inputs[7])
    nn.link(detail.outputs[2], nn.bsdf.inputs["Base Color"])
    nn.bsdf.inputs["Roughness"].default_value = 0.8
    nn.normal(nn.mix(sloped, (0.5, 0.5, 1.0, 1), nn.tex("clay_roof_tiles_02", "nor_gl", tuv, False)))
    return m


# ---- terrain ----
def dem_at(dx, dy):
    """bilinear DEM height at metres east/north of the region corner (the grid holds 1 m cell centres)"""
    c = np.clip(dx / DEM_RES - 0.5, 0, dem.shape[1] - 1.001)
    r = np.clip(((UY1 - UY0) - dy) / DEM_RES - 0.5, 0, dem.shape[0] - 1.001)
    c0, r0 = c.astype(int), r.astype(int)
    fc, fr = c - c0, r - r0
    return (dem[r0, c0] * (1 - fc) * (1 - fr) + dem[r0, c0 + 1] * fc * (1 - fr)
            + dem[r0 + 1, c0] * (1 - fc) * fr + dem[r0 + 1, c0 + 1] * fc * fr)


if REGION:
    # the user's outline (10 m cells), grown by 30 m so the edge of the world is not drawn tight along it
    _terr = np.load(os.path.join(P, "territory.npz"))
    TE0, TN1, TRES = int(_terr["E0"]), int(_terr["N1"]), int(_terr["res"])
    from numpy.lib.stride_tricks import sliding_window_view as _swv
    TMASK = _swv(np.pad(_terr["mask"], 3), (7, 7)).any(axis=(2, 3))

    def in_territory_d(dx, dy):
        """metres east/north of the region corner -> inside the outline"""
        r = ((TN1 * 1000 - (dy + UY0)) / TRES).astype(int)
        c = ((dx + UX0 - TE0 * 1000) / TRES).astype(int)
        ok = (r >= 0) & (r < TMASK.shape[0]) & (c >= 0) & (c < TMASK.shape[1])
        out = np.zeros(len(r), bool)
        out[ok] = TMASK[r[ok], c[ok]]
        return out

if REGION:
    # one mesh, one material slot per 1 km tile (its own orthophoto); 2 m grid in the old town, 2.5 m in the rest
    # of Wiehl and Brüchermühle, 5 m elsewhere; a 1.5 m skirt round every tile hides the cracks between grids
    parts_co, parts_q, parts_uv, parts_mat, parts_step = [], [], [], [], []
    base = 0
    for ti, (e, n) in enumerate(TILES):
        key = f"{e}_{n}"
        step = 2.0 if key in meta["core"] else 2.5 if key in meta["detail"] else 5.0
        k = int(round(1000 / step)) + 1
        ox, oy = e * 1000 - UX0, n * 1000 - UY0
        gx, gy = np.meshgrid(ox + np.arange(k) * step, oy + np.arange(k) * step)
        dx, dy = gx.ravel(), gy.ravel()
        z = dem_at(dx, dy)
        ring = np.r_[np.arange(k - 1), np.arange(k - 1) * k + k - 1, (k - 1) * k + np.arange(k - 1, 0, -1),
                     np.arange(k - 1, 0, -1) * k]
        dx, dy, z = np.r_[dx, dx[ring]], np.r_[dy, dy[ring]], np.r_[z, z[ring] - 1.5]
        q = (np.arange(k - 1)[:, None] * k + np.arange(k - 1)[None, :]).ravel()
        quads = np.c_[q, q + 1, q + k + 1, q + k]
        a, low = ring, k * k + np.arange(len(ring))
        skirt = np.c_[a, low, np.roll(low, -1), np.roll(a, -1)]
        quads = np.r_[quads, skirt]
        quads = quads[in_territory_d(dx[quads].mean(1), dy[quads].mean(1))] + base
        parts_co.append(np.c_[to_local(dx, dy), z + offset])
        parts_q.append(quads)
        parts_uv.append(np.c_[(dx - ox) / 1000, (dy - oy) / 1000])
        parts_mat.append(np.full(len(quads), ti, np.int32))
        parts_step.append(np.full(len(dx), step, np.float32))
        base += len(dx)
    co = np.vstack(parts_co).astype(np.float32)
    quads = np.vstack(parts_q).astype(np.int32)
    tuv = np.vstack(parts_uv).astype(np.float32)
    terrain_step = np.concatenate(parts_step)
else:
    d = dem[::TERRAIN_STEP, ::TERRAIN_STEP]
    rows, cols = d.shape
    cc, rr = np.meshgrid(np.arange(cols) * TERRAIN_STEP + 0.5, np.arange(rows) * TERRAIN_STEP + 0.5)
    dx, dy = cc.ravel(), (UY1 - UY0) - rr.ravel()
    xy = to_local(dx, dy)
    co = np.c_[xy, d.ravel() + offset].astype(np.float32)
    q = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]).ravel()
    quads = np.c_[q, q + cols, q + cols + 1, q + 1].astype(np.int32)
    tuv = local_to_uv(xy).astype(np.float32)

me = bpy.data.meshes.new("NRW_Terrain")
me.vertices.add(len(co)); me.vertices.foreach_set("co", co.ravel())
me.loops.add(quads.size); me.loops.foreach_set("vertex_index", quads.ravel())
me.polygons.add(len(quads))
me.polygons.foreach_set("loop_start", np.arange(0, quads.size, 4, dtype=np.int32))
me.update(calc_edges=True)
uv = me.uv_layers.new(name="UVMap")
uv.data.foreach_set("uv", tuv[quads.ravel()].ravel())
if REGION:
    for e, n in TILES:
        me.materials.append(ortho_material(f"NRW_Ground_{e}_{n}", tile_image("ortho", f"{e}_{n}")))
    me.polygons.foreach_set("material_index", np.concatenate(parts_mat))
else:
    me.materials.append(ortho_material("NRW_Ground"))
me.shade_smooth()
terrain = bpy.data.objects.new("NRW_Terrain", me)
col.objects.link(terrain)

# ---- buildings ----
V = np.load(os.path.join(P, "bld_verts.npy"))
FL = np.load(os.path.join(P, "bld_face_len.npy"))
FT = np.load(os.path.join(P, "bld_face_type.npy"))
FB = np.load(os.path.join(P, "bld_face_bld.npy"))
V[:, 2] += offset
keep = FT != 2  # ground faces sit under the terrain
starts = np.r_[0, np.cumsum(FL)[:-1]]
vb = np.repeat(FB, FL)
n_b = FB.max() + 1
zmin = np.full(n_b, np.inf); np.minimum.at(zmin, vb, V[:, 2])
zmax = np.full(n_b, -np.inf); np.maximum.at(zmax, vb, V[:, 2])
bvar = np.random.default_rng(1).random(n_b)
# LoD2 has the bus station canopies as solid blocks; busbf.py models them (and the kiosk under the big one) instead
bcen = np.zeros((n_b, 2)); np.add.at(bcen, vb, V[:, :2]); bcen /= np.maximum(np.bincount(vb, minlength=n_b), 1)[:, None]
BUSBF_AT = np.array(((27.8, -18.4), (54.7, -20.3), (37.2, -11.2)))
BUSBF_BLOCKS = np.nonzero(np.linalg.norm(bcen[:, None] - BUSBF_AT[None], axis=2).min(1) < 1.5)[0]
keep &= ~np.isin(FB, BUSBF_BLOCKS)
# ALKIS function code -> 0 dwelling, 1 commercial/public, 2 no windows (garage, carport, shed), 3 church,
# 5 hall (sports hall, indoor pool), 6 school; building_details.py adds OSM roles (4 = Rewe glass front)
BLANK = ("31001_2460", "31001_2700", "51")
CHURCH = {"31001_3040", "31001_3041", "31001_3042", "31001_3043", "31001_3045", "31001_3046", "31001_3047"}
HALL = {"31001_3211", "31001_3212", "31001_3220", "31001_3221"}
funcs = json.load(open(os.path.join(P, "bld_function.json")))


def function_kind(f, hb):
    # NRW also tags whole attached house parts as 51009_1610 (canopy), so "no windows" only applies to low structures
    if f.startswith(BLANK) and hb < 6:
        return 2
    if f in CHURCH:
        return 3
    if f in HALL:
        return 5
    if f.startswith("31001_302"):
        return 6
    return 1 if f.startswith(("31001_2", "31001_3")) and f != "31001_2460" else 0


bkind = np.array([function_kind(f, hb) for f, hb in zip(funcs, zmax - zmin)], np.float32)

bm = bmesh.new()
uvl = bm.loops.layers.uv.new("UVMap")
wuv = bm.loops.layers.uv.new("WallUV")
l_var = bm.faces.layers.float.new("bvar")
l_h = bm.faces.layers.float.new("bheight")
l_s = bm.faces.layers.float.new("sloped")
l_b = bm.faces.layers.int.new("bid")
l_kind = bm.faces.layers.float.new("kind")
l_wlen = bm.faces.layers.float.new("wlen")
l_wbot = bm.faces.layers.float.new("wbot")
l_weave = bm.faces.layers.float.new("weave")
l_door_u = bm.faces.layers.float.new("door_u")
l_door_t = bm.faces.layers.float.new("door_t")
l_shop = bm.faces.layers.float.new("shop")
l_dcol = bm.faces.layers.float.new("dcol")
l_scol = bm.faces.layers.float.new("scol")
l_band = bm.faces.layers.float.new("band")
l_white = bm.faces.layers.float.new("white")
bverts = [bm.verts.new(v) for v in V]
V_uv = local_to_uv(V[:, :2])
Z = Vector((0, 0, 1))
for s, n, t, k, b in zip(starts, FL, FT, keep, FB):
    if not k:
        continue
    try:
        f = bm.faces.new([bverts[i] for i in range(s, s + n)])
    except ValueError:
        continue
    f.normal_update()
    nrm = f.normal
    f.material_index = 0 if t == 0 else 1
    f[l_var] = bvar[b]
    f[l_h] = zmax[b] - zmin[b]
    f[l_s] = 1.0 if t == 0 and abs(nrm.z) < 0.96 else 0.0
    f[l_b] = b
    # metre-based UV: u along the horizontal edge, v up the face (walls: height above the building's base)
    tan = Z.cross(nrm)
    tan = tan.normalized() if tan.length > 1e-4 else Vector((1, 0, 0))
    bit = nrm.cross(tan)
    if bit.z < 0:
        bit = -bit
    base = Vector((0, 0, zmin[b]))
    us = [(bverts[i].co - base).dot(tan) for i in range(s, s + n)]
    zs = V[s:s + n, 2] - zmin[b]
    # upper half only: on a slope the uphill bottom corner is above the minimum but is not an eave
    upper = np.sort(zs[zs > max(zs.min() + 0.5, (zs.min() + zs.max()) / 2)])
    f[l_kind] = bkind[b]
    f[l_wlen] = max(us) - min(us)
    f[l_wbot] = zs.min()
    # eave = lowest upper corner; a lone upper corner is a gable apex, so the face has no full floor
    f[l_weave] = upper[0] if len(upper) >= 2 else zs.min()
    for lp, i in zip(f.loops, range(s, s + n)):
        lp[uvl].uv = V_uv[i]
        p = lp.vert.co - base
        lp[wuv].uv = (p.dot(tan) - min(us), p.dot(bit))
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.005)
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")

# LoD2 splits one facade into several polygons (storey strips, gable triangle): group connected coplanar
# wall faces, then give the whole facade one window grid, one bottom and one eave
bm.faces.ensure_lookup_table()
walls = [f for f in bm.faces if f.material_index == 1 and abs(f.normal.z) < 0.3]
parent = {f.index: f.index for f in walls}


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for f in walls:
    for e in f.edges:
        for g in e.link_faces:
            if g is not f and g.index in parent and g.normal.dot(f.normal) > 0.99 \
                    and abs(f.normal.dot(g.calc_center_median() - f.calc_center_median())) < 0.05:
                parent[find(g.index)] = find(f.index)
groups = {}
for f in walls:
    groups.setdefault(find(f.index), []).append(f)
for fs in groups.values():
    b = fs[0][l_b]
    nrm = fs[0].normal
    tan = Z.cross(nrm).normalized()
    base = Vector((0, 0, zmin[b]))
    vs = {v for f in fs for v in f.verts}
    us = [(v.co - base).dot(tan) for v in vs]
    zs = np.array([v.co.z for v in vs]) - zmin[b]
    upper = np.sort(zs[zs > max(zs.min() + 0.5, (zs.min() + zs.max()) / 2)])
    if len(upper) >= 2:
        eave = upper[0]
    else:
        # lone apex above the midline: the eave is the highest corner below it
        rest = zs[zs < zs.max() - 0.3]
        eave = rest.max() if len(rest) else zs.min()
    for f in fs:
        f[l_wlen] = max(us) - min(us)
        f[l_wbot] = zs.min()
        f[l_weave] = eave
        for lp in f.loops:
            lp[wuv].uv = ((lp.vert.co - base).dot(tan) - min(us), lp[wuv].uv[1])
print("FACADES", len(groups), "from wall faces", len(walls))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "building_details.py"), encoding="utf-8").read())

me = bpy.data.meshes.new("NRW_Buildings")
bm.to_mesh(me); bm.free()
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "google_colors.py"), encoding="utf-8").read())


def pack_face_attrs(name, parts):
    vals = []
    for part in parts:
        a = np.empty(len(me.polygons), np.float32)
        me.attributes[part].data.foreach_get("value", a)
        vals.append(a)
    me.attributes.new(name, "FLOAT_VECTOR", "FACE").data.foreach_set("vector", np.stack(vals, 1).ravel())


for name, parts in (("w_geo", ("wlen", "wbot", "weave")), ("w_door", ("door_u", "door_t", "bheight")),
                    ("w_kind", ("kind", "shop", "band")), ("w_look", ("white", "brick", "dcol"))):
    pack_face_attrs(name, parts)
me.materials.append(roof_material())
me.materials.append(wall_material())
bld = bpy.data.objects.new("NRW_Buildings", me)
col.objects.link(bld)

for part in ("build_extras.py", "build_streets.py", "build_water.py", "build_props.py", "busbf.py", "rewe.py", "landmarks.py", "labels.py"):
    if CHUNK and part in ("busbf.py", "rewe.py", "landmarks.py", "labels.py"):
        continue
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), part), encoding="utf-8").read())

if REGION:
    # roofs take their colour from the orthophoto tile under them (slot 2 + tile), UVs relative to that tile
    bme = bpy.data.objects["NRW_Buildings"].data
    npoly = len(bme.polygons)
    pmat = np.empty(npoly, np.int32); bme.polygons.foreach_get("material_index", pmat)
    pctr = np.empty(npoly * 3, np.float32); bme.polygons.foreach_get("center", pctr); pctr = pctr.reshape(-1, 3)
    roofs = np.nonzero(pmat == 0)[0]
    d = local_to_d(pctr[roofs, :2].astype(np.float64))
    te = np.floor((d[:, 0] + UX0) / 1000).astype(int)
    tn = np.floor((d[:, 1] + UY0) / 1000).astype(int)
    slot = {}
    for e, n in sorted(set(zip(te.tolist(), tn.tolist()))):
        if (e, n) in TILES:
            slot[(e, n)] = len(bme.materials)
            bme.materials.append(roof_material(tile_image("ortho", f"{e}_{n}"), f"NRW_Roof_{e}_{n}"))
    new_mat = np.array([slot.get((e, n), 0) for e, n in zip(te, tn)], np.int32)
    pmat[roofs] = new_mat
    bme.polygons.foreach_set("material_index", pmat)
    lstart = np.empty(npoly, np.int32); bme.polygons.foreach_get("loop_start", lstart)
    ltotal = np.empty(npoly, np.int32); bme.polygons.foreach_get("loop_total", ltotal)
    lvert = np.empty(len(bme.loops), np.int32); bme.loops.foreach_get("vertex_index", lvert)
    vco = np.empty(len(bme.vertices) * 3, np.float32); bme.vertices.foreach_get("co", vco); vco = vco.reshape(-1, 3)
    ruv = np.empty(len(bme.loops) * 2, np.float32); bme.uv_layers["UVMap"].data.foreach_get("uv", ruv)
    ruv = ruv.reshape(-1, 2)
    li = np.concatenate([np.arange(s, s + t) for s, t in zip(lstart[roofs], ltotal[roofs])])
    owner = np.repeat(np.arange(len(roofs)), ltotal[roofs])
    dl = local_to_d(vco[lvert[li], :2].astype(np.float64))
    ruv[li] = np.c_[(dl[:, 0] - (te[owner] * 1000 - UX0)) / 1000, (dl[:, 1] - (tn[owner] * 1000 - UY0)) / 1000]
    bme.uv_layers["UVMap"].data.foreach_set("uv", ruv.ravel())
    print("ROOF_TILES", len(slot), "roof_faces", len(roofs))

# ---- look: physical sky + low sun, AgX with contrast, so plaster does not burn out to white ----
sc = bpy.context.scene
sc.view_settings.view_transform = "AgX"
for look in ("AgX - Medium High Contrast", "Medium High Contrast", "AgX - Base Contrast"):
    try:
        sc.view_settings.look = look
        break
    except TypeError:
        pass
sc.view_settings.exposure = -0.85
sc.render.engine = "BLENDER_EEVEE"
world = bpy.data.worlds.new("NRW_Sky")
world.use_nodes = True
wn = world.node_tree
sky = wn.nodes.new("ShaderNodeTexSky")
for st in ("MULTIPLE_SCATTERING", "NISHITA", "HOSEK_WILKIE"):
    try:
        sky.sky_type = st
        break
    except TypeError:
        pass
SUN_ELEV, SUN_AZ = math.radians(38), math.radians(215)
if hasattr(sky, "sun_elevation"):
    sky.sun_elevation, sky.sun_rotation = SUN_ELEV, SUN_AZ
wn.links.new(sky.outputs["Color"], wn.nodes["Background"].inputs["Color"])
wn.nodes["Background"].inputs["Strength"].default_value = 0.32  # enough sky light that shaded walls aren't black
sc.world = world
print("LOOK", sc.view_settings.view_transform, sc.view_settings.look, sky.sky_type)
sun = bpy.data.objects.new("NRW_Sun", bpy.data.lights.new("NRW_Sun", "SUN"))
sun.data.energy = 3.0
sun.data.color = (1.0, 0.95, 0.87)
sun.data.angle = math.radians(1.5)
sun.rotation_euler = (math.pi / 2 - SUN_ELEV, 0, SUN_AZ + math.pi / 2)
col.objects.link(sun)

if google:
    google.hide_set(True)
    google.hide_render = True

# open straight into the town-centre view (same as live/blender_live.py aim()); real scale needs a long clip range
VIEW_EL, VIEW_AZ = math.radians(30), math.radians(212)
view_eye = Vector((math.sin(VIEW_AZ) * math.cos(VIEW_EL), math.cos(VIEW_AZ) * math.cos(VIEW_EL), math.sin(VIEW_EL)))
for scr in bpy.data.screens:
    for area in scr.areas:
        for sp in area.spaces:
            if sp.type != "VIEW_3D":
                continue
            sp.clip_start, sp.clip_end, sp.lens = 0.5, 8000, 35
            sp.shading.type = "MATERIAL"
            if REGION:  # ~190 EEVEE materials over 60 km² crash a laptop GPU on open: textured solid view instead
                sp.shading.type, sp.shading.color_type, sp.clip_end = "SOLID", "TEXTURE", 4000
            sp.shading.use_scene_world = sp.shading.use_scene_lights = True
            ov = sp.overlay
            ov.show_floor = ov.show_axis_x = ov.show_axis_y = ov.show_extras = False
            if sp.region_3d:
                sp.region_3d.view_perspective = "PERSP"
                sp.region_3d.view_location = (20, 0, float(co[:, 2].min() + co[:, 2].max()) / 2)
                sp.region_3d.view_rotation = view_eye.to_track_quat("Z", "Y")
                sp.region_3d.view_distance = 420

bpy.ops.wm.save_as_mainfile(filepath=OUT, compress=True)
print("BUILT", OUT, "terrain_verts", len(co), "building_faces", len(me.polygons))
