"""Export the Wiehl scene for Unity (dexstand's Assets/Maps/Import pipeline).
Blender's procedural materials don't survive FBX, so: facades and roofs are baked into colour + normal atlases per
building cluster, the ground is split into orthophoto tiles, everything else gets plain image materials with the
texture scale baked into the UVs. Collision comes from hidden "*_col" meshes (ground, buildings, sidewalks) so
trees and bushes don't block the player. Materials named "Cutout_*" are alpha-tested in Unity (DexstandMapCutout).
Run: blender -b wiehl_nrw.blend --python export_unity.py -- <out_dir>"""
import bpy, bmesh, json, math, os, sys
import numpy as np
from mathutils import Vector

OUT = sys.argv[sys.argv.index("--") + 1]
os.makedirs(OUT, exist_ok=True)
TEX = r"A:\blender_work\textures"
UTEX = r"A:\blender_work\unity_export\tex"
OSM_EXTRA = r"A:\blender_work\wiehl_data\osm_extra.json"
ATLAS = 2048
AREA_PER_ATLAS = 13000.0  # m² of facade + roof per 2048² atlas: about 6 cm per pixel, ~40 atlases
BUSH_KEEP = 0.4
TILES_X, TILES_Y, OVERLAP = 3, 2, 16  # must match unity_textures.py
VEG_GRID = (4, 3)
TEX_M = {k: v[0] / 1000 for k, v in json.load(open(os.path.join(TEX, "dims.json"))).items()}

sc = bpy.context.scene
dg = bpy.context.evaluated_depsgraph_get()

# ---------------- keep only the NRW scene ----------------
KEEP = {"NRW", "NRW_Signs", "NRW_Props", "NRW_VegModels"}
for ob in list(bpy.data.objects):
    if not {c.name for c in ob.users_collection} & KEEP or ob.type in {"LIGHT", "CAMERA"}:
        bpy.data.objects.remove(ob, do_unlink=True)
exp = bpy.data.collections.new("EXPORT")
sc.collection.children.link(exp)
exported = []


def add_export(ob):
    if exp.name not in {c.name for c in ob.users_collection}:
        exp.objects.link(ob)
    exported.append(ob)
    return ob


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


def mesh_copy(ob, name):
    me = ob.data.copy()
    me.name = name
    new = bpy.data.objects.new(name, me)
    new.matrix_world = ob.matrix_world
    return new


def keep_uv(me, name):
    """make `name` the only UV map, so it becomes uv0 in Unity"""
    for uvl in list(me.uv_layers):
        if uvl.name != name:
            me.uv_layers.remove(uvl)


def scale_uv(me, uv_name, per_material):
    """divide metre-based UVs by the texture's size in metres, per material slot"""
    uvl = me.uv_layers[uv_name]
    uv = np.empty(len(me.loops) * 2, np.float32)
    uvl.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    mat = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("material_index", mat)
    total = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_total", total)
    k = np.array([per_material.get(i, 1.0) for i in range(max(len(me.materials), 1))], np.float32)
    uv /= np.repeat(k[mat], total)[:, None]
    uvl.data.foreach_set("uv", uv.ravel())


def set_materials(me, mats):
    # clearing the slots resets every face to slot 0, so carry the indices across
    idx = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("material_index", idx)
    me.materials.clear()
    for m in mats:
        me.materials.append(m)
    me.polygons.foreach_set("material_index", np.clip(idx, 0, max(len(mats) - 1, 0)))


def select_only(objs):
    layer = bpy.context.view_layer
    layer.update()
    for o in layer.objects:
        if o is not None:
            o.select_set(False)
    for o in objs:
        o.select_set(True)
    layer.objects.active = objs[0]


# ---------------- Cycles for baking ----------------
sc.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
for dev in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = dev
    except TypeError:
        continue
    prefs.get_devices()
    gpus = [d for d in prefs.devices if d.type == dev]
    if gpus:
        for d in prefs.devices:
            d.use = d.type == dev
        sc.cycles.device = "GPU"
        break
sc.cycles.samples = 4
sc.render.bake.margin = 8
print("BAKE DEVICE", sc.cycles.device, prefs.compute_device_type)

# ---------------- buildings: clusters, UV atlas, bake ----------------
bld = bpy.data.objects["NRW_Buildings"]
bme = bld.data
n = len(bme.polygons)
bid = np.empty(n, np.int32); bme.attributes["bid"].data.foreach_get("value", bid)
farea = np.empty(n, np.float32); bme.polygons.foreach_get("area", farea)
ctr = np.empty(n * 3, np.float32); bme.polygons.foreach_get("center", ctr); ctr = ctr.reshape(-1, 3)
nb = bid.max() + 1
b_area = np.bincount(bid, weights=farea, minlength=nb)
b_cen = np.zeros((nb, 2)); np.add.at(b_cen, bid, ctr[:, :2]); b_cen /= np.maximum(np.bincount(bid, minlength=nb), 1)[:, None]


def split(ids):
    if b_area[ids].sum() <= AREA_PER_ATLAS or len(ids) == 1:
        return [ids]
    axis = int(np.ptp(b_cen[ids, 0]) < np.ptp(b_cen[ids, 1]))
    order = ids[np.argsort(b_cen[ids, axis])]
    cum = np.cumsum(b_area[order])
    k = min(max(int(np.searchsorted(cum, cum[-1] / 2)) + 1, 1), len(order) - 1)
    return split(order[:k]) + split(order[k:])


clusters = split(np.unique(bid))
print("BUILDINGS area", round(float(farea.sum())), "m2 ->", len(clusters), "atlases")

bake_nodes = []
for mat in bme.materials:
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.name = "BAKE_TARGET"
    mat.node_tree.nodes.active = node
    bake_nodes.append(node)

for ob in list(sc.objects):
    ob.hide_render = True
chunks = []
for ci, ids in enumerate(clusters):
    bm = bmesh.new()
    bm.from_mesh(bme)
    lay = bm.faces.layers.int.get("bid")
    keep = set(int(i) for i in ids)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f[lay] not in keep], context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    name = f"Buildings_{ci:02d}"
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for mat in bme.materials:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    exp.objects.link(ob)
    uv = me.uv_layers.new(name="BakeUV")
    me.uv_layers.active = uv
    uv.active_render = True
    select_only([ob])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.004, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    ob.hide_render = False
    maps = {}
    for kind, btype in (("col", "DIFFUSE"), ("nor", "NORMAL")):
        img = bpy.data.images.new(f"{name}_{kind}", ATLAS, ATLAS, alpha=False)
        if kind == "nor":
            img.colorspace_settings.name = "Non-Color"
        for node in bake_nodes:
            node.image = img
        if btype == "DIFFUSE":
            bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, margin=8, use_clear=True, uv_layer="BakeUV")
        else:
            bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", margin=8, use_clear=True, uv_layer="BakeUV")
        path = os.path.join(OUT, f"{name}_{kind}." + ("jpg" if kind == "col" else "png"))
        img.filepath_raw = path
        img.file_format = "JPEG" if kind == "col" else "PNG"
        img.save()
        maps[kind] = path
    ob.hide_render = True
    set_materials(me, [export_material(f"Facade_{ci:02d}", maps["col"], normal=maps["nor"], rough=0.8)])
    me.polygons.foreach_set("material_index", np.zeros(len(me.polygons), np.int32))
    keep_uv(me, "BakeUV")
    chunks.append(ob)
    exported.append(ob)
    print("BAKED", name, len(me.polygons), "faces", flush=True)
sc.render.engine = "BLENDER_EEVEE"

cb = add_export(mesh_copy(bld, "Buildings_col"))
cb.data.materials.clear()

# ---------------- ground: orthophoto tiles + collision ----------------
ter = bpy.data.objects["NRW_Terrain"]
tc = add_export(mesh_copy(ter, "Terrain_col"))
tc.data.materials.clear()
W, H = 6095, 3940  # prepared/ortho.jpg
tme = ter.data
luv = np.empty(len(tme.loops) * 2, np.float32); tme.uv_layers["UVMap"].data.foreach_get("uv", luv); luv = luv.reshape(-1, 2)
fuv = luv.reshape(-1, 4, 2).mean(1)
tile_of = (np.clip((fuv[:, 0] * TILES_X).astype(int), 0, TILES_X - 1)
           + TILES_X * np.clip((fuv[:, 1] * TILES_Y).astype(int), 0, TILES_Y - 1))
for j in range(TILES_Y):
    for i in range(TILES_X):
        x0, x1 = max(0, W * i // TILES_X - OVERLAP), min(W, W * (i + 1) // TILES_X + OVERLAP)
        y0, y1 = max(0, H * (TILES_Y - 1 - j) // TILES_Y - OVERLAP), min(H, H * (TILES_Y - j) // TILES_Y + OVERLAP)
        bm = bmesh.new()
        bm.from_mesh(tme)
        bm.faces.ensure_lookup_table()
        want = i + TILES_X * j
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if tile_of[f.index] != want], context="FACES")
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
        uvl = bm.loops.layers.uv["UVMap"]
        for f in bm.faces:
            for lp in f.loops:
                u, v = lp[uvl].uv
                lp[uvl].uv = ((u * W - x0) / (x1 - x0), 1 - ((1 - v) * H - y0) / (y1 - y0))
        me = bpy.data.meshes.new(f"Ground_{i}{j}")
        bm.to_mesh(me)
        bm.free()
        keep_uv(me, "UVMap")
        set_materials(me, [export_material(f"Ground_{i}{j}", os.path.join(UTEX, f"ground_{i}{j}.jpg"), rough=0.95)])
        add_export(bpy.data.objects.new(f"Ground_{i}{j}", me))

# ---------------- streets ----------------
asphalt = export_material("Asphalt", os.path.join(TEX, "asphalt_02_Diffuse.jpg"),
                          normal=os.path.join(TEX, "asphalt_02_nor_gl.jpg"), rough=0.9)
for name in ("NRW_Roads", "NRW_Parking"):
    ob = bpy.data.objects[name]
    scale_uv(ob.data, "RoadUV", {0: TEX_M["asphalt_02"]})
    keep_uv(ob.data, "RoadUV")
    set_materials(ob.data, [asphalt])
    add_export(ob)
ob = bpy.data.objects["NRW_RoadMarkings"]
keep_uv(ob.data, "RoadUV")
set_materials(ob.data, [export_material("RoadPaint", rgb=(0.8, 0.8, 0.78), rough=0.6)])
add_export(ob)
ob = bpy.data.objects["NRW_Sidewalks"]
sw_col = add_export(mesh_copy(ob, "Sidewalks_col"))
sw_col.data.materials.clear()
scale_uv(ob.data, "WallUV", {0: TEX_M["brushed_concrete"], 1: TEX_M["concrete_pavers_02"]})
keep_uv(ob.data, "WallUV")
set_materials(ob.data, [export_material("Curb", os.path.join(UTEX, "curb_grey.jpg"), normal=os.path.join(TEX, "brushed_concrete_nor_gl.jpg")),
                        export_material("Pavers", os.path.join(UTEX, "pavers_grey.jpg"), normal=os.path.join(TEX, "concrete_pavers_02_nor_gl.jpg"))])
add_export(ob)

metal = export_material("Metal", rgb=(0.07, 0.075, 0.08), rough=0.4, metal=0.8)
ob = bpy.data.objects.get("NRW_Barriers")
if ob:
    scale_uv(ob.data, "RoadUV", {0: 2.0, 2: 1.0, 3: TEX_M["brushed_concrete"]})
    keep_uv(ob.data, "RoadUV")
    set_materials(ob.data, [export_material("Hedge", os.path.join(UTEX, "hedge.jpg"), normal=os.path.join(TEX, "leafy_grass_nor_gl.jpg")),
                            metal, export_material("Cutout_Fence", os.path.join(UTEX, "fence.png"), alpha=True),
                            export_material("GardenWall", os.path.join(UTEX, "curb_grey.jpg"))])
    add_export(ob)
ob = bpy.data.objects.get("NRW_Railway")
if ob:
    scale_uv(ob.data, "RoadUV", {2: TEX_M["concrete_pavers_02"], 3: TEX_M["brushed_concrete"]})
    keep_uv(ob.data, "RoadUV")
    set_materials(ob.data, [export_material("RailSteel", rgb=(0.32, 0.3, 0.28), rough=0.35, metal=0.9),
                            export_material("Sleeper", rgb=(0.42, 0.41, 0.39)),
                            export_material("PlatformTop", os.path.join(UTEX, "pavers_grey.jpg")),
                            export_material("PlatformEdge", os.path.join(UTEX, "curb_grey.jpg"))])
    add_export(ob)
ob = bpy.data.objects.get("NRW_WindowSills")
if ob:
    set_materials(ob.data, [export_material("Sill", rgb=(0.62, 0.62, 0.6), rough=0.5)])
    add_export(ob)


def join(objs, name):
    """one mesh from many objects (instances of shared meshes become real copies)"""
    objs = [o for o in objs if o.type == "MESH"]
    if not objs:
        return None
    copies = []
    for o in objs:
        c = bpy.data.objects.new(o.name + "_j", o.data.copy())
        c.matrix_world = o.matrix_world
        exp.objects.link(c)
        copies.append(c)
    select_only(copies)
    bpy.ops.object.join()
    res = bpy.context.view_layer.objects.active
    res.name = res.data.name = name
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return res


props = [o for o in bpy.data.collections["NRW_Props"].objects]
remap = {"NRW_PoleMetal": metal, "NRW_LampHead": export_material("LampHead", rgb=(0.55, 0.56, 0.58), rough=0.35, metal=0.6),
         "NRW_ShelterGlass": export_material("Glass_Shelter", rgb=(0.6, 0.7, 0.75), rough=0.05, alpha_value=0.25),
         "NRW_BusSign": export_material("Cutout_BusSign", os.path.join(TEX, "nrw_bus_sign.png"), alpha=True)}
for prefix, name in (("Lamp", "Lamps"), ("BusStop_", "BusStops"), ("BusShelter_", "BusShelters")):
    res = join([o for o in props if o.name.startswith(prefix)], name)
    if res:
        set_materials(res.data, [remap.get(m.name, m) for m in res.data.materials])
        add_export(res)

# signs: lettering to meshes, one object
sign_objs = []
dg = bpy.context.evaluated_depsgraph_get()
for o in bpy.data.collections["NRW_Signs"].objects:
    if o.type == "FONT":
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
        c = bpy.data.objects.new(o.name + "_m", me)
        c.matrix_world = o.matrix_world
        exp.objects.link(c)
        sign_objs.append(c)
    elif o.type == "MESH":
        sign_objs.append(o)
signs = join(sign_objs, "Signs")
if signs:
    for m in signs.data.materials:
        if m.name == "NRW_LogoAldo":
            m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.4
    add_export(signs)

# ---------------- trees and bushes: realized per grid cell ----------------
ng = bpy.data.node_groups["NRW_Scatter"]
out_node = next(nd for nd in ng.nodes if nd.type == "GROUP_OUTPUT")
src_link = out_node.inputs[0].links[0]
# each instance remembers its crown centre, so leaf cards can be lit as one rounded crown after realizing
crown_z = ng.nodes.new("ShaderNodeSeparateXYZ")
scale_attr = ng.nodes.new("GeometryNodeInputNamedAttribute")
scale_attr.data_type = "FLOAT_VECTOR"
scale_attr.inputs["Name"].default_value = "scale"
ng.links.new(scale_attr.outputs["Attribute"], crown_z.inputs[0])
lift = ng.nodes.new("ShaderNodeMath")
lift.operation = "MULTIPLY"
lift.inputs[1].default_value = 0.62
ng.links.new(crown_z.outputs["Z"], lift.inputs[0])
up = ng.nodes.new("ShaderNodeCombineXYZ")
ng.links.new(lift.outputs[0], up.inputs["Z"])
centre = ng.nodes.new("ShaderNodeVectorMath")
centre.operation = "ADD"
ng.links.new(ng.nodes.new("GeometryNodeInputPosition").outputs[0], centre.inputs[0])
ng.links.new(up.outputs[0], centre.inputs[1])
store = ng.nodes.new("GeometryNodeStoreNamedAttribute")
store.data_type = "FLOAT_VECTOR"
store.domain = "INSTANCE"
store.inputs["Name"].default_value = "crown"
ng.links.new(src_link.from_socket, store.inputs["Geometry"])
ng.links.new(centre.outputs[0], store.inputs["Value"])
realize = ng.nodes.new("GeometryNodeRealizeInstances")
ng.links.new(store.outputs["Geometry"], realize.inputs[0])
ng.links.new(realize.outputs[0], out_node.inputs[0])


def crown_normals(me, leaf_slots):
    """leaf-card vertices get normals pointing out of their tree's crown (a little upward); trunks keep theirs"""
    if "crown" not in me.attributes or me.attributes["crown"].domain != "POINT":
        print("NO CROWN ATTRIBUTE on", me.name)
        return
    n = len(me.vertices)
    pos = np.empty(n * 3, np.float32); me.vertices.foreach_get("co", pos); pos = pos.reshape(-1, 3)
    crown = np.empty(n * 3, np.float32); me.attributes["crown"].data.foreach_get("vector", crown); crown = crown.reshape(-1, 3)
    nrm = np.array([v.vector for v in me.vertex_normals], np.float32)
    mi = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("material_index", mi)
    total = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_total", total)
    loop_vert = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", loop_vert)
    leaf = np.zeros(n, bool)
    leaf[loop_vert[np.repeat(np.isin(mi, leaf_slots), total)]] = True
    out = pos - crown
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)
    out = out * 0.8 + np.array([0, 0, 0.35], np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    nrm[leaf] = out[leaf]
    me.normals_split_custom_set_from_vertices(nrm.tolist())
foliage = {"NRW_Bark": export_material("Bark", rgb=(0.2, 0.15, 0.1), rough=0.9),
           "NRW_Leaves": export_material("Cutout_Leaves", os.path.join(UTEX, "foliage_leaves.png"), alpha=True, rough=0.7),
           "NRW_Needles": export_material("Cutout_Needles", os.path.join(UTEX, "foliage_needles.png"), alpha=True, rough=0.7),
           "NRW_Bush": export_material("Cutout_Bush", os.path.join(UTEX, "foliage_bush.png"), alpha=True, rough=0.7)}
bb = np.array([ter.matrix_world @ Vector(c) for c in ter.bound_box])
gx0, gy0 = bb[:, 0].min(), bb[:, 1].min()
gw, gh = (bb[:, 0].max() - gx0) / VEG_GRID[0], (bb[:, 1].max() - gy0) / VEG_GRID[1]
rng = np.random.default_rng(3)
veg_tris = 0
sources = []
for src_name, label, keep_share in (("NRW_Trees", "Trees", 1.0), ("NRW_Bushes", "Bushes", BUSH_KEEP)):
    pm = bpy.data.objects[src_name].data
    npts = len(pm.vertices)
    co = np.empty(npts * 3, np.float32); pm.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    attrs = {}
    for a in ("variant", "scale", "rot", "tint"):
        at = pm.attributes[a]
        width = 3 if at.data_type == "FLOAT_VECTOR" else 1
        buf = np.empty(npts * width, np.int32 if at.data_type == "INT" else np.float32)
        at.data.foreach_get("vector" if width == 3 else "value", buf)
        attrs[a] = (at.data_type, buf.reshape(npts, width) if width == 3 else buf)
    sources.append((label, keep_share, npts, co, attrs))
    # the originals now realize millions of triangles on every depsgraph update
    bpy.data.objects.remove(bpy.data.objects[src_name], do_unlink=True)
for label, keep_share, npts, co, attrs in sources:
    chosen = rng.random(npts) < keep_share
    cell = (np.clip(((co[:, 0] - gx0) / gw).astype(int), 0, VEG_GRID[0] - 1)
            + VEG_GRID[0] * np.clip(((co[:, 1] - gy0) / gh).astype(int), 0, VEG_GRID[1] - 1))
    for c in range(VEG_GRID[0] * VEG_GRID[1]):
        sel = np.nonzero(chosen & (cell == c))[0]
        if not len(sel):
            continue
        pts = bpy.data.meshes.new(f"{label}_pts{c}")
        pts.vertices.add(len(sel))
        pts.vertices.foreach_set("co", co[sel].ravel())
        for a, (dtype, buf) in attrs.items():
            na = pts.attributes.new(a, dtype, "POINT")
            na.data.foreach_set("vector" if dtype == "FLOAT_VECTOR" else "value", buf[sel].ravel())
        tmp = bpy.data.objects.new(f"{label}_tmp{c}", pts)
        exp.objects.link(tmp)
        mod = tmp.modifiers.new("Scatter", "NODES")
        mod.node_group = ng
        proxy = ng.interface.items_tree.get("Viewport Proxy")
        if proxy is not None:  # background evaluation counts as viewport: ask for the real models
            mod[proxy.identifier] = False
        dg.update()
        real = bpy.data.meshes.new_from_object(tmp.evaluated_get(bpy.context.evaluated_depsgraph_get()))
        bpy.data.objects.remove(tmp, do_unlink=True)
        real.name = f"{label}_{c:02d}"
        crown_normals(real, [i for i, m in enumerate(real.materials) if m.name in ("NRW_Leaves", "NRW_Needles", "NRW_Bush")])
        set_materials(real, [foliage.get(m.name, m) for m in list(real.materials)])
        if "UVMap" in real.uv_layers:
            keep_uv(real, "UVMap")
        veg_tris += sum(len(p.vertices) - 2 for p in real.polygons)
        add_export(bpy.data.objects.new(real.name, real))
print("VEGETATION triangles", veg_tris)

# ---------------- centre the map's bounds on the bus station, so the player spawns in the street ----------------
e = next(e for e in json.load(open(OSM_EXTRA, encoding="utf-8"))["elements"]
         if e.get("tags", {}).get("amenity") == "bus_station" and "geometry" in e)
R = 6378137.0


def tm(lat, lon):
    la, lo = math.radians(lat), math.radians(lon - sc["lon"])
    b = math.sin(lo) * math.cos(la)
    return 0.5 * R * math.log((1 + b) / (1 - b)), R * (math.atan(math.tan(la) / math.cos(lo)) - math.radians(sc["lat"]))


sx, sy = np.mean([tm(p["lat"], p["lon"]) for p in e["geometry"]], axis=0)
dg = bpy.context.evaluated_depsgraph_get()
pts = np.vstack([np.array([o.matrix_world @ Vector(c) for c in o.bound_box]) for o in exported])
lo_, hi_ = pts.min(0), pts.max(0)
ext = []
for axis, s in ((0, sx), (1, sy)):
    c = (lo_[axis] + hi_[axis]) / 2
    if abs(s - c) > 0.5:
        edge = 2 * s - (hi_[axis] if s < c else lo_[axis])
        p = [float(sx), float(sy)]
        p[axis] = float(edge)
        ext.append(p)
if ext:
    # a plain (rendered, non-colliding) scrap far below the ground: renderer bounds are what the scene builder centres on
    me = bpy.data.meshes.new("Extent")
    z = float(lo_[2] - 5)
    verts, faces = [], []
    for x, y in ext:
        b0 = len(verts)
        verts += [(x - 0.5, y - 0.5, z), (x + 0.5, y - 0.5, z), (x + 0.5, y + 0.5, z), (x - 0.5, y + 0.5, z)]
        faces.append((b0, b0 + 1, b0 + 2, b0 + 3))
    me.from_pydata(verts, [], faces)
    me.materials.append(bpy.data.materials["RoadPaint"])
    add_export(bpy.data.objects.new("Extent", me))
print("SPAWN at bus station", round(sx, 1), round(sy, 1), "bounds", lo_.round(0), hi_.round(0), "extent helpers", ext)

# ---------------- write the FBX ----------------
for o in exported:
    if o.type != "MESH":
        print("SKIP non-mesh", o.name)
select_only([o for o in exported if o.type == "MESH"])
fbx = os.path.join(OUT, "wiehl.fbx")
bpy.ops.export_scene.fbx(filepath=fbx, use_selection=True, object_types={"MESH"}, apply_unit_scale=True,
                         apply_scale_options="FBX_SCALE_ALL", bake_space_transform=True, axis_forward="-Z", axis_up="Y",
                         use_mesh_modifiers=True, mesh_smooth_type="OFF", colors_type="NONE", path_mode="COPY",
                         embed_textures=False, add_leaf_bones=False, bake_anim=False)
tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in exported if o.type == "MESH")
print("EXPORTED", fbx, round(os.path.getsize(fbx) / 2 ** 20), "MB,", len(exported), "objects,", tris, "triangles")
