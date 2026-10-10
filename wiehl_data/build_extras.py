"""Run from build_blend.py (exec): trees/bushes via geometry-nodes instancing (streets live in build_streets.py).
Uses build_blend globals: P, UX0, UY0, UX1, UY1, AFF, INV, dem, offset, col, Nodes, TEX."""
import math
import numpy as np

rng_e = np.random.default_rng(11)
BUSH_KEEP = 0.6


# ---------------- vegetation models (unit size: height 1, crown radius 1) ----------------
def bark_material():
    m = bpy.data.materials.new("NRW_Bark")
    m.use_nodes = True
    nn = Nodes(m)
    uv = nn.new("ShaderNodeTexCoord").outputs["Object"]
    nn.link(nn.tex("bark_brown_02", "Diffuse", uv), nn.bsdf.inputs["Base Color"])
    nn.bsdf.inputs["Roughness"].default_value = 0.9
    return m


def leaf_atlas(name, seed, needles=False):
    """512px RGBA card: a cluster of leaves (or needle sprigs) inside a disc, transparent elsewhere."""
    S = 512
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:S, 0:S] / S
    rgba = np.zeros((S, S, 4), np.float32)
    for _ in range(260 if needles else 110):
        ang, rad = r.uniform(0, 2 * math.pi), 0.42 * math.sqrt(r.random())
        cx, cy = 0.5 + rad * math.cos(ang), 0.5 + rad * math.sin(ang)
        L = r.uniform(0.05, 0.08) if needles else r.uniform(0.05, 0.09)
        W = L * (0.1 if needles else 0.5)
        th = r.uniform(0, 2 * math.pi)
        dx, dy = xx - cx, yy - cy
        a, b = dx * math.cos(th) + dy * math.sin(th), -dx * math.sin(th) + dy * math.cos(th)
        t = np.clip(a / L, -1, 1)
        m = (np.abs(a) < L) & (np.abs(b) < W * (1 - t * t) ** 0.8)
        v = r.uniform(0.55, 1.0) * (0.85 + 0.15 * t[m])
        hue = r.uniform(-0.06, 0.06)
        rgba[m] = np.c_[v * (0.55 + hue), v * 0.85, v * (0.35 - hue), np.ones(m.sum())]
    img = bpy.data.images.new(name, S, S, alpha=True)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = os.path.join(TEX, name + ".png")
    img.file_format = "PNG"
    img.save()
    return img


def leaf_material(name, img, dark, light):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nn = Nodes(m)
    tex = nn.new("ShaderNodeTexImage", image=img)
    tint = nn.new("ShaderNodeAttribute", attribute_name="tint", attribute_type="INSTANCER").outputs["Fac"]
    ramp = nn.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = dark
    ramp.color_ramp.elements[1].color = light
    nn.link(tint, ramp.inputs[0])
    col = nn.new("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    col.inputs[0].default_value = 1.0
    nn.link(tex.outputs["Color"], col.inputs[6])
    nn.link(ramp.outputs["Color"], col.inputs[7])
    nn.link(col.outputs[2], nn.bsdf.inputs["Base Color"])
    nn.link(tex.outputs["Alpha"], nn.bsdf.inputs["Alpha"])
    nn.bsdf.inputs["Roughness"].default_value = 0.7
    nn.bsdf.inputs["Subsurface Weight"].default_value = 0.15
    return m


BARK = bark_material()
LEAF_IMG = leaf_atlas("nrw_leaves", 3)
LEAF = leaf_material("NRW_Leaves", LEAF_IMG, (0.18, 0.3, 0.12, 1), (0.42, 0.55, 0.22, 1))
NEEDLE = leaf_material("NRW_Needles", leaf_atlas("nrw_needles", 5, True), (0.1, 0.2, 0.12, 1), (0.2, 0.32, 0.16, 1))
BUSH = leaf_material("NRW_Bush", LEAF_IMG, (0.2, 0.32, 0.1, 1), (0.48, 0.6, 0.2, 1))

models = bpy.data.collections.new("NRW_VegModels")
bpy.context.scene.collection.children.link(models)


def blob_mesh(bm, center, radius, subdiv, seed, rough=0.18):
    r = np.random.default_rng(seed)
    geom = bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
    for v in geom["verts"]:
        d = v.co.normalized()
        k = 1 + rough * (math.sin(d.x * 5 + seed) * math.cos(d.y * 4 - seed) + r.uniform(-0.5, 0.5))
        v.co = Vector((d.x * radius[0] * k + center[0], d.y * radius[1] * k + center[1], d.z * radius[2] * k + center[2]))
    return {f for v in geom["verts"] for f in v.link_faces}


def make_model(name, parts):
    """parts: list of ('trunk', r, h) or ('blob', center, radii, subdiv) or ('cone', z0, z1, r); mat index 0 bark, 1 foliage."""
    bm = bmesh.new()
    for i, p in enumerate(parts):
        if p[0] == "trunk":
            g = bmesh.ops.create_cone(bm, cap_ends=True, segments=7, radius1=p[1], radius2=p[1] * 0.7, depth=p[2])
            bmesh.ops.translate(bm, verts=g["verts"], vec=(0, 0, p[2] / 2))
            for f in {f for v in g["verts"] for f in v.link_faces}:
                f.material_index = 0
        elif p[0] == "blob":
            for f in blob_mesh(bm, p[1], p[2], p[3], i * 17 + len(name)):
                f.material_index = 1
        elif p[0] in ("cards", "conecards"):
            # leaf cards: random orientation, biased to the crown's outer shell
            r = np.random.default_rng(i * 31 + len(name))
            uvl_ = bm.loops.layers.uv.verify()
            for _ in range(p[3] if p[0] == "cards" else p[4]):
                if p[0] == "cards":
                    d = r.normal(size=3); d /= np.linalg.norm(d)
                    pos = np.array(p[1]) + d * np.array(p[2]) * r.uniform(0.55, 1.0) ** 0.5
                    size = p[4]
                else:
                    z = r.uniform(p[1], p[2])
                    rad = p[3] * (p[2] - z) / (p[2] - p[1]) * r.uniform(0.4, 1.0)
                    ang = r.uniform(0, 2 * math.pi)
                    pos = np.array((rad * math.cos(ang), rad * math.sin(ang), z))
                    size = p[5]
                ax = r.normal(size=3); ax /= np.linalg.norm(ax)
                t1 = np.cross(ax, (0.3, 0.5, 0.8)); t1 /= np.linalg.norm(t1)
                t2 = np.cross(ax, t1)
                h = size / 2
                vs = [bm.verts.new(pos + (sx * t1 + sy * t2) * h) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                f = bm.faces.new(vs)
                f.material_index = 1
                for lp, uv in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
                    lp[uvl_].uv = uv
        elif p[0] == "branch":
            g = bmesh.ops.create_cone(bm, cap_ends=False, segments=5, radius1=p[3], radius2=p[3] * 0.3,
                                      depth=float(np.linalg.norm(np.subtract(p[2], p[1]))))
            dvec = Vector(p[2]) - Vector(p[1])
            bmesh.ops.rotate(bm, verts=g["verts"], cent=(0, 0, 0), matrix=dvec.to_track_quat("Z", "Y").to_matrix())
            bmesh.ops.translate(bm, verts=g["verts"], vec=(Vector(p[1]) + Vector(p[2])) / 2)
            for f in {f for v in g["verts"] for f in v.link_faces}:
                f.material_index = 0
        elif p[0] == "cone":
            g = bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=p[3], radius2=0.0, depth=p[2] - p[1])
            bmesh.ops.translate(bm, verts=g["verts"], vec=(0, 0, (p[1] + p[2]) / 2))
            for f in {f for v in g["verts"] for f in v.link_faces}:
                f.material_index = 1
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    me.materials.append(BARK)
    me.materials.append(NEEDLE if "Conifer" in name else BUSH if "Bush" in name else LEAF)
    me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    models.objects.link(ob)
    return ob


# names sort alphabetically -> instance index
def branches(z0, tips, r):
    return [("branch", (0, 0, z0), t, r) for t in tips]


make_model("0_TreeRound", [("trunk", 0.045, 0.55)] + branches(0.4, [(0.4, 0.2, 0.62), (-0.35, 0.3, 0.66), (0.05, -0.4, 0.6), (0, 0, 0.85)], 0.02)
           + [("cards", (0, 0, 0.62), (0.8, 0.8, 0.34), 170, 0.32), ("cards", (0.3, 0.1, 0.72), (0.45, 0.45, 0.22), 50, 0.28)])
make_model("1_TreeTall", [("trunk", 0.04, 0.5)] + branches(0.35, [(0.25, 0.1, 0.7), (-0.2, -0.15, 0.75), (0, 0, 0.9)], 0.018)
           + [("cards", (0, 0, 0.64), (0.62, 0.62, 0.36), 170, 0.3)])
make_model("2_TreeWide", [("trunk", 0.055, 0.42)] + branches(0.35, [(0.55, 0, 0.58), (-0.5, 0.2, 0.55), (0.1, -0.5, 0.62), (-0.1, 0.45, 0.66)], 0.022)
           + [("cards", (0.25, 0, 0.6), (0.65, 0.65, 0.32), 110, 0.32), ("cards", (-0.3, 0.1, 0.6), (0.6, 0.6, 0.3), 100, 0.32)])
make_model("3_TreeConifer", [("trunk", 0.035, 0.95), ("conecards", 0.12, 1.0, 0.6, 220, 0.24)])
make_model("4_BushA", [("cards", (0, 0, 0.5), (0.95, 0.95, 0.5), 70, 0.55)])
make_model("5_BushB", [("cards", (0.2, 0, 0.45), (0.75, 0.75, 0.45), 40, 0.5), ("cards", (-0.3, 0.1, 0.4), (0.65, 0.65, 0.4), 35, 0.5)])
bpy.context.view_layer.layer_collection.children[models.name].exclude = True


def proxy_crown():
    """a 20-triangle crown in model space (models are unit-sized, crown round z 0.62), for the viewport only"""
    bmp = bmesh.new()
    bmesh.ops.create_icosphere(bmp, subdivisions=1, radius=1.0)
    bmesh.ops.scale(bmp, vec=(0.75, 0.75, 0.38), verts=bmp.verts)
    bmesh.ops.translate(bmp, vec=(0, 0, 0.62), verts=bmp.verts)
    me = bpy.data.meshes.new("NRW_CrownProxy")
    bmp.to_mesh(me); bmp.free()
    m = bpy.data.materials.new("NRW_CrownProxy")
    m.diffuse_color = (0.09, 0.17, 0.06, 1)
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.09, 0.17, 0.06, 1)
    me.materials.append(m)
    pc = bpy.data.collections.new("NRW_VegProxy")
    bpy.context.scene.collection.children.link(pc)
    ob = bpy.data.objects.new("NRW_CrownProxy", me)
    pc.objects.link(ob)
    bpy.context.view_layer.layer_collection.children[pc.name].exclude = True
    return ob


def scatter_group():
    ng = bpy.data.node_groups.new("NRW_Scatter", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    # the region has ~700k trees and bushes: full models (~600 triangles) would take the viewport down, so the viewport
    # shows 20-triangle crowns; export_unity switches this off to realise the real models
    proxy_sock = ng.interface.new_socket("Viewport Proxy", in_out="INPUT", socket_type="NodeSocketBool")
    proxy_sock.default_value = True
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    ci = N.new("GeometryNodeCollectionInfo")
    ci.inputs["Collection"].default_value = models
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    iop = N.new("GeometryNodeInstanceOnPoints")
    iop.inputs["Pick Instance"].default_value = True

    def named(name, dtype):
        n = N.new("GeometryNodeInputNamedAttribute")
        n.data_type = dtype
        n.inputs["Name"].default_value = name
        return n.outputs["Attribute"]

    rot = N.new("ShaderNodeCombineXYZ")
    L.new(named("rot", "FLOAT"), rot.inputs["Z"])
    L.new(gi.outputs[0], iop.inputs["Points"])
    L.new(ci.outputs[0], iop.inputs["Instance"])
    L.new(named("variant", "INT"), iop.inputs["Instance Index"])
    L.new(rot.outputs[0], iop.inputs["Rotation"])
    L.new(named("scale", "FLOAT_VECTOR"), iop.inputs["Scale"])

    oi = N.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = proxy_crown()
    iop_p = N.new("GeometryNodeInstanceOnPoints")
    L.new(gi.outputs[0], iop_p.inputs["Points"])
    L.new(oi.outputs["Geometry"], iop_p.inputs["Instance"])
    L.new(rot.outputs[0], iop_p.inputs["Rotation"])
    L.new(named("scale", "FLOAT_VECTOR"), iop_p.inputs["Scale"])
    use_proxy = N.new("FunctionNodeBooleanMath")
    use_proxy.operation = "AND"
    L.new(N.new("GeometryNodeIsViewport").outputs[0], use_proxy.inputs[0])
    L.new(gi.outputs["Viewport Proxy"], use_proxy.inputs[1])
    sw = N.new("GeometryNodeSwitch")
    sw.input_type = "GEOMETRY"
    L.new(use_proxy.outputs[0], sw.inputs["Switch"])
    L.new(iop.outputs["Instances"], sw.inputs["False"])
    L.new(iop_p.outputs["Instances"], sw.inputs["True"])
    L.new(sw.outputs[0], go.inputs[0])
    return ng


def point_object(name, xyz, variant, scale, ng):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(xyz))
    me.vertices.foreach_set("co", xyz.astype(np.float32).ravel())
    for aname, dtype, vals in (("variant", "INT", variant), ("scale", "FLOAT_VECTOR", scale),
                               ("rot", "FLOAT", rng_e.uniform(0, 2 * math.pi, len(xyz))),
                               ("tint", "FLOAT", rng_e.random(len(xyz)))):
        a = me.attributes.new(aname, dtype, "POINT")
        a.data.foreach_set("vector" if dtype == "FLOAT_VECTOR" else "value", np.asarray(vals).astype(np.int32 if dtype == "INT" else np.float32).ravel())
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    ob.modifiers.new("Scatter", "NODES").node_group = ng
    return ob


ng = scatter_group()
T = np.load(os.path.join(P, "trees.npy"))
tz = T[:, 2] + offset
tvar = rng_e.choice([0, 1, 2, 3], len(T), p=[0.4, 0.2, 0.25, 0.15])
tscale = np.c_[T[:, 4], T[:, 4], T[:, 3]]
tscale[tvar == 3, :2] *= 0.7
point_object("NRW_Trees", np.c_[T[:, 0], T[:, 1], tz - 0.3], tvar, tscale, ng)

B = np.load(os.path.join(P, "bushes.npy"))
keep = np.full(len(B), BUSH_KEEP)
if REGION:  # 1.2 M bushes over the region: full share in the towns, a sprinkle out in the country
    dB = local_to_d(B[:, :2].astype(np.float64))
    code = np.floor((dB[:, 0] + UX0) / 1000).astype(int) * 10000 + np.floor((dB[:, 1] + UY0) / 1000).astype(int)
    detail_codes = [int(k.split("_")[0]) * 10000 + int(k.split("_")[1]) for k in meta["detail"]]
    keep[~np.isin(code, detail_codes)] = 0.15
B = B[rng_e.random(len(B)) < keep]
bs = np.clip(B[:, 3] * 0.9, 0.6, 1.8)
point_object("NRW_Bushes", np.c_[B[:, 0], B[:, 1], B[:, 2] + offset - 0.1], rng_e.choice([4, 5], len(B)),
             np.c_[bs, bs, np.clip(B[:, 3], 0.4, 3.0)], ng)
print("VEG trees", len(T), "bushes", len(B))
