"""Run from build_blend.py (exec) after build_streets.py: street lamps, bus stops with shelters, hedges, fences,
walls, railway tracks and the station platform.
Uses globals from build_blend/build_streets/building_details: ways, extra, width, ground, densify, frame, road_count,
inside_terrain, tm, MeshBuilder, CURB_H, CURB_W, foot, cen, in_poly, col, Nodes, TEX."""
import math
import numpy as np

LAMP_ROADS = {"primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link",
              "residential", "unclassified", "living_street"}
LAMP_SPACING = 32.0


def simple_material(name, rgb, rough=0.5, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    return m


def box(mb, c, ax, ay, az, mat=0):
    """box centred at c with half-axes ax, ay, az (numpy 3-vectors)"""
    corners = [c + sx * ax + sy * ay + sz * az for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
    for q in ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)):
        mb.face([corners[i] for i in q], mat)


def building_hit(p):
    near = np.nonzero(np.linalg.norm(cen - p, axis=1) < 60)[0]
    return any(in_poly(p[None], poly)[0] for b in near for poly in foot[b])


def place(obj_mesh, name, loc, rot_z, coll):
    ob = bpy.data.objects.new(name, obj_mesh)
    ob.location = loc
    ob.rotation_euler = (0, 0, rot_z)
    coll.objects.link(ob)
    return ob


props = bpy.data.collections.new("NRW_Props")
col.children.link(props)
METAL = simple_material("NRW_PoleMetal", (0.07, 0.075, 0.08), 0.4, 0.8)
GLASS_P = simple_material("NRW_ShelterGlass", (0.6, 0.7, 0.75), 0.05)
GLASS_P.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.25

# ---------------- street lamp (modern LED head on a 6 m pole, arm along +X) ----------------
lamp = MeshBuilder()
for z0, z1, r in ((0.0, 6.0, 0.065),):
    for k in range(8):
        a0, a1 = 2 * math.pi * k / 8, 2 * math.pi * (k + 1) / 8
        p = [(r * math.cos(a0), r * math.sin(a0), z0), (r * math.cos(a1), r * math.sin(a1), z0),
             (0.75 * r * math.cos(a1), 0.75 * r * math.sin(a1), z1), (0.75 * r * math.cos(a0), 0.75 * r * math.sin(a0), z1)]
        lamp.face(p)
box(lamp, np.array((0.65, 0, 5.95)), np.array((0.65, 0, 0)), np.array((0, 0.03, 0)), np.array((0, 0, 0.03)))
box(lamp, np.array((1.45, 0, 5.93)), np.array((0.3, 0, 0)), np.array((0, 0.13, 0)), np.array((0, 0, 0.045)), 1)
lamp_me = bpy.data.meshes.new("NRW_Lamp")
lamp_me.from_pydata(lamp.v, [], lamp.f)
lamp_me.polygons.foreach_set("material_index", np.array(lamp.m, np.int32))
lamp_me.materials.append(METAL)
lamp_me.materials.append(simple_material("NRW_LampHead", (0.55, 0.56, 0.58), 0.35, 0.6))

n_lamps = 0
for t, xy in ways:
    if t["highway"] not in LAMP_ROADS:
        continue
    pts = densify(xy, 1.0)
    tan, nrm, along = frame(pts)
    off = width(t) / 2 + CURB_W + 0.45
    for s in np.arange(10.0, along[-1], LAMP_SPACING):
        i = int(np.searchsorted(along, s))
        if i >= len(pts):
            break
        p = pts[i] - nrm[i] * off
        if not inside_terrain(p[None])[0] or road_count(p[None])[0] > 0 or building_hit(p):
            continue
        if REGION and np.hypot(*(cen - p).T).min() > 60:  # country roads between the villages are unlit
            continue
        place(lamp_me, "Lamp", (p[0], p[1], ground(p[None])[0] + CURB_H), math.atan2(nrm[i][1], nrm[i][0]), props)
        n_lamps += 1


# ---------------- bus stops: "H" sign on a pole, glass shelter where tagged ----------------
def h_sign_image():
    S = 256
    yy, xx = (np.mgrid[0:S, 0:S] + 0.5) / S - 0.5
    rgba = np.zeros((S, S, 4), np.float32)
    rr = np.hypot(xx, yy)
    rgba[rr < 0.48] = (0.9, 0.75, 0.05, 1)
    rgba[rr < 0.42] = (0.0, 0.35, 0.15, 1)
    h = (np.abs(yy) < 0.26) & ((np.abs(xx + 0.13) < 0.045) | (np.abs(xx - 0.13) < 0.045) | ((np.abs(xx) < 0.13) & (np.abs(yy) < 0.04)))
    rgba[h] = (0.9, 0.75, 0.05, 1)
    img = bpy.data.images.new("nrw_bus_sign", S, S, alpha=True)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = os.path.join(TEX, "nrw_bus_sign.png")
    img.file_format = "PNG"
    img.save()
    return img


sign_mat = bpy.data.materials.new("NRW_BusSign")
sign_mat.use_nodes = True
nn_s = Nodes(sign_mat)
st_tex = nn_s.new("ShaderNodeTexImage", image=h_sign_image())
nn_s.link(st_tex.outputs["Color"], nn_s.bsdf.inputs["Base Color"])
nn_s.link(st_tex.outputs["Alpha"], nn_s.bsdf.inputs["Alpha"])
stop = MeshBuilder()
box(stop, np.array((0, 0, 1.4)), np.array((0.035, 0, 0)), np.array((0, 0.035, 0)), np.array((0, 0, 1.4)))
stop.face([(0.04, -0.25, 2.35), (0.04, 0.25, 2.35), (0.04, 0.25, 2.85), (0.04, -0.25, 2.85)], 1,
          [(0, 0), (1, 0), (1, 1), (0, 1)])
stop_me = bpy.data.meshes.new("NRW_BusStopSign")
stop_me.from_pydata(stop.v, [], stop.f)
stop_me.polygons.foreach_set("material_index", np.array(stop.m, np.int32))
stop_me.uv_layers.new(name="UVMap").data.foreach_set("uv", np.array(stop.uv, np.float32).ravel())
stop_me.materials.append(METAL)
stop_me.materials.append(sign_mat)

shelter = MeshBuilder()  # open side towards -X (the road), 3 m long along Y
for py in (-1.5, 1.5):
    for px in (-0.7, 0.7):
        box(shelter, np.array((px, py, 1.2)), np.array((0.03, 0, 0)), np.array((0, 0.03, 0)), np.array((0, 0, 1.2)))
box(shelter, np.array((0.0, 0.0, 2.45)), np.array((0.85, 0, 0)), np.array((0, 1.65, 0)), np.array((0, 0, 0.05)))
shelter.face([(0.7, -1.5, 0.1), (0.7, 1.5, 0.1), (0.7, 1.5, 2.3), (0.7, -1.5, 2.3)], 1)
for py in (-1.5, 1.5):
    shelter.face([(-0.7, py, 0.1), (0.7, py, 0.1), (0.7, py, 2.3), (-0.7, py, 2.3)], 1)
box(shelter, np.array((0.35, 0.0, 0.45)), np.array((0.18, 0, 0)), np.array((0, 1.1, 0)), np.array((0, 0, 0.03)))
shelter_me = bpy.data.meshes.new("NRW_BusShelter")
shelter_me.from_pydata(shelter.v, [], shelter.f)
shelter_me.polygons.foreach_set("material_index", np.array(shelter.m, np.int32))
shelter_me.materials.append(METAL)
shelter_me.materials.append(GLASS_P)

seg_a = np.vstack([xy[:-1] for _, xy in ways])
seg_b = np.vstack([xy[1:] for _, xy in ways])
n_stops = 0
for e in extra:
    tg = e.get("tags", {})
    if e["type"] != "node" or tg.get("highway") != "bus_stop":
        continue
    p = tm(np.array([e["lat"]]), np.array([e["lon"]]))[0]
    if not inside_terrain(p[None])[0]:
        continue
    ab = seg_b - seg_a
    tpar = np.clip(((p - seg_a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    proj = seg_a + ab * tpar[:, None]
    i = int(np.argmin(np.linalg.norm(proj - p, axis=1)))
    away = p - proj[i]
    if np.linalg.norm(away) < 0.5:
        away = np.array((-ab[i][1], ab[i][0]))
    away /= np.linalg.norm(away)
    rot = math.atan2(away[1], away[0])  # local +X points away from the road, the shelter opens towards it
    z = ground(p[None])[0] + CURB_H
    place(stop_me, "BusStop_" + tg.get("name", ""), (p[0], p[1], z), rot, props)
    if tg.get("shelter") == "yes":
        q = p + away * 1.6
        place(shelter_me, "BusShelter_" + tg.get("name", ""), (q[0], q[1], ground(q[None])[0] + CURB_H), rot, props)
    n_stops += 1

# ---------------- hedges, fences, walls ----------------
hedge_mat = bpy.data.materials.new("NRW_Hedge")
hedge_mat.use_nodes = True
nn_h = Nodes(hedge_mat)
huv = nn_h.new("ShaderNodeTexCoord").outputs["Object"]
hmix = nn_h.new("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
hmix.inputs[0].default_value = 1.0
nn_h.link(nn_h.tex("leafy_grass", "Diffuse", huv), hmix.inputs[6])
hmix.inputs[7].default_value = (0.35, 0.6, 0.25, 1)
nn_h.link(hmix.outputs[2], nn_h.bsdf.inputs["Base Color"])
nn_h.normal(nn_h.tex("leafy_grass", "nor_gl", huv, False), 1.5)

fence_mat = bpy.data.materials.new("NRW_FenceMesh")
fence_mat.use_nodes = True
nn_f = Nodes(fence_mat)
fsep = nn_f.new("ShaderNodeSeparateXYZ")
nn_f.link(nn_f.uv("RoadUV"), fsep.inputs[0])
wire = nn_f.math("MAXIMUM", nn_f.math("LESS_THAN", nn_f.math("FRACT", nn_f.math("DIVIDE", fsep.outputs[0], 0.2)), 0.08),
                 nn_f.math("LESS_THAN", nn_f.math("FRACT", nn_f.math("DIVIDE", fsep.outputs[1], 0.2)), 0.08))
nn_f.link(wire, nn_f.bsdf.inputs["Alpha"])
nn_f.bsdf.inputs["Base Color"].default_value = (0.03, 0.09, 0.04, 1)

barriers = MeshBuilder()
wall_mat = paving_material("NRW_GardenWall", "brushed_concrete", "RoadUV", 0.3)
n_bar = 0
for e in extra:
    kind = e.get("tags", {}).get("barrier")
    if e["type"] != "way" or kind not in ("hedge", "fence", "wall") or "geometry" not in e:
        continue
    xy = tm(np.array([p["lat"] for p in e["geometry"]]), np.array([p["lon"] for p in e["geometry"]]))
    pts = densify(xy, 0.5 if kind != "fence" else 2.5)
    ins = inside_terrain(pts)
    tan, nrm, along = frame(pts)
    gz = ground(pts)
    if kind == "fence":
        for i in range(len(pts)):
            if ins[i]:
                box(barriers, np.array((*pts[i], gz[i] + 0.6)), np.array((0.03, 0, 0)), np.array((0, 0.03, 0)), np.array((0, 0, 0.6)), 1)
        for i in range(len(pts) - 1):
            if ins[i] and ins[i + 1]:
                q = [(*pts[i], gz[i] + 0.05), (*pts[i + 1], gz[i + 1] + 0.05), (*pts[i + 1], gz[i + 1] + 1.2), (*pts[i], gz[i] + 1.2)]
                barriers.face(q, 2, [(along[i], 0.05), (along[i + 1], 0.05), (along[i + 1], 1.2), (along[i], 1.2)])
    else:
        h, hw = (1.4, 0.4) if kind == "hedge" else (1.0, 0.15)
        if e["tags"].get("wall") == "noise_barrier":
            h, hw = 3.5, 0.2  # the Autobahn's Lärmschutzwände
        mat = 0 if kind == "hedge" else 3
        L, R = pts + nrm * hw, pts - nrm * hw
        for i in range(len(pts) - 1):
            if not (ins[i] and ins[i + 1]):
                continue
            a0, a1 = along[i], along[i + 1]
            barriers.face([(*R[i], gz[i] + h), (*R[i + 1], gz[i + 1] + h), (*L[i + 1], gz[i + 1] + h), (*L[i], gz[i] + h)], mat)
            barriers.face([(*R[i], gz[i] - 0.1), (*R[i + 1], gz[i + 1] - 0.1), (*R[i + 1], gz[i + 1] + h), (*R[i], gz[i] + h)], mat,
                          [(a0, 0), (a1, 0), (a1, h), (a0, h)])
            barriers.face([(*L[i + 1], gz[i + 1] - 0.1), (*L[i], gz[i] - 0.1), (*L[i], gz[i] + h), (*L[i + 1], gz[i + 1] + h)], mat,
                          [(a1, 0), (a0, 0), (a0, h), (a1, h)])
    n_bar += 1
# Autobahn guard rails (Schutzplanken) along both edges of every carriageway: a galvanised band on posts every 8 m,
# at the road's own height (on an embankment the rail stays with the road, the posts reach down)
RAIL = simple_material("NRW_GuardRail", (0.55, 0.56, 0.57), 0.35, 0.8)
# the outer rails (right of the travel direction) are solid: an invisible wall from below the road to 1.2 m, both
# faces, exported as collision only (the user, 2026-09-27; the median ones may stay passable)
rail_wall = MeshBuilder()
n_rail_m = 0.0
for t, xy in ways:
    if t["highway"] not in MOTORWAY:
        continue
    pts = densify(xy, 2.0)
    if len(pts) < 2 or not inside_terrain(pts).any():
        continue
    tan, nrm, along = frame(pts)
    w2 = width(t) / 2
    RZ_LAYER = 1 if t["highway"] in MW_MAIN else 0  # the road's own height layer (build_streets)
    zr = ground_road(pts) + CLASS_Z[t["highway"]]
    RZ_LAYER = 0
    for s in (-1, 1):
        c = pts + s * nrm * (w2 + 0.6)
        ok = inside_terrain(c) & (road_count(c) == 0)
        for i in range(len(c) - 1):
            if not (ok[i] and ok[i + 1]):
                continue
            a, b = c[i], c[i + 1]
            for z0, z1 in ((0.45, 0.8),):
                q = [(*a, zr[i] + z0), (*b, zr[i + 1] + z0), (*b, zr[i + 1] + z1), (*a, zr[i] + z1)]
                barriers.face(q, 4, [(along[i], z0), (along[i + 1], z0), (along[i + 1], z1), (along[i], z1)])
                barriers.face(q[::-1], 4, [(along[i], z1), (along[i + 1], z1), (along[i + 1], z0), (along[i], z0)][::-1])
            if i % 4 == 0:
                box(barriers, np.array((*a, zr[i] + 0.2)), np.array((0.05, 0, 0)), np.array((0, 0.05, 0)), np.array((0, 0, 0.6)), 4)
            if s == -1:
                w = [(*a, zr[i] - 0.3), (*b, zr[i + 1] - 0.3), (*b, zr[i + 1] + 1.2), (*a, zr[i] + 1.2)]
                rail_wall.face(w)
                rail_wall.face(w[::-1])
            n_rail_m += float(np.linalg.norm(b - a))
print("GUARD_RAILS km", round(n_rail_m / 1000, 1), "solid faces", len(rail_wall.f))
if rail_wall.f:
    rail_wall.build("NRW_RailWall", "RoadUV", [METAL])
if barriers.f:
    barriers.build("NRW_Barriers", "RoadUV", [hedge_mat, METAL, fence_mat, wall_mat, RAIL])

# ---------------- railway: rails on concrete sleepers, station platform ----------------
track = MeshBuilder()
n_rail = 0
for e in extra:
    tg = e.get("tags", {})
    if e["type"] != "way" or tg.get("railway") != "rail" or "geometry" not in e:
        continue
    xy = tm(np.array([p["lat"] for p in e["geometry"]]), np.array([p["lon"] for p in e["geometry"]]))
    pts = densify(xy, 0.6)
    ins = inside_terrain(pts)
    tan, nrm, along = frame(pts)
    gz = ground(pts)
    for i in range(len(pts)):
        if ins[i]:
            box(track, np.array((*pts[i], gz[i] + 0.06)), np.array((*nrm[i], 0)) * 1.3, np.array((*tan[i], 0)) * 0.13,
                np.array((0, 0, 0.07)), 1)
    for s in (-0.7175, 0.7175):
        rl = pts + nrm * s
        L, R = rl + nrm * 0.035, rl - nrm * 0.035
        for i in range(len(pts) - 1):
            if not (ins[i] and ins[i + 1]):
                continue
            top = (gz[i] + 0.28, gz[i + 1] + 0.28)
            bot = (gz[i] + 0.12, gz[i + 1] + 0.12)
            track.face([(*R[i], top[0]), (*R[i + 1], top[1]), (*L[i + 1], top[1]), (*L[i], top[0])], 0)
            track.face([(*R[i], bot[0]), (*R[i + 1], bot[1]), (*R[i + 1], top[1]), (*R[i], top[0])], 0)
            track.face([(*L[i + 1], bot[1]), (*L[i], bot[0]), (*L[i], top[0]), (*L[i + 1], top[1])], 0)
    n_rail += 1
for e in extra:
    tg = e.get("tags", {})
    if e["type"] != "way" or tg.get("railway") != "platform" or "geometry" not in e:
        continue
    g = e["geometry"]
    if (g[0]["lat"], g[0]["lon"]) != (g[-1]["lat"], g[-1]["lon"]):
        continue
    poly = tm(np.array([p["lat"] for p in g]), np.array([p["lon"] for p in g]))[:-1]
    top_z = ground(poly).max() + 0.55
    for tri in mgeom.tessellate_polygon([[Vector((x, y, 0)) for x, y in poly]]):
        pt3 = poly[list(tri)]
        if np.cross(pt3[1] - pt3[0], pt3[2] - pt3[0]) < 0:
            pt3 = pt3[::-1]
        track.face([(*q, top_z) for q in pt3], 2)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        za, zb = ground(np.array([a, b]))
        track.face([(*a, za - 0.2), (*b, zb - 0.2), (*b, top_z), (*a, top_z)], 3)
if track.f:
    track.build("NRW_Railway", "RoadUV", [simple_material("NRW_RailSteel", (0.32, 0.3, 0.28), 0.35, 0.9),
                                          simple_material("NRW_Sleeper", (0.42, 0.41, 0.39), 0.8),
                                          paving_material("NRW_PlatformTop", "concrete_pavers_02", "RoadUV", 0.15),
                                          paving_material("NRW_PlatformEdge", "brushed_concrete", "RoadUV", 0.2)])
print("PROPS lamps", n_lamps, "bus_stops", n_stops, "barriers", n_bar, "rails", n_rail)


# ---------------- traffic signs: give way / stop where a side street meets a main road, yellow town signs
# (Ortstafel) where the main roads enter the towns. OSM has neither here; this is how German roads are signed ----
def sign_image(name, draw):
    S = 256
    yy, xx = (np.mgrid[0:S, 0:S] + 0.5) / S - 0.5
    yy = -yy  # image rows run down, pixels.foreach_set up
    rgba = np.zeros((S, S, 4), np.float32)
    draw(xx, yy, rgba)
    img = bpy.data.images.new(name, S, S, alpha=True)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = os.path.join(TEX, name + ".png")
    img.file_format = "PNG"
    img.save()
    return img


def draw_give_way(xx, yy, rgba):
    # a triangle standing on its point: red rim, white face
    def inside(m):
        return (yy <= 0.33 - m) & (np.abs(xx) <= (yy + 0.42 - 2 * m) / 1.732)
    rgba[inside(0.0)] = (0.75, 0.02, 0.03, 1)
    rgba[inside(0.07)] = (0.95, 0.95, 0.93, 1)


GLYPHS = {"S": ["01110", "10000", "10000", "01110", "00001", "00001", "11110"],
          "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
          "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
          "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"]}


def draw_stop(xx, yy, rgba):
    oct_ = np.maximum(np.maximum(np.abs(xx), np.abs(yy)), (np.abs(xx) + np.abs(yy)) / 1.414)
    rgba[oct_ < 0.47] = (0.95, 0.95, 0.93, 1)
    rgba[oct_ < 0.44] = (0.75, 0.02, 0.03, 1)
    for k, ch in enumerate("STOP"):  # white block letters, 5 x 7 cells
        x0 = -0.3 + k * 0.155
        for r, row in enumerate(GLYPHS[ch]):
            for c, bit in enumerate(row):
                if bit == "1":
                    cell = (xx >= x0 + c * 0.024) & (xx < x0 + (c + 1) * 0.024) & \
                           (yy <= 0.085 - r * 0.024) & (yy > 0.085 - (r + 1) * 0.024)
                    rgba[cell] = (0.95, 0.95, 0.93, 1)


def sign_material(name, img):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nn_ = Nodes(m)
    t = nn_.new("ShaderNodeTexImage", image=img)
    nn_.link(t.outputs["Color"], nn_.bsdf.inputs["Base Color"])
    nn_.link(t.outputs["Alpha"], nn_.bsdf.inputs["Alpha"])
    return m


def pole_sign(name, mat, size, z):
    """a grey pole with a square picture plate facing +X (cut out by the picture's alpha)"""
    mb = MeshBuilder()
    box(mb, np.array((0, 0, z / 2)), np.array((0.03, 0, 0)), np.array((0, 0.03, 0)), np.array((0, 0, z / 2)))
    h = size / 2
    mb.face([(0.04, -h, z - h), (0.04, h, z - h), (0.04, h, z + h), (0.04, -h, z + h)], 1, [(0, 0), (1, 0), (1, 1), (0, 1)])
    mb.face([(0.035, h, z - h), (0.035, -h, z - h), (0.035, -h, z + h), (0.035, h, z + h)], 0, [(0, 0), (1, 0), (1, 1), (0, 1)])
    me = bpy.data.meshes.new(name)
    me.from_pydata(mb.v, [], mb.f)
    me.polygons.foreach_set("material_index", np.array(mb.m, np.int32))
    me.uv_layers.new(name="UVMap").data.foreach_set("uv", np.array(mb.uv, np.float32).ravel())
    me.materials.append(METAL)
    me.materials.append(mat)
    return me


give_way_me = pole_sign("NRW_SignGiveWay", sign_material("NRW_SignGiveWay", sign_image("nrw_sign_giveway", draw_give_way)), 0.9, 2.2)
stop_sign_me = pole_sign("NRW_SignStop", sign_material("NRW_SignStop", sign_image("nrw_sign_stop", draw_stop)), 0.9, 2.2)
MAIN_ROADS = {"primary", "secondary", "tertiary", "primary_link", "secondary_link", "tertiary_link"}
main_nodes = {(round(float(x), 1), round(float(y), 1)) for t, xy in ways if t["highway"] in MAIN_ROADS for x, y in xy}
rng_sign = np.random.default_rng(23)
n_give = 0
for t, xy in ways:
    if t["highway"] not in ("residential", "unclassified", "living_street"):
        continue
    for end, nxt in ((0, 1), (-1, -2)):
        j = xy[end]
        if (round(float(j[0]), 1), round(float(j[1]), 1)) not in main_nodes:
            continue
        pts = densify(xy[::-1] if end == 0 else xy, 1.0)  # running towards the junction
        if len(pts) < 12:
            continue
        i = len(pts) - 9                                  # 8 m before it
        d = pts[-1] - pts[-12]
        d /= max(np.linalg.norm(d), 1e-6)
        right = np.array((d[1], -d[0]))
        p = pts[i] + right * (width(t) / 2 + 0.7)
        if not inside_terrain(p[None])[0] or road_count(p[None])[0] > 0 or building_hit(p):
            continue
        me_s = stop_sign_me if rng_sign.random() < 0.15 else give_way_me
        place(me_s, "Sign", (p[0], p[1], ground(p[None])[0] + CURB_H), math.atan2(-d[1], -d[0]), props)
        n_give += 1

# Ortstafel: yellow plate, black rim, the town's name, where a main road crosses into the town
YELLOW = simple_material("NRW_TownSignYellow", (0.9, 0.62, 0.02), 0.5)
BLACK = simple_material("NRW_TownSignBlack", (0.02, 0.02, 0.02), 0.6)
TOWN_SIGNS = [("Wiehl", (50.9470, 7.5496), 1300), ("Gummersbach-Dieringhausen", (50.9850, 7.5370), 900),
              ("Gummersbach", (51.0240, 7.5656), 1600), ("Wiehl-Bielstein", (50.9570, 7.4950), 700),
              ("Wiehl-Alferzhagen", (50.9870, 7.5760), 450), ("Reichshof-Brüchermühle", (50.9265, 7.6380), 500)]
n_town = 0
for town, (la, lo), rad in TOWN_SIGNS:
    c = tm(np.array([la]), np.array([lo]))[0]
    placed = []
    for t, xy in ways:
        if t["highway"] not in MAIN_ROADS:
            continue
        pts = densify(xy, 2.0)
        dist = np.hypot(*(pts - c).T)
        for i in np.nonzero((dist[:-1] - rad) * (dist[1:] - rad) < 0)[0]:
            d = pts[i + 1] - pts[i] if dist[i + 1] < dist[i] else pts[i] - pts[i + 1]  # driving into the town
            d /= max(np.linalg.norm(d), 1e-6)
            right = np.array((d[1], -d[0]))
            p = pts[i] + right * (width(t) / 2 + 1.2)
            if any(np.hypot(*(p - q)) < 60 for q in placed) or not inside_terrain(p[None])[0] or road_count(p[None])[0] > 0:
                continue
            placed.append(p)
            z = float(ground(p[None])[0])
            w = max(1.6, 0.16 * len(town) + 0.5)
            mb = MeshBuilder()
            for s in (-1, 1):  # two posts
                box(mb, np.array((0, s * w * 0.35, z + 1.0)), np.array((0.035, 0, 0)), np.array((0, 0.035, 0)), np.array((0, 0, 1.0)))
            box(mb, np.array((0, 0, z + 1.75)), np.array((0.03, 0, 0)), np.array((0, w / 2, 0)), np.array((0, 0, 0.45)), 2)
            box(mb, np.array((0.004, 0, z + 1.75)), np.array((0.03, 0, 0)), np.array((0, w / 2 - 0.04, 0)), np.array((0, 0, 0.41)), 1)
            me_t = bpy.data.meshes.new("NRW_TownSign")
            me_t.from_pydata(mb.v, [], mb.f)
            me_t.polygons.foreach_set("material_index", np.array(mb.m, np.int32))
            me_t.materials.append(METAL)
            me_t.materials.append(YELLOW)
            me_t.materials.append(BLACK)
            rot = math.atan2(-d[1], -d[0])
            place(me_t, "TownSign_" + town, (p[0], p[1], 0.0), rot, props)
            cu = bpy.data.curves.new("TownName_" + town, "FONT")
            cu.body, cu.size, cu.extrude = town if len(town) < 14 else town.replace("-", "-\n"), 0.28, 0.005
            cu.align_x, cu.align_y = "CENTER", "CENTER"
            cu.font = FONT_BOLD
            cu.materials.append(BLACK)
            ob = bpy.data.objects.new("TownName_" + town, cu)
            ob.location = (p[0] + math.cos(rot) * 0.04, p[1] + math.sin(rot) * 0.04, z + 1.75)
            ob.rotation_euler = (math.pi / 2, 0, rot + math.pi / 2)
            signs.objects.link(ob)
            n_town += 1
print("SIGNS give_way/stop", n_give, "town", n_town)
