"""Map labels on pins above key OSM places (viewport aid only: not rendered, not for export).
exec'd by build_blend.py and by the live link; needs NRW_Terrain and scene["lat"], scene["lon"]."""
import bpy, bmesh, json, math

sc = bpy.context.scene
LABEL_FACE_AZ = 212
OSM_EXTRA = r"A:\blender_work\wiehl_data\osm_extra.json"
MAJOR = {"supermarket", "school", "ice_rink", "stadium", "place_of_worship", "townhall", "bus_station", "station",
         "park", "water_park", "community_centre", "church", "fire_station", "library", "hostel", "hotel"}
MINOR = {"restaurant", "cafe", "fast_food", "pub", "bank", "pharmacy", "fuel", "post_office", "bakery", "chemist",
         "clothes", "shoes", "butcher", "theatre"}


def label_tm(lat, lon):
    R = 6378137.0
    la, lo = math.radians(lat), math.radians(lon - sc["lon"])
    b = math.sin(lo) * math.cos(la)
    return 0.5 * R * math.log((1 + b) / (1 - b)), R * (math.atan(math.tan(la) / math.cos(lo)) - math.radians(sc["lat"]))


def label_mat(name, rgb):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Emission Color"].default_value = (*rgb, 1)
    b.inputs["Emission Strength"].default_value = 3.0
    return m


old = bpy.data.collections.get("NRW_Labels")
if old:
    for o in list(old.objects):
        bpy.data.objects.remove(o)
    bpy.data.collections.remove(old)
lcol = bpy.data.collections.new("NRW_Labels")
sc.collection.children.link(lcol)
lcol.hide_render = True
mats = {"major": label_mat("NRW_Label_Major", (1.0, 0.8, 0.1)), "minor": label_mat("NRW_Label_Minor", (0.95, 0.95, 0.95)),
        "parking": label_mat("NRW_Label_Parking", (0.1, 0.35, 1.0))}
terrain = bpy.data.objects["NRW_Terrain"]
dg = bpy.context.evaluated_depsgraph_get()

seen, n = set(), 0
for e in json.load(open(OSM_EXTRA, encoding="utf-8"))["elements"]:
    t = e.get("tags", {})
    vals = {t.get(k) for k in ("shop", "amenity", "leisure", "tourism", "building", "railway")}
    if t.get("amenity") == "parking" and e["type"] == "way":
        kind, text = "parking", "P"
    elif vals & MAJOR:
        kind, text = "major", t.get("name") or ("Bahnhof" if t.get("railway") == "station" else None)
    elif vals & MINOR:
        kind, text = "minor", t.get("name")
    else:
        continue
    if not text or (kind != "parking" and text in seen):
        continue
    seen.add(text)
    pts = [(p["lat"], p["lon"]) for p in e["geometry"]] if "geometry" in e else [(e["lat"], e["lon"])]
    x, y = label_tm(sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    ok, hit, *_ = terrain.ray_cast((x, y, 3000), (0, 0, -1), depsgraph=dg)
    if not ok:
        continue
    gz = hit.z
    size, lift = {"major": (14, 55), "minor": (7, 32), "parking": (10, 20)}[kind]
    cu = bpy.data.curves.new("L_" + text, "FONT")
    cu.body, cu.size, cu.extrude = text, size, 0.4
    cu.align_x, cu.align_y = "CENTER", "CENTER"
    cu.materials.append(mats[kind])
    ob = bpy.data.objects.new("L_" + text, cu)
    ob.location = (x, y, gz + lift)
    # stand upright, facing the default view from the south-west (azimuth 212)
    ob.rotation_euler = (math.pi / 2, 0, math.radians(180 - LABEL_FACE_AZ))
    lcol.objects.link(ob)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=6, radius1=0.5, radius2=0.5, depth=lift)
    me = bpy.data.meshes.new("Pin_" + text)
    bm.to_mesh(me); bm.free()
    me.materials.append(mats[kind])
    pin = bpy.data.objects.new("Pin_" + text, me)
    pin.location = (x, y, gz + lift / 2)
    lcol.objects.link(pin)
    n += 1
for o in lcol.objects:
    o.hide_render = True
print("LABELS", n)
