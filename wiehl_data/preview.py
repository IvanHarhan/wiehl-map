import bpy, sys, math
from mathutils import Vector
out, mode = sys.argv[sys.argv.index("--") + 1:][:2]
sc = bpy.context.scene
g = bpy.data.objects.get("Google 3D Tiles")
if mode == "compare" and g:
    g.hide_set(False); g.hide_render = False
    bpy.data.objects["NRW_Terrain"].hide_render = True
cam = bpy.data.objects.new("PreviewCam", bpy.data.cameras.new("PreviewCam"))
sc.collection.objects.link(cam)
t = bpy.data.objects["NRW_Terrain"]
tz = sum((t.matrix_world @ Vector(c)).z for c in t.bound_box) / 8
target = Vector((20, 0, tz))
if mode == "church":
    bm_ = bpy.data.objects["NRW_Buildings"].data
    kinds = [d.value for d in bm_.attributes["kind"].data]
    hs = [d.value for d in bm_.attributes["bheight"].data]
    bids = [d.value for d in bm_.attributes["bid"].data]
    tallest = max((h, b) for k, h, b in zip(kinds, hs, bids) if k > 2.5)[1]
    pts = [p.center for p, b in zip(bm_.polygons, bids) if b == tallest]
    target = sum(pts, Vector()) / len(pts)
offsets = {"church": (-45, -60, 25),"top": (0, 0, 1500), "close": (-70, -110, 45)}
cam.location = target + Vector(offsets.get(mode, (-260, -420, 260)))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
cam.data.lens = 35
cam.data.clip_end = 20000
sc.camera = cam
if "NRW_Sun" not in bpy.data.objects:
    sun = bpy.data.objects.new("PreviewSun", bpy.data.lights.new("PreviewSun", "SUN"))
    sun.data.energy = 4
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    sc.collection.objects.link(sun)
    if sc.world is None:
        sc.world = bpy.data.worlds.new("W")
    sc.world.color = (0.6, 0.7, 0.85)
sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = 1280, 720
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
print("PREVIEW", out)
