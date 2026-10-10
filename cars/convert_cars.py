"""Sketchfab cars (glTF, CC-BY) -> one light FBX each for Unity's Resources/Cars (parked cars, traffic, drivable).
Joined into one mesh, decimated to TARGET triangles, turned so the nose points along -Y (Blender), which the FBX
export (-Z forward, Y up) turns into Unity +Z, wheels on the ground, length kept real.
Run: blender -b --python convert_cars.py -- SRC_DIR OUT_DIR [preview]
FRONT_YAW: per car, the turn (degrees about Z) that brings its nose to -Y, found from the previews."""
import bpy, math, os, sys
from mathutils import Vector, Matrix

ARGS = sys.argv[sys.argv.index("--") + 1:]
SRC, OUT = ARGS[0], ARGS[1]
PREVIEW = len(ARGS) > 2 and ARGS[2] == "preview"
TARGET = 5000
FRONT_YAW = {"moskvich-408": 180.0}  # from the previews and the user's drive: the Moskvich came in backwards
LENGTH_M = {"skoda-octavia": 4.69, "vw-passat": 4.77, "bmw-e46": 4.47, "mercedes-g": 4.6, "moskvich-408": 4.09}
NAMES = {"low-poly_skoda_octavia_rs_2020": "skoda-octavia", "low-poly_volkswagen_passat_b8": "vw-passat",
         "low_poly_car_-_bmw_e46_1998": "bmw-e46", "mercedes-benz_g-class_free_download": "mercedes-g",
         "32c17732fa9f4ff2bf8d5055400f1b62": "moskvich-408"}
os.makedirs(OUT, exist_ok=True)

for folder, name in NAMES.items():
    path = os.path.join(SRC, folder, "scene.gltf")
    if not os.path.exists(path):
        continue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    # bake every parent transform in, then one object
    for o in meshes:
        o.data = o.data.copy()
        o.data.transform(o.matrix_world)
        o.parent = None
        o.matrix_world = Matrix.Identity(4)
    for o in [o for o in bpy.context.scene.objects if o.type != "MESH"]:
        bpy.data.objects.remove(o)
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    car = bpy.context.view_layer.objects.active
    car.name = name
    # glTF materials reach the colour through node chains the FBX export cannot follow (Unity got white cars):
    # each becomes a plain Principled BSDF with its colour or its picture wired straight in, pictures saved as
    # PNG beside the FBX (Unity finds them in Textures/), no metal (without reflections it paints bodies black)
    tex_dir = os.path.join(OUT, "Textures")
    os.makedirs(tex_dir, exist_ok=True)
    for slot in car.material_slots:
        m = slot.material
        if m is None or not m.use_nodes:
            continue
        bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        colour = tuple(bsdf.inputs["Base Color"].default_value)
        image = None
        stack = [l.from_node for l in bsdf.inputs["Base Color"].links]
        while stack and image is None:
            n = stack.pop()
            if n.type == "TEX_IMAGE" and n.image is not None:
                image = n.image
            else:
                stack += [l.from_node for i in n.inputs for l in i.links]
        rough = bsdf.inputs["Roughness"].default_value
        alpha = bsdf.inputs["Alpha"].default_value
        nt = m.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        p = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(p.outputs["BSDF"], out.inputs["Surface"])
        p.inputs["Base Color"].default_value = colour
        p.inputs["Metallic"].default_value = 0.0
        p.inputs["Roughness"].default_value = min(max(rough, 0.25), 0.8)
        p.inputs["Alpha"].default_value = alpha
        if image is not None:
            fname = f"{name}_{bpy.path.clean_name(image.name)}.png"
            image.filepath_raw = os.path.join(tex_dir, fname)
            image.file_format = "PNG"
            image.save()
            t = nt.nodes.new("ShaderNodeTexImage")
            t.image = image
            nt.links.new(t.outputs["Color"], p.inputs["Base Color"])
        m.name = f"{name}_{m.name}"
    tris = sum(len(p.vertices) - 2 for p in car.data.polygons)
    if tris > TARGET:
        m = car.modifiers.new("Decimate", "DECIMATE")
        m.ratio = TARGET / tris
        m.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=m.name)
    # longest horizontal side along Y, then the per-car turn, real length, wheels on z = 0, centred
    def extent():
        vs = [v.co for v in car.data.vertices]
        return (Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs))),
                Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs))))
    lo, hi = extent()
    yaw = (90.0 if hi.x - lo.x > hi.y - lo.y else 0.0) + FRONT_YAW.get(name, 0.0)
    car.data.transform(Matrix.Rotation(math.radians(yaw), 4, "Z"))
    lo, hi = extent()
    car.data.transform(Matrix.Scale(LENGTH_M.get(name, 4.5) / (hi.y - lo.y), 4))
    lo, hi = extent()
    car.data.transform(Matrix.Translation(-Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))))
    lo, hi = extent()
    after = sum(len(p.vertices) - 2 for p in car.data.polygons)
    print("CAR", name, "tris", tris, "->", after, "size", tuple(round(v, 2) for v in (hi - lo)), flush=True)

    if PREVIEW:
        sc = bpy.context.scene
        cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
        sc.collection.objects.link(cam)
        L = max(hi - lo)
        cam.location = (L * 2.2, 0, (hi.z - lo.z) * 0.5)
        cam.rotation_euler = (math.radians(90), 0, math.radians(90))
        sc.camera = cam
        light = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
        light.rotation_euler = (0.6, 0.3, 0.8)
        sc.collection.objects.link(light)
        sc.render.engine = "BLENDER_WORKBENCH"
        sc.display.shading.color_type = "TEXTURE"
        sc.render.resolution_x, sc.render.resolution_y = 640, 320
        sc.render.filepath = os.path.join(OUT, f"preview_{name}.png")
        bpy.ops.render.render(write_still=True)
        continue

    bpy.ops.object.select_all(action="DESELECT")
    car.select_set(True)
    bpy.ops.export_scene.fbx(filepath=os.path.join(OUT, name + ".fbx"), use_selection=True, object_types={"MESH"},
                             apply_scale_options="FBX_SCALE_ALL", axis_forward="-Z", axis_up="Y",
                             path_mode="STRIP", embed_textures=False, mesh_smooth_type="FACE", add_leaf_bones=False,
                             bake_anim=False)
