"""Viewing copy: whole scene scaled 1:10 under one empty; the working file stays at real scale."""
import bpy, sys
out = sys.argv[sys.argv.index("--") + 1]
root = bpy.data.objects.new("VIEW_SCALE_0.1", None)
bpy.context.scene.collection.objects.link(root)
for ob in bpy.data.objects:
    if ob is not root and ob.parent is None and ob.type not in {"CAMERA", "LIGHT"}:
        mw = ob.matrix_world.copy()
        ob.parent = root
        ob.matrix_world = mw
root.scale = (0.1, 0.1, 0.1)
for scr in bpy.data.screens:
    for area in scr.areas:
        for sp in area.spaces:
            if sp.type == "VIEW_3D":
                sp.clip_start, sp.clip_end = 0.01, 5000
bpy.ops.wm.save_as_mainfile(filepath=out, compress=True, copy=True)
print("SMALL", out)
