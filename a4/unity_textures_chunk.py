"""Orthophoto per 500 m map cell for Unity: each 1 km tile of prepared/ortho is cut into its four quarters with a
32 m overlap inside the tile (roofs that cross a cell edge still find their pixels), at most 2048 px.
Writes <out>/Textures/ortho_<E>_<N>_<q>.jpg (q = qx + 2 * qy, qx 0 = west, qy 0 = south) and cells.json with
each cell's crop in tile metres, which export_region.py needs for the UVs. Also copies the facade textures.
Usage: python unity_textures_region.py OUT_DIR"""
import glob, json, os, shutil, sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = sys.argv[2]
P = os.path.join(D, "prepared")
OUT = sys.argv[1]
FORCE = "--force" in sys.argv  # redo crops that already exist (after a colour change)
TEX_OUT = os.path.join(OUT, "Textures")
os.makedirs(TEX_OUT, exist_ok=True)
MARGIN, MAX_PX = 32, 2048
FACADE_TEX = [("painted_plaster_wall", ("Diffuse", "nor_gl")), ("red_brick_03", ("Diffuse", "nor_gl")),
              ("painted_metal_shutter", ("Diffuse",))]

cells = {}
for f in sorted(glob.glob(os.path.join(P, "ortho", "*.jpg"))):
    key = os.path.splitext(os.path.basename(f))[0]
    im = None
    for qy in (0, 1):
        for qx in (0, 1):
            q = qx + 2 * qy
            name = f"ortho_{key}_{q}"
            x0, x1 = max(0, qx * 500 - MARGIN), min(1000, qx * 500 + 500 + MARGIN)
            y0, y1 = max(0, qy * 500 - MARGIN), min(1000, qy * 500 + 500 + MARGIN)
            cells[f"{key}_{q}"] = {"x0": x0, "x1": x1, "y0": y0, "y1": y1}
            out = os.path.join(TEX_OUT, name + ".jpg")
            if os.path.exists(out) and not FORCE:
                continue
            if im is None:
                im = Image.open(f)
                im.load()
            s = im.size[0] / 1000.0
            crop = im.crop((round(x0 * s), round((1000 - y1) * s), round(x1 * s), round((1000 - y0) * s)))
            if max(crop.size) > MAX_PX:
                k = MAX_PX / max(crop.size)
                crop = crop.resize((round(crop.size[0] * k), round(crop.size[1] * k)), Image.LANCZOS)
            # the summer orthophoto is pale and bluish: deepen the midtones and warm it a touch, as the Blender ground
            # material does (gamma 1.2), so paving and lawns do not read as white at street level
            a = (np.asarray(crop, np.float32) / 255.0) ** 1.25
            a *= np.array([0.92, 0.9, 0.84], np.float32)
            Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out, quality=88)
    print("cells of", key, flush=True)
json.dump(cells, open(os.path.join(OUT, "cells.json"), "w"), indent=0)

src = r"A:\blender_work\textures"
for asset, kinds in FACADE_TEX:
    for k in kinds:
        f = os.path.join(src, f"{asset}_{k}.jpg")
        if os.path.exists(f):
            shutil.copy(f, os.path.join(TEX_OUT, f"{asset}_{k}.jpg"))
for f in glob.glob(r"A:\blender_work\unity_export\tex\*.*"):  # street textures of the old export (pavers, curbs, ...)
    shutil.copy(f, TEX_OUT)
for f in ("asphalt_02_Diffuse.jpg", "asphalt_02_nor_gl.jpg", "brushed_concrete_nor_gl.jpg", "concrete_pavers_02_nor_gl.jpg",
          "leafy_grass_nor_gl.jpg", "nrw_bus_sign.png"):
    if os.path.exists(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), TEX_OUT)
print("TEXTURES_DONE cells", len(cells))
