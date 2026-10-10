"""Bring the exported A4 chunks (unity_export/a4_K) into dexstand next to the main region: tile FBX and their
orthophoto crops into Assets/Maps/Import, trees and parked cars merged with the main region's into Resources/Map.
Unity then needs DexstandSceneBuilder.StreamNewTiles (trigger "streamnew") to stream the new tiles."""
import glob, json, os, shutil
import numpy as np

W = r"A:\blender_work"
IMP = r"C:\dexstand\Assets\Maps\Import"
RES = r"C:\dexstand\Assets\Resources\Map"
MAIN = os.path.join(W, "unity_export", "region")
chunks = sorted(c for c in glob.glob(os.path.join(W, "unity_export", "a4_*")) if glob.glob(os.path.join(c, "Tile_*.fbx")))
print("chunks", [os.path.basename(c) for c in chunks])

n_tiles = n_tex = 0
for c in chunks:
    for f in glob.glob(os.path.join(c, "Tile_*.fbx")):
        shutil.copy(f, IMP)
        n_tiles += 1
    for f in glob.glob(os.path.join(c, "Textures", "*")):
        dst = os.path.join(IMP, "Textures", os.path.basename(f))
        # shared facade/street textures are there already; copying them again would only reimport them
        if not os.path.exists(dst):
            shutil.copy(f, dst)
            n_tex += 1

trees = [np.fromfile(os.path.join(d, "trees.bytes"), np.float32) for d in [MAIN] + chunks
         if os.path.exists(os.path.join(d, "trees.bytes"))]
allt = np.concatenate(trees)
assert allt.size % 8 == 0
allt.tofile(os.path.join(RES, "trees.bytes"))


def cars(path):
    return json.load(open(path))["cars"] if os.path.exists(path) else []


for name in ("parked", "parked_drive_0", "parked_drive_1", "parked_drive_2"):
    merged = []
    for d in [MAIN] + chunks:
        merged += cars(os.path.join(d, name + ".json"))
    json.dump({"cars": merged}, open(os.path.join(RES, name + ".json"), "w"))
print("MERGE_DONE tiles", n_tiles, "new textures", n_tex, "trees", allt.size // 8)
