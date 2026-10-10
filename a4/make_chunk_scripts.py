"""Chunk copies of region_data/prep_region.py and prep_veg_region.py: tiles from a4/tiles, the chunk dir from argv,
buildings only in the 600 m detail band, no Wiehl towns; meta carries the main region's height offset and origin."""
src = open(r"A:\blender_work\region_data\prep_region.py", encoding="utf-8").read()
rep = [
    ('D = os.path.dirname(os.path.abspath(__file__))\nT = os.path.join(D, "tiles")',
     'D = sys.argv[1]\nT = r"A:\\blender_work\\a4\\tiles"'),
    ('CORE = {(397, 5644), (397, 5645), (398, 5644), (398, 5645)}', 'CORE = set()'),
    ('DETAIL = {(e, n) for e in range(396, 400) for n in range(5644, 5647)} | {(e, n) for e in (404, 405) for n in (5641, 5642)}',
     'DETAIL = set()'),
    ('TOWNS = [(50.9875, 7.5755, 900),', 'TOWNS = []\n_OLD_TOWNS = [(50.9875, 7.5755, 900),'),
    ('tmask, tE0, tN1, tres = territory["mask"], int(territory["E0"]), int(territory["N1"]), int(territory["res"])',
     'tmask, tE0, tN1, tres = territory["mask"], int(territory["E0"]), int(territory["N1"]), int(territory["res"])\n'
     'dmask = territory["detail"]  # buildings only within 600 m of the A4'),
    ('    return tmask[max(0, r - grow):r + grow + 1, max(0, c - grow):c + grow + 1].any()',
     '    return dmask[max(0, r - grow):r + grow + 1, max(0, c - grow):c + grow + 1].any()'),
    ('np.savez_compressed(os.path.join(OUT, "territory.npz"), mask=tmask, E0=tE0, N1=tN1, res=tres)',
     'np.savez_compressed(os.path.join(OUT, "territory.npz"), mask=tmask, detail=dmask, E0=tE0, N1=tN1, res=tres)'),
    ('"detail": sorted(f"{e}_{n}" for e, n in DETAIL), "buildings": n_bld, "faces": len(face_len)}',
     '"detail": sorted(f"{e}_{n}" for e, n in DETAIL), "buildings": n_bld, "faces": len(face_len),\n'
     '        "chunk": {"height_offset": 46.290374755859375, "origin": [50.747467041015625, -19.01202964782715, 233.75]}}'),
]
for a, b in rep:
    assert a in src, a[:60]
    src = src.replace(a, b)
# the 600 m band gets the finer terrain and orthophoto, like the towns
src = src.replace('print("detail tiles", len(DETAIL & set(tiles)), flush=True)',
                  'DETAIL |= {(int(e), int(n)) for e, n in np.load(r"A:\\blender_work\\a4\\corridor.npz")["detail_tiles"]}\n'
                  'print("detail tiles", len(DETAIL & set(tiles)), flush=True)')
assert 'corridor.npz' in src
open(r"A:\blender_work\a4\prep_chunk.py", "w", encoding="utf-8").write(src)

v = open(r"A:\blender_work\region_data\prep_veg_region.py", encoding="utf-8").read()
rep = [
    ('D = os.path.dirname(os.path.abspath(__file__))\nT, P = os.path.join(D, "tiles"), os.path.join(D, "prepared")',
     'import sys\nD = sys.argv[1]\nT, P = r"A:\\blender_work\\a4\\tiles", os.path.join(D, "prepared")'),
    # every tree is kept only near buildings; the corridor's woods are thinned like the region's countryside
    ('DETAIL = set(meta["detail"])', 'DETAIL = set()'),
]
for a, b in rep:
    assert a in v, a[:60]
    v = v.replace(a, b)
open(r"A:\blender_work\a4\prep_veg_chunk.py", "w", encoding="utf-8").write(v)

o = open(r"A:\blender_work\region_data\fetch_osm_region.py", encoding="utf-8").read()
a = 'D = os.path.dirname(os.path.abspath(__file__))'
assert a in o
o = o.replace(a, 'import sys\nD = sys.argv[1]')
open(r"A:\blender_work\a4\fetch_osm_chunk.py", "w", encoding="utf-8").write(o)

t = open(r"A:\blender_work\region_data\unity_textures_region.py", encoding="utf-8").read()
a = 'D = os.path.dirname(os.path.abspath(__file__))\nP = os.path.join(D, "prepared")\nOUT = sys.argv[1]'
assert a in t
t = t.replace(a, 'D = sys.argv[2]\nP = os.path.join(D, "prepared")\nOUT = sys.argv[1]')
open(r"A:\blender_work\a4\unity_textures_chunk.py", "w", encoding="utf-8").write(t)
print("ok")
