"""Trees and bushes for the region, tile by tile: height above ground (DOM1 - DGM1) where the orthophoto is
vegetation (NDVI), like wiehl_data/prep_veg.py. Near buildings every tree is kept; out in the woods only the tallest
one per 15 m cell, with its crown widened to close the gaps, so the forests stay affordable.
Writes prepared/trees.npy (x, y, ground z, height, crown radius), bushes.npy (x, y, ground z, height) and
prepared/vegmask/<E>_<N>.png (1 m, for the grass detail texture)."""
import glob, json, os, re
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
import sys
D = sys.argv[1]
T, P = r"A:\blender_work\a4\tiles", os.path.join(D, "prepared")
os.makedirs(os.path.join(P, "vegmask"), exist_ok=True)
meta = json.load(open(os.path.join(P, "meta.json")))
UX0, UY0, UX1, UY1 = meta["utm_bbox"]
A = np.array(meta["local_poly"])
DETAIL = set()
TILES = {tuple(map(int, k.split("_"))) for k in meta["ortho_m_per_px"]}
terr = np.load(os.path.join(P, "territory.npz"))
tmask, tE0, tN1, tres = terr["mask"], int(terr["E0"]), int(terr["N1"]), int(terr["res"])
NEAR_M, FAR_CELL = 120, 15


def basis(dx, dy):
    return np.c_[dx, dy, np.ones_like(dx), dx * dx, dx * dy, dy * dy]


def tile_of(name):
    m = re.search(r"_32_(\d{3})_(\d{4})_", name)
    return int(m.group(1)), int(m.group(2))


def find(kind, e, n):
    fs = glob.glob(os.path.join(T, kind, f"*_32_{e}_{n}_*"))
    return fs[0] if fs else None


# LoD2 footprints in UTM metres relative to the region corner: invert the local fit by Newton steps on the affine part
V = np.load(os.path.join(P, "bld_verts.npy"))
FL = np.load(os.path.join(P, "bld_face_len.npy"))
FT = np.load(os.path.join(P, "bld_face_type.npy"))
starts = np.r_[0, np.cumsum(FL)[:-1]]
ground_faces = [(s, n) for s, n, t in zip(starts, FL, FT) if t == 2]
gidx = np.concatenate([np.arange(s, s + n) for s, n in ground_faces])
loc = V[gidx, :2]
J = A[:2].T  # d local / d (dx, dy) at the origin, good enough as a Newton matrix
uv = np.linalg.solve(J, (loc - A[2]).T).T
for _ in range(4):
    uv += np.linalg.solve(J, (loc - basis(uv[:, 0], uv[:, 1]) @ A).T).T
foot_utm, k = [], 0
for s, n in ground_faces:
    foot_utm.append(uv[k:k + n]); k += n
foot_c = np.array([f.mean(0) for f in foot_utm])

rng = np.random.default_rng(7)
trees, bushes = [], []
for f in sorted(glob.glob(os.path.join(T, "dom1", "*.tif"))):
    e, n = tile_of(f)
    if (e, n) not in TILES:
        continue
    ox, oy = e * 1000 - UX0, n * 1000 - UY0            # tile's SW corner, metres from the region corner
    dom = np.array(Image.open(f), np.float32)[:1000, :1000]
    g = np.array(Image.open(find("dgm1", e, n)), np.float32)[:1000, :1000]
    # the user's outline over this tile, 1 m cells (the mask is 10 m)
    rr0, cc0 = (tN1 * 1000 - (n * 1000 + 1000)) // tres, (e * 1000 - tE0 * 1000) // tres
    inside = np.repeat(np.repeat(tmask[rr0:rr0 + 100, cc0:cc0 + 100], tres, 0), tres, 1)
    ndom = np.nan_to_num(dom - g, nan=0.0)
    im = Image.open(find("dop10", e, n))
    im.reduce = 2
    im.load()
    r, _, _, nir = (np.asarray(b.resize((1000, 1000), Image.BILINEAR), np.float32) for b in im.split()[:4])
    ndvi = np.nan_to_num((nir - r) / np.maximum(nir + r, 1), nan=0.0)
    Image.fromarray((np.clip((ndvi - 0.06) / 0.14, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(P, "vegmask", f"{e}_{n}.png"))

    # footprints (+1 m) and the "near buildings" zone, on a canvas padded by NEAR_M
    pad = NEAR_M
    canvas = Image.new("L", (1000 + 2 * pad, 1000 + 2 * pad), 0)
    draw = ImageDraw.Draw(canvas)
    cx, cy = ox + 500, oy + 500
    for i in np.nonzero((np.abs(foot_c[:, 0] - cx) < 500 + pad + 50) & (np.abs(foot_c[:, 1] - cy) < 500 + pad + 50))[0]:
        p = foot_utm[i]
        draw.polygon([(x - ox + pad, 1000 + pad - (y - oy)) for x, y in p], fill=255, outline=255)
    big = np.asarray(canvas) > 0
    bld = sliding_window_view(np.pad(big, 1), (3, 3)).max(axis=(2, 3))[pad:pad + 1000, pad:pad + 1000]
    if f"{e}_{n}" in DETAIL:
        near = np.ones((1000, 1000), bool)
    else:
        c10 = big.reshape((1000 + 2 * pad) // 10, 10, -1, 10).any(axis=(1, 3))
        w = NEAR_M // 10
        c10 = sliding_window_view(np.pad(c10, w), (2 * w + 1, 2 * w + 1)).max(axis=(2, 3))
        near = np.repeat(np.repeat(c10, 10, 0), 10, 1)[pad:pad + 1000, pad:pad + 1000]

    veg = (ndvi > 0.12) & ~bld & inside
    blur = sliding_window_view(np.pad(ndom, 1, mode="edge"), (3, 3)).mean(axis=(2, 3))
    peak = blur >= sliding_window_view(np.pad(blur, 2, mode="edge"), (5, 5)).max(axis=(2, 3))
    cand = peak & veg & (blur > 3.0)
    tr, tc = np.nonzero(cand & near)
    th, crown = ndom[tr, tc], np.clip(0.4 * ndom[tr, tc], 2.0, 8.0)
    # far away: the tallest candidate per FAR_CELL cell, crown grown to fill the cell
    fr, fc = np.nonzero(cand & ~near)
    if len(fr):
        cell = (fr // FAR_CELL) * 1000 + fc // FAR_CELL
        order = np.lexsort((-ndom[fr, fc], cell))
        first = np.r_[True, cell[order][1:] != cell[order][:-1]]
        keep = order[first]
        fr, fc = fr[keep], fc[keep]
        tr, tc = np.r_[tr, fr], np.r_[tc, fc]
        th = np.r_[th, ndom[fr, fc]]
        crown = np.r_[crown, np.clip(0.5 * ndom[fr, fc], 4.0, 9.0)]
    dx = ox + tc + 0.5 + rng.uniform(-0.3, 0.3, len(tc))
    dy = oy + 1000 - tr - 0.5 + rng.uniform(-0.3, 0.3, len(tr))
    trees.append(np.c_[basis(dx, dy) @ A, g[tr, tc], th, crown])

    br, bc = np.nonzero(veg & near & (ndom > 0.4) & (ndom <= 3.0) & ((np.arange(1000)[:, None] + np.arange(1000)[None, :]) % 2 == 0))
    dx = ox + bc + 0.5 + rng.uniform(-0.6, 0.6, len(bc))
    dy = oy + 1000 - br - 0.5 + rng.uniform(-0.6, 0.6, len(br))
    bushes.append(np.c_[basis(dx, dy) @ A, g[br, bc], ndom[br, bc]])
    print("veg", e, n, "trees", len(tr), "bushes", len(br), flush=True)

trees, bushes = np.vstack(trees).astype(np.float32), np.vstack(bushes).astype(np.float32)
np.save(os.path.join(P, "trees.npy"), trees)
np.save(os.path.join(P, "bushes.npy"), bushes)
print("VEG_DONE trees", len(trees), "bushes", len(bushes))
