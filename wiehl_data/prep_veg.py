"""Find trees and bushes: height above ground (DOM1 - DGM1) where the orthophoto is vegetation (NDVI)."""
import glob, json, math, os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
D = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(D, "prepared")
meta = json.load(open(os.path.join(P, "meta.json")))
UX0, UY0, UX1, UY1 = meta["utm_bbox"]
AFF = np.array(meta["affine"])
INV = np.linalg.inv(AFF[:2])
Wd, Hd = UX1 - UX0, UY1 - UY0
dem = np.load(os.path.join(P, "dem.npy"))


def mosaic(pattern, load):
    """1 m grid over the bbox, row 0 = north; load(path) returns a 1000x1000 array for the tile."""
    out = np.full((Hd, Wd), np.nan, np.float32)
    for f in glob.glob(os.path.join(D, pattern)):
        tx, ty = (int(v) * 1000 for v in os.path.basename(f).split("_")[2:4])
        a = load(f)
        n = a.shape[0]
        x0, x1 = max(UX0, tx), min(UX1, tx + n)
        y0, y1 = max(UY0, ty), min(UY1, ty + n)
        if x0 < x1 and y0 < y1:
            out[UY1 - y1:UY1 - y0, x0 - UX0:x1 - UX0] = a[ty + n - y1:ty + n - y0, x0 - tx:x1 - tx]
    return out


def load_ndvi(f):
    im = Image.open(f)
    im.reduce = 2
    im.load()
    r, _, _, nir = (np.asarray(b.resize((1000, 1000), Image.BILINEAR), np.float32) for b in im.split()[:4])
    return (nir - r) / np.maximum(nir + r, 1)


dom = mosaic("dom1_32_*.tif", lambda f: np.array(Image.open(f), np.float32))
ndvi = mosaic("dop10rgbi_32_*.jp2", load_ndvi)
ndom = np.nan_to_num(dom - dem, nan=0.0)
ndvi = np.nan_to_num(ndvi, nan=0.0)
# vegetation mask for the ground material (grass detail texture), same grid/orientation as ortho.jpg
Image.fromarray((np.clip((ndvi - 0.06) / 0.14, 0, 1) * 255).astype(np.uint8)).save(os.path.join(P, "vegmask.png"))

# building footprints (LoD2 ground faces), grown by 1 m
V = np.load(os.path.join(P, "bld_verts.npy"))
FL = np.load(os.path.join(P, "bld_face_len.npy"))
FT = np.load(os.path.join(P, "bld_face_type.npy"))
starts = np.r_[0, np.cumsum(FL)[:-1]]
mask_img = Image.new("L", (Wd, Hd), 0)
draw = ImageDraw.Draw(mask_img)
for s, n, t in zip(starts, FL, FT):
    if t != 2:
        continue
    d = (V[s:s + n, :2] - AFF[2]) @ INV
    draw.polygon([(x, Hd - y) for x, y in d], fill=255, outline=255)
bld = np.asarray(mask_img) > 0
bld = sliding_window_view(np.pad(bld, 1), (3, 3)).max(axis=(2, 3))

veg = (ndvi > 0.12) & ~bld
blur = sliding_window_view(np.pad(ndom, 1, mode="edge"), (3, 3)).mean(axis=(2, 3))
peak = blur >= sliding_window_view(np.pad(blur, 2, mode="edge"), (5, 5)).max(axis=(2, 3))


def cell_to_local(rows, cols, jitter=0.0, rng=None):
    dx = cols + 0.5; dy = Hd - rows - 0.5
    if jitter:
        dx = dx + rng.uniform(-jitter, jitter, len(dx)); dy = dy + rng.uniform(-jitter, jitter, len(dy))
    return np.c_[dx, dy, np.ones(len(dx))] @ AFF


rng = np.random.default_rng(7)
tr, tc = np.nonzero(peak & veg & (blur > 3.0))
th = ndom[tr, tc]
trees = np.c_[cell_to_local(tr, tc, 0.3, rng), dem[tr, tc], th, np.clip(0.4 * th, 2.0, 8.0)]

br, bc = np.nonzero(veg & (ndom > 0.4) & (ndom <= 3.0) & ((np.arange(Hd)[:, None] + np.arange(Wd)[None, :]) % 2 == 0))
bushes = np.c_[cell_to_local(br, bc, 0.6, rng), dem[br, bc], ndom[br, bc]]

np.save(os.path.join(P, "trees.npy"), trees.astype(np.float32))
np.save(os.path.join(P, "bushes.npy"), bushes.astype(np.float32))
print("VEG_DONE trees", len(trees), "bushes", len(bushes), "tree_h_median", float(np.median(th)) if len(th) else 0,
      "veg_share", float(veg.mean()))
