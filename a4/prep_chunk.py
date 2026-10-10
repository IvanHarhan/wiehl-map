"""Prepare the NRW tiles in region_data/tiles for Blender, in the same local frame as the Wiehl map
(Blosm transverse mercator around карта.blend's origin), so the old town and everything placed by hand stays put.
Writes region_data/prepared: dem.npy (DGM1, 1 m, row 0 = north, whole region), ortho/<E>_<N>.jpg per 1 km tile
(finer in the town tiles), LoD2 buildings as bld_*.npy like wiehl_data/prepared, meta.json."""
import glob, json, math, os, re, sys
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

sys.path.insert(0, r"A:\blender_work\wiehl_data")
Image.MAX_IMAGE_PIXELS = None
D = sys.argv[1]
T = r"A:\blender_work\a4\tiles"
OUT = os.path.join(D, "prepared")
os.makedirs(os.path.join(OUT, "ortho"), exist_ok=True)

LAT0, LON0 = 50.9471549987793, 7.548830032348633  # карта.blend origin: the whole map shares this frame
# orthophoto metres per pixel: the old town keeps its 0.2 m, the other towns 0.4 m (see TOWNS), the country 0.8 m
CORE = set()
DETAIL = set()
# lat, lon, radius (m) of the towns the user wants in detail
TOWNS = []
_OLD_TOWNS = [(50.9875, 7.5755, 900),   # Alferzhagen / Marienhagen
         (50.9855, 7.5400, 1300),  # Dieringhausen: the station, its bus stops, Action and Kaufland
         (51.0278, 7.5631, 1100),  # Gummersbach centre
         (50.9625, 7.5034, 500)]   # Bielstein: the basketball court and the playground
DEM_RES = 2  # m: 1 m over the whole territory (15 x 17 km) would be a gigabyte

E_A, E_F, K0, UTM_LON0 = 6378137.0, 1 / 298.257222101, 0.9996, math.radians(9)
E2 = E_F * (2 - E_F); EP2 = E2 / (1 - E2)


def utm_fwd(lat, lon):
    phi, lam = math.radians(lat), math.radians(lon)
    N = E_A / math.sqrt(1 - E2 * math.sin(phi) ** 2)
    T_ = math.tan(phi) ** 2; C = EP2 * math.cos(phi) ** 2; A_ = math.cos(phi) * (lam - UTM_LON0)
    M = E_A * ((1 - E2/4 - 3*E2**2/64 - 5*E2**3/256) * phi - (3*E2/8 + 3*E2**2/32 + 45*E2**3/1024) * math.sin(2*phi)
               + (15*E2**2/256 + 45*E2**3/1024) * math.sin(4*phi) - (35*E2**3/3072) * math.sin(6*phi))
    x = 500000 + K0 * N * (A_ + (1 - T_ + C) * A_**3 / 6 + (5 - 18*T_ + T_*T_ + 72*C - 58*EP2) * A_**5 / 120)
    y = K0 * (M + N * math.tan(phi) * (A_*A_/2 + (5 - T_ + 9*C + 4*C*C) * A_**4 / 24 + (61 - 58*T_ + T_*T_ + 600*C - 330*EP2) * A_**6 / 720))
    return x, y


def utm_inv(x, y):
    e1 = (1 - np.sqrt(1 - E2)) / (1 + np.sqrt(1 - E2))
    mu = y / K0 / (E_A * (1 - E2/4 - 3*E2**2/64 - 5*E2**3/256))
    p1 = mu + (3*e1/2 - 27*e1**3/32) * np.sin(2*mu) + (21*e1**2/16 - 55*e1**4/32) * np.sin(4*mu) + (151*e1**3/96) * np.sin(6*mu) + (1097*e1**4/512) * np.sin(8*mu)
    N1 = E_A / np.sqrt(1 - E2 * np.sin(p1)**2); T1 = np.tan(p1)**2; C1 = EP2 * np.cos(p1)**2
    R1 = E_A * (1 - E2) / (1 - E2 * np.sin(p1)**2) ** 1.5; Dd = (x - 500000) / (N1 * K0)
    lat = p1 - (N1 * np.tan(p1) / R1) * (Dd*Dd/2 - (5 + 3*T1 + 10*C1 - 4*C1*C1 - 9*EP2) * Dd**4/24 + (61 + 90*T1 + 298*C1 + 45*T1*T1 - 252*EP2 - 3*C1*C1) * Dd**6/720)
    lon = UTM_LON0 + (Dd - (1 + 2*T1 + C1) * Dd**3/6 + (5 - 2*C1 + 28*T1 - 3*C1*C1 + 8*EP2 + 24*T1*T1) * Dd**5/120) / np.cos(p1)
    return np.degrees(lat), np.degrees(lon)


def blosm_tm(lat, lon):
    R = 6378137.0
    la, lo = np.radians(lat), np.radians(lon - LON0)
    B = np.sin(lo) * np.cos(la)
    return np.c_[0.5 * R * np.log((1 + B) / (1 - B)), R * (np.arctan(np.tan(la) / np.cos(lo)) - math.radians(LAT0))]


def tile_of(name):
    m = re.search(r"_32_(\d{3})_(\d{4})_", name)
    return int(m.group(1)), int(m.group(2))


territory = np.load(os.path.join(D, "territory.npz"))
t_tiles = {(int(e), int(n)) for e, n in territory["tiles"]}
tiles = sorted({tile_of(f) for f in glob.glob(os.path.join(T, "dgm1", "*.tif"))} & t_tiles)
print("tiles", len(tiles), "of", len(t_tiles), "in the territory", flush=True)
es, ns = [t[0] for t in tiles], [t[1] for t in tiles]
UX0, UX1, UY0, UY1 = min(es) * 1000, (max(es) + 1) * 1000, min(ns) * 1000, (max(ns) + 1) * 1000

# UTM -> local: quadratic fit on a grid (an affine one is off by decimetres over 10 km)
gx, gy = np.meshgrid(np.linspace(UX0, UX1, 21), np.linspace(UY0, UY1, 21))
gx, gy = gx.ravel(), gy.ravel()
dst = blosm_tm(*utm_inv(gx, gy))


def basis(dx, dy):
    return np.c_[dx, dy, np.ones_like(dx), dx * dx, dx * dy, dy * dy]


A = np.linalg.lstsq(basis(gx - UX0, gy - UY0), dst, rcond=None)[0]
fit_err = float(np.abs(basis(gx - UX0, gy - UY0) @ A - dst).max())
aff = np.linalg.lstsq(basis(gx - UX0, gy - UY0)[:, :3], dst, rcond=None)[0]
aff_err = float(np.abs(basis(gx - UX0, gy - UY0)[:, :3] @ aff - dst).max())
print(f"region E {UX0}..{UX1} N {UY0}..{UY1}, quadratic fit {fit_err * 1000:.2f} mm, affine {aff_err * 1000:.1f} mm", flush=True)


def to_local(dxy):
    return basis(dxy[:, 0], dxy[:, 1]) @ A


# town tiles: every tile a town circle reaches
for lat, lon, rad in TOWNS:
    cx, cy = utm_fwd(lat, lon)
    DETAIL |= {(e, n) for e, n in tiles
               if np.hypot(max(e * 1000 - cx, 0, cx - e * 1000 - 1000), max(n * 1000 - cy, 0, cy - n * 1000 - 1000)) < rad}
DETAIL |= {(int(e), int(n)) for e, n in np.load(r"A:\blender_work\a4\corridor.npz")["detail_tiles"]}
print("detail tiles", len(DETAIL & set(tiles)), flush=True)

# ---------- DGM1 at DEM_RES, row 0 = north ----------
k2 = 1000 // DEM_RES
dem = np.full(((UY1 - UY0) // DEM_RES, (UX1 - UX0) // DEM_RES), np.nan, np.float32)
for e, n in tiles:
    f = glob.glob(os.path.join(T, "dgm1", f"*_32_{e}_{n}_*"))[0]
    a = np.array(Image.open(f), np.float32)[:1000, :1000]
    a = a.reshape(k2, DEM_RES, k2, DEM_RES).mean(axis=(1, 3))
    r0, c0 = (UY1 - n * 1000 - 1000) // DEM_RES, (e * 1000 - UX0) // DEM_RES
    dem[r0:r0 + k2, c0:c0 + k2] = a
dem[dem < -1000] = np.nan
missing = int(np.isnan(dem).sum())
if missing:
    dem[np.isnan(dem)] = np.nanmedian(dem)
np.save(os.path.join(OUT, "dem.npy"), dem)
print("DEM", dem.shape, "missing", missing, "range", float(dem.min()), float(dem.max()), flush=True)

# ---------- DOP10 -> one JPEG per 1 km tile ----------
ortho_res = {}
for f in sorted(glob.glob(os.path.join(T, "dop10", "*.jp2"))):
    e, n = tile_of(f)
    if (e, n) not in tiles:
        continue
    mpp = 0.2 if (e, n) in CORE else 0.4 if (e, n) in DETAIL else 0.8
    ortho_res[f"{e}_{n}"] = mpp
    out = os.path.join(OUT, "ortho", f"{e}_{n}.jpg")
    if os.path.exists(out):
        continue
    im = Image.open(f)
    im.reduce = int(math.floor(math.log2(mpp / 0.1)))
    im.load()
    bands = im.split()
    rgb = Image.merge("RGB", bands[:3]) if len(bands) >= 3 else im.convert("RGB")
    side = int(round(1000 / mpp))
    if rgb.size != (side, side):
        rgb = rgb.resize((side, side), Image.LANCZOS)
    rgb.save(out, quality=90)
    print("ortho", e, n, mpp, flush=True)

# ---------- LoD2 ----------
GML = "{http://www.opengis.net/gml}"
BLDG = "{http://www.opengis.net/citygml/building/1.0}"
SURF = {"RoofSurface": 0, "WallSurface": 1, "GroundSurface": 2, "ClosureSurface": 1}
verts, face_len, face_type, face_bld, functions = [], [], [], [], []
n_bld = 0
tmask, tE0, tN1, tres = territory["mask"], int(territory["E0"]), int(territory["N1"]), int(territory["res"])
dmask = territory["detail"]  # buildings only within 600 m of the A4


def in_territory(x, y, grow=5):
    """UTM point inside the user's outline (grown by `grow` cells of 10 m)"""
    r, c = int((tN1 * 1000 - y) / tres), int((x - tE0 * 1000) / tres)
    return dmask[max(0, r - grow):r + grow + 1, max(0, c - grow):c + grow + 1].any()


for e, n in tiles:
    f = glob.glob(os.path.join(T, "lod2", f"LoD2_32_{e}_{n}_*"))
    if not f:
        continue
    f = f[0]
    for _, b in ET.iterparse(f, events=("end",)):
        if b.tag != BLDG + "Building":
            continue
        polys = []
        for tag, t in SURF.items():
            for s in b.iter(BLDG + tag):
                for pos in s.iter(GML + "posList"):
                    p = np.array(pos.text.split(), float).reshape(-1, 3)
                    if len(p) > 1 and np.allclose(p[0], p[-1]):
                        p = p[:-1]
                    if len(p) >= 3:
                        polys.append((t, p))
        fn = next(b.iter(BLDG + "function"), None)
        fn = fn.text.strip() if fn is not None and fn.text else ""
        b.clear()
        if not polys:
            continue
        c = np.vstack([p for _, p in polys]).mean(0)
        if not (UX0 <= c[0] <= UX1 and UY0 <= c[1] <= UY1) or not in_territory(c[0], c[1]):
            continue
        for t, p in polys:
            verts.append(np.c_[to_local(np.c_[p[:, 0] - UX0, p[:, 1] - UY0]), p[:, 2]])
            face_len.append(len(p)); face_type.append(t); face_bld.append(n_bld)
        functions.append(fn)
        n_bld += 1
    print("lod2", os.path.basename(f), n_bld, flush=True)

np.save(os.path.join(OUT, "bld_verts.npy"), np.vstack(verts))
np.save(os.path.join(OUT, "bld_face_len.npy"), np.array(face_len, np.int32))
np.save(os.path.join(OUT, "bld_face_type.npy"), np.array(face_type, np.int8))
np.save(os.path.join(OUT, "bld_face_bld.npy"), np.array(face_bld, np.int32))
json.dump(functions, open(os.path.join(OUT, "bld_function.json"), "w"))

np.savez_compressed(os.path.join(OUT, "territory.npz"), mask=tmask, detail=dmask, E0=tE0, N1=tN1, res=tres)
meta = {"utm_bbox": [UX0, UY0, UX1, UY1], "dem_res": DEM_RES, "local_poly": A.tolist(), "affine": aff.tolist(), "fit_err_m": fit_err,
        "affine_err_m": aff_err, "lat0": LAT0, "lon0": LON0, "dem_shape": list(dem.shape), "dem_missing": missing,
        "ortho_m_per_px": ortho_res, "core": sorted(f"{e}_{n}" for e, n in CORE),
        "detail": sorted(f"{e}_{n}" for e, n in DETAIL), "buildings": n_bld, "faces": len(face_len),
        "chunk": {"height_offset": 46.290374755859375, "origin": [50.747467041015625, -19.01202964782715, 233.75]}}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("PREP_DONE buildings", n_bld, "faces", len(face_len))
