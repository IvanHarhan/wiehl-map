"""Crop NRW open data (LoD2, DGM1, DOP10) to the Blosm extent and convert to Blosm local coordinates."""
import glob, json, math, os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "prepared")
os.makedirs(OUT, exist_ok=True)

# from карта.blend (scene["lat"], scene["lon"], scene.blosm extent)
LAT0, LON0 = 50.9471549987793, 7.548830032348633
MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 50.94504165649414, 50.951908111572266, 7.541989803314209, 7.559110164642334
ORTHO_M_PER_PX = 0.2

E_A, E_F, K0, UTM_LON0 = 6378137.0, 1 / 298.257222101, 0.9996, math.radians(9)
E2 = E_F * (2 - E_F); EP2 = E2 / (1 - E2)


def utm_fwd(lat, lon):
    phi, lam = math.radians(lat), math.radians(lon)
    N = E_A / math.sqrt(1 - E2 * math.sin(phi) ** 2)
    T = math.tan(phi) ** 2; C = EP2 * math.cos(phi) ** 2; A = math.cos(phi) * (lam - UTM_LON0)
    M = E_A * ((1 - E2/4 - 3*E2**2/64 - 5*E2**3/256) * phi - (3*E2/8 + 3*E2**2/32 + 45*E2**3/1024) * math.sin(2*phi)
               + (15*E2**2/256 + 45*E2**3/1024) * math.sin(4*phi) - (35*E2**3/3072) * math.sin(6*phi))
    x = 500000 + K0 * N * (A + (1 - T + C) * A**3 / 6 + (5 - 18*T + T*T + 72*C - 58*EP2) * A**5 / 120)
    y = K0 * (M + N * math.tan(phi) * (A*A/2 + (5 - T + 9*C + 4*C*C) * A**4 / 24 + (61 - 58*T + T*T + 600*C - 330*EP2) * A**6 / 720))
    return x, y


def utm_inv(x, y):
    e1 = (1 - math.sqrt(1 - E2)) / (1 + math.sqrt(1 - E2))
    mu = y / K0 / (E_A * (1 - E2/4 - 3*E2**2/64 - 5*E2**3/256))
    p1 = mu + (3*e1/2 - 27*e1**3/32) * math.sin(2*mu) + (21*e1**2/16 - 55*e1**4/32) * math.sin(4*mu) + (151*e1**3/96) * math.sin(6*mu) + (1097*e1**4/512) * math.sin(8*mu)
    N1 = E_A / math.sqrt(1 - E2 * math.sin(p1)**2); T1 = math.tan(p1)**2; C1 = EP2 * math.cos(p1)**2
    R1 = E_A * (1 - E2) / (1 - E2 * math.sin(p1)**2) ** 1.5; Dd = (x - 500000) / (N1 * K0)
    lat = p1 - (N1 * math.tan(p1) / R1) * (Dd*Dd/2 - (5 + 3*T1 + 10*C1 - 4*C1*C1 - 9*EP2) * Dd**4/24 + (61 + 90*T1 + 298*C1 + 45*T1*T1 - 252*EP2 - 3*C1*C1) * Dd**6/720)
    lon = UTM_LON0 + (Dd - (1 + 2*T1 + C1) * Dd**3/6 + (5 - 2*C1 + 28*T1 - 3*C1*C1 + 8*EP2 + 24*T1*T1) * Dd**5/120) / math.cos(p1)
    return math.degrees(lat), math.degrees(lon)


def blosm_tm(lat, lon):
    # same formula as blosm/util/transverse_mercator.py
    R = 6378137.0
    la, lo = math.radians(lat), math.radians(lon - LON0)
    B = math.sin(lo) * math.cos(la)
    return 0.5 * R * math.log((1 + B) / (1 - B)), R * (math.atan(math.tan(la) / math.cos(lo)) - math.radians(LAT0))


corners = [utm_fwd(la, lo) for la in (MIN_LAT, MAX_LAT) for lo in (MIN_LON, MAX_LON)]
UX0, UX1 = math.floor(min(c[0] for c in corners)), math.ceil(max(c[0] for c in corners))
UY0, UY1 = math.floor(min(c[1] for c in corners)), math.ceil(max(c[1] for c in corners))

# affine UTM -> Blosm local, fitted on a grid
src, dst = [], []
for x in np.linspace(UX0, UX1, 6):
    for y in np.linspace(UY0, UY1, 6):
        src.append((x - UX0, y - UY0, 1)); dst.append(blosm_tm(*utm_inv(x, y)))
AFF = np.linalg.lstsq(np.array(src), np.array(dst), rcond=None)[0]
fit_err = float(np.abs(np.array(src) @ AFF - np.array(dst)).max())


def to_local(xy):
    return np.c_[xy[:, 0] - UX0, xy[:, 1] - UY0, np.ones(len(xy))] @ AFF


# ---------- DGM1: 1 m grid, row 0 = north ----------
dem = np.full((UY1 - UY0, UX1 - UX0), np.nan, np.float32)
for f in glob.glob(os.path.join(D, "dgm1_32_*.tif")):
    tx, ty = (int(v) * 1000 for v in os.path.basename(f).split("_")[2:4])
    a = np.array(Image.open(f), np.float32)
    n = a.shape[0]
    x0, x1 = max(UX0, tx), min(UX1, tx + n)
    y0, y1 = max(UY0, ty), min(UY1, ty + n)
    if x0 >= x1 or y0 >= y1:
        continue
    dem[UY1 - y1:UY1 - y0, x0 - UX0:x1 - UX0] = a[ty + n - y1:ty + n - y0, x0 - tx:x1 - tx]
dem[dem < -1000] = np.nan
missing = int(np.isnan(dem).sum())
if missing:
    dem[np.isnan(dem)] = np.nanmedian(dem)
np.save(os.path.join(OUT, "dem.npy"), dem)

# ---------- DOP10 -> ortho at ORTHO_M_PER_PX, row 0 = north ----------
W = int(round((UX1 - UX0) / ORTHO_M_PER_PX)); H = int(round((UY1 - UY0) / ORTHO_M_PER_PX))
ortho = Image.new("RGB", (W, H))
for f in glob.glob(os.path.join(D, "dop10rgbi_32_*.jp2")):
    tx, ty = (int(v) * 1000 for v in os.path.basename(f).split("_")[2:4])
    im = Image.open(f)
    im.reduce = int(round(math.log2(ORTHO_M_PER_PX / 0.1)))
    im.load()
    bands = im.split()
    rgb = Image.merge("RGB", bands[:3]) if len(bands) >= 3 else im.convert("RGB")
    side = int(round(1000 / ORTHO_M_PER_PX))
    if rgb.size != (side, side):
        rgb = rgb.resize((side, side), Image.LANCZOS)
    ortho.paste(rgb, (int(round((tx - UX0) / ORTHO_M_PER_PX)), int(round((UY1 - ty - 1000) / ORTHO_M_PER_PX))))
ortho.save(os.path.join(OUT, "ortho.jpg"), quality=92)

# ---------- LoD2 ----------
GML = "{http://www.opengis.net/gml}"
BLDG = "{http://www.opengis.net/citygml/building/1.0}"
SURF = {"RoofSurface": 0, "WallSurface": 1, "GroundSurface": 2, "ClosureSurface": 1}
verts, face_len, face_type, face_bld = [], [], [], []
n_bld = 0
functions = []
for f in glob.glob(os.path.join(D, "LoD2_32_*.gml")):
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
        fn = b.find(BLDG + "function")
        if fn is None:
            fn = next(b.iter(BLDG + "function"), None)
        fn = fn.text.strip() if fn is not None and fn.text else ""
        b.clear()
        if not polys:
            continue
        c = np.vstack([p for _, p in polys]).mean(0)
        if not (UX0 <= c[0] <= UX1 and UY0 <= c[1] <= UY1):
            continue
        for t, p in polys:
            verts.append(np.c_[to_local(p[:, :2]), p[:, 2]])
            face_len.append(len(p)); face_type.append(t); face_bld.append(n_bld)
        functions.append(fn)
        n_bld += 1

np.save(os.path.join(OUT, "bld_verts.npy"), np.vstack(verts))
np.save(os.path.join(OUT, "bld_face_len.npy"), np.array(face_len, np.int32))
np.save(os.path.join(OUT, "bld_face_type.npy"), np.array(face_type, np.int8))
np.save(os.path.join(OUT, "bld_face_bld.npy"), np.array(face_bld, np.int32))
json.dump(functions, open(os.path.join(OUT, "bld_function.json"), "w"))
from collections import Counter
print("FUNCTIONS", Counter(functions).most_common())

meta = {"utm_bbox": [UX0, UY0, UX1, UY1], "affine": AFF.tolist(), "fit_err_m": fit_err,
        "dem_shape": list(dem.shape), "dem_missing": missing, "ortho_size": [W, H],
        "buildings": n_bld, "faces": len(face_len)}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("PREP_DONE", json.dumps(meta))
