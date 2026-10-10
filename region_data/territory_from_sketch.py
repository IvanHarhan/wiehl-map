"""The user's sketch (pink = map boundary, grey = roads that must be there) on a screenshot of the Blosm extent map,
georeferenced from town labels, into UTM: prepared/territory.npz (10 m mask over the covering tiles, grey road points)
and a check image with the result drawn back on the sketch.
Usage: python territory_from_sketch.py SKETCH.webp"""
import json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
from numpy.lib.stride_tricks import sliding_window_view

D = os.path.dirname(os.path.abspath(__file__))
img = Image.open(sys.argv[1]).convert("RGB")
a = np.asarray(img).astype(int)
H, W = a.shape[:2]

# town labels on the screenshot (pixel centre of the name) -> OSM place coordinates
CONTROL = [((465, 152), (51.0277658, 7.5630545)),   # Gummersbach
           ((652, 157), (51.0207425, 7.6490729)),   # Bergneustadt
           ((87, 338), (50.9872964, 7.4084206)),    # Engelskirchen
           ((317, 372), (50.9625430, 7.5034129)),   # Bielstein
           ((741, 346), (50.9507695, 7.6911008)),   # Reichshof
           ((432, 410), (50.9478489, 7.5530059)),   # Wiehl
           ((623, 506), (50.9275734, 7.6435012))]   # Brüchermühle
px = np.array([c[0] for c in CONTROL], float)
ll = np.array([c[1] for c in CONTROL], float)
M = np.linalg.lstsq(np.c_[px, np.ones(len(px))], ll, rcond=None)[0]
res = np.c_[px, np.ones(len(px))] @ M - ll
print("control residuals (m):", np.round(np.c_[res[:, 0] * 111200, res[:, 1] * 70100], 0).tolist())


def to_ll(xy):
    return np.c_[xy, np.ones(len(xy))] @ M


r, g, b = a[..., 0], a[..., 1], a[..., 2]
pink = (r > 130) & (g < 90) & (b > 60) & (b < 190) & (r - g > 90)
grey = (np.abs(r - g) < 14) & (np.abs(g - b) < 14) & (r > 60) & (r < 120)


def dilate(m, k):
    return sliding_window_view(np.pad(m, k), (2 * k + 1, 2 * k + 1)).any(axis=(2, 3))


# grey strokes are thick; thin grey text is not: keep grey pixels whose neighbourhood is mostly grey
g_thick = sliding_window_view(np.pad(grey, 3), (7, 7)).mean(axis=(2, 3)) > 0.55
g_thick = dilate(g_thick, 1) & grey

# the labels sit only roughly on their towns (residuals up to a km): pull the grey strokes onto the real major
# roads with an iterative closest point fit (scale, rotation, shift), starting from the label fit
KX, KY, LAT_C, LON_C = 70.1, 111.2, 50.95, 7.55  # km per degree near Wiehl


def ll_to_km(lat, lon):
    return np.c_[(lon - LON_C) * KX, (lat - LAT_C) * KY]


roads = []
for e in json.load(open(os.path.join(D, "osm_major_big.json"), encoding="utf-8"))["elements"]:
    g = e.get("geometry", [])
    for p, q in zip(g[:-1], g[1:]):
        a0, a1 = ll_to_km(p["lat"], p["lon"])[0], ll_to_km(q["lat"], q["lon"])[0]
        n = max(1, int(np.linalg.norm(a1 - a0) / 0.02))
        roads.append(a0 + (a1 - a0) * (np.arange(n) / n)[:, None])
roads = np.vstack(roads)
cell = {}
for i, (x, y) in enumerate(np.floor(roads / 0.25).astype(int)):
    cell.setdefault((x, y), []).append(i)


def nearest(pts):
    out = np.full(len(pts), -1)
    dist = np.full(len(pts), np.inf)
    for k, (x, y) in enumerate(np.floor(pts / 0.25).astype(int)):
        cand = [i for dx in (-1, 0, 1) for dy in (-1, 0, 1) for i in cell.get((x + dx, y + dy), ())]
        if cand:
            d = np.linalg.norm(roads[cand] - pts[k], axis=1)
            j = int(np.argmin(d))
            out[k], dist[k] = cand[j], d[j]
    return out, dist


gy0, gx0 = np.nonzero(sliding_window_view(np.pad(grey, 3), (7, 7)).mean(axis=(2, 3)) > 0.55)
src = np.c_[gx0 + 0.5, gy0 + 0.5][::3]
ll0 = np.c_[src, np.ones(len(src))] @ M
cur = ll_to_km(ll0[:, 0], ll0[:, 1])
# similarity px -> km: [x y] = s * R(th) * [px, -py] + t, solved as a linear map with a = s cos, b = s sin
A_ = np.c_[src[:, 0], -src[:, 1], np.ones(len(src))]
sol = np.linalg.lstsq(np.c_[np.r_[A_[:, 0], A_[:, 1]], np.r_[-A_[:, 1], A_[:, 0]],
                              np.r_[np.ones(len(src)), np.zeros(len(src))], np.r_[np.zeros(len(src)), np.ones(len(src))]],
                      np.r_[cur[:, 0], cur[:, 1]], rcond=None)[0]
for it, thr in enumerate([1.0, 0.8, 0.6, 0.45, 0.35, 0.25, 0.2, 0.15, 0.15, 0.12, 0.12, 0.1]):
    a_, b_, tx, ty = sol
    km = np.c_[a_ * src[:, 0] - b_ * (-src[:, 1]) + tx, b_ * src[:, 0] + a_ * (-src[:, 1]) + ty]
    idx, dist = nearest(km)
    ok = (idx >= 0) & (dist < thr)
    s0, t0 = src[ok], roads[idx[ok]]
    X = np.c_[np.r_[s0[:, 0], -s0[:, 1]], np.r_[s0[:, 1], s0[:, 0]], np.r_[np.ones(len(s0)), np.zeros(len(s0))],
              np.r_[np.zeros(len(s0)), np.ones(len(s0))]]
    sol = np.linalg.lstsq(X, np.r_[t0[:, 0], t0[:, 1]], rcond=None)[0]
    print(f"icp {it}: {ok.sum()} pairs, median {np.median(dist[ok]) * 1000:.0f} m")
a_, b_, tx, ty = sol


def to_ll(xy):
    xy = np.asarray(xy, float)
    x = a_ * xy[:, 0] + b_ * xy[:, 1] + tx
    y = b_ * xy[:, 0] - a_ * xy[:, 1] + ty
    return np.c_[y / KY + LAT_C, x / KX + LON_C]


res = to_ll(px) - ll
print("label residuals after road fit (m):", np.round(np.c_[res[:, 0] * 111200, res[:, 1] * 70100], 0).tolist())

# inside of the pink outline: everything the outside cannot reach without crossing pink
wall = dilate(pink, 2)
outside = np.zeros_like(wall)
stack = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
while stack:
    y, x = stack.pop()
    if y < 0 or y >= H or x < 0 or x >= W or outside[y, x] or wall[y, x]:
        continue
    outside[y, x] = True
    stack.extend(((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)))
inside = ~outside
print("inside px", int(inside.sum()), "grey px", int(g_thick.sum()))

# into UTM: 10 m cells over the tiles that the territory touches
sys.path.insert(0, D)
E_A, E_F, K0, LON0 = 6378137.0, 1 / 298.257222101, 0.9996, math.radians(9)
E2 = E_F * (2 - E_F); EP2 = E2 / (1 - E2)


def utm(lat, lon):
    phi, lam = np.radians(lat), np.radians(lon)
    N = E_A / np.sqrt(1 - E2 * np.sin(phi) ** 2)
    T = np.tan(phi) ** 2; C = EP2 * np.cos(phi) ** 2; A = np.cos(phi) * (lam - LON0)
    Mm = E_A * ((1 - E2/4 - 3*E2**2/64 - 5*E2**3/256) * phi - (3*E2/8 + 3*E2**2/32 + 45*E2**3/1024) * np.sin(2*phi)
                + (15*E2**2/256 + 45*E2**3/1024) * np.sin(4*phi) - (35*E2**3/3072) * np.sin(6*phi))
    x = 500000 + K0 * N * (A + (1 - T + C) * A**3 / 6 + (5 - 18*T + T*T + 72*C - 58*EP2) * A**5 / 120)
    y = K0 * (Mm + N * np.tan(phi) * (A*A/2 + (5 - T + 9*C + 4*C*C) * A**4 / 24 + (61 - 58*T + T*T + 600*C - 330*EP2) * A**6 / 720))
    return np.c_[x, y]


iy, ix = np.nonzero(inside)
u_in = utm(*to_ll(np.c_[ix + 0.5, iy + 0.5]).T)
gy, gx = np.nonzero(g_thick)
u_grey = utm(*to_ll(np.c_[gx + 0.5, gy + 0.5]).T)
E0, E1 = int(u_in[:, 0].min() // 1000), int(u_in[:, 0].max() // 1000) + 1
N0, N1 = int(u_in[:, 1].min() // 1000), int(u_in[:, 1].max() // 1000) + 1
RES = 10
mask = np.zeros(((N1 - N0) * 100, (E1 - E0) * 100), bool)
# one screenshot pixel is ~40 m: stamp each inside pixel as a 5x5-cell square so the mask has no holes
c = ((u_in[:, 0] - E0 * 1000) / RES).astype(int)
rr = ((N1 * 1000 - u_in[:, 1]) / RES).astype(int)
for dy in range(-3, 4):
    for dx in range(-3, 4):
        cc, ry = np.clip(c + dx, 0, mask.shape[1] - 1), np.clip(rr + dy, 0, mask.shape[0] - 1)
        mask[ry, cc] = True
tiles = sorted({(e, n) for e in range(E0, E1) for n in range(N0, N1)
                if mask[(N1 - n - 1) * 100:(N1 - n) * 100, (e - E0) * 100:(e - E0 + 1) * 100].any()})
print("utm tiles E", E0, "..", E1 - 1, "N", N0, "..", N1 - 1, "->", len(tiles), "tiles touched, area km²",
      round(mask.sum() * RES * RES / 1e6, 1))
os.makedirs(os.path.join(D, "prepared"), exist_ok=True)
np.savez_compressed(os.path.join(D, "territory.npz"), mask=mask, E0=E0, N1=N1, res=RES, grey=u_grey.astype(np.float32),
                    tiles=np.array(tiles))

# check: the extracted territory and roads drawn back on the sketch, plus a few known places
chk = img.copy()
over = Image.new("RGBA", img.size, (0, 0, 0, 0))
od = ImageDraw.Draw(over)
ys, xs = np.nonzero(inside)
for y, x in zip(ys[::7], xs[::7]):
    od.point((int(x), int(y)), fill=(0, 255, 0, 160))
for y, x in zip(gy, gx):
    od.point((int(x), int(y)), fill=(0, 120, 255, 255))
for name, lat, lon in (("bus", 50.94698, 7.54955), ("home", 50.95806, 7.54296), ("koeln29a", 50.9265, 7.6371),
                       ("gummersbach", 51.0277658, 7.5630545), ("bielstein", 50.962543, 7.5034129)):
    kx, ky = ll_to_km(lat, lon)[0]
    x, y = np.linalg.solve([[a_, b_], [b_, -a_]], [kx - tx, ky - ty])
    od.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 255, 0, 255), width=2)
chk = Image.alpha_composite(chk.convert("RGBA"), over)
chk.save(os.path.join(D, "territory_check.png"))
json.dump([list(t) for t in tiles], open(os.path.join(D, "territory_tiles.json"), "w"))
