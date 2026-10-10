"""Corridor along the A4 from Köln's edge (LON_W) to the existing territory, minus the old territory.
Writes a4/corridor.npz: mask (10 m cells, relief band OUTER_M), detail (DETAIL_M: buildings, trees, props),
E0, N1, res, tiles; and one chunk_<k>/territory.npz per ~CHUNK tiles (west to east) in the format prep_region reads."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, r"A:\blender_work\region_data")
D = r"A:\blender_work\a4"
LON_W, LON_E = 7.07, 7.56
DETAIL_M, OUTER_M, RES, CHUNK = 600.0, 1200.0, 10, 40
E_A, E_F, K0, LON0 = 6378137.0, 1 / 298.257222101, 0.9996, math.radians(9)
E2 = E_F * (2 - E_F); EP2 = E2 / (1 - E2)


def utm(lat, lon):
    phi, lam = np.radians(lat), np.radians(lon)
    N = E_A / np.sqrt(1 - E2 * np.sin(phi) ** 2)
    T = np.tan(phi) ** 2; C = EP2 * np.cos(phi) ** 2; A = np.cos(phi) * (lam - LON0)
    M = E_A * ((1 - E2/4 - 3*E2**2/64 - 5*E2**3/256) * phi - (3*E2/8 + 3*E2**2/32 + 45*E2**3/1024) * np.sin(2*phi)
               + (15*E2**2/256 + 45*E2**3/1024) * np.sin(4*phi) - (35*E2**3/3072) * np.sin(6*phi))
    x = 500000 + K0 * N * (A + (1 - T + C) * A**3 / 6 + (5 - 18*T + T*T + 72*C - 58*EP2) * A**5 / 120)
    y = K0 * (M + N * np.tan(phi) * (A*A/2 + (5 - T + 9*C + 4*C*C) * A**4 / 24 + (61 - 58*T + T*T + 600*C - 330*EP2) * A**6 / 720))
    return np.c_[x, y]


pts = []
for e in json.load(open(os.path.join(D, "a4_ways.json")))["elements"]:
    g = e["geometry"]
    lat = np.array([p["lat"] for p in g]); lon = np.array([p["lon"] for p in g])
    u = utm(lat, lon)
    for a, b, la in zip(u[:-1], u[1:], lon[:-1]):
        if LON_W <= la <= LON_E:
            n = max(1, int(np.hypot(*(b - a)) / 20))
            pts += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)]
pts = np.array(pts)
E0, E1 = int((pts[:, 0].min() - OUTER_M) // 1000), int((pts[:, 0].max() + OUTER_M) // 1000) + 1
N0, N1 = int((pts[:, 1].min() - OUTER_M) // 1000), int((pts[:, 1].max() + OUTER_M) // 1000) + 1
H, W = (N1 - N0) * 100, (E1 - E0) * 100


def band(radius):
    m = np.zeros((H, W), bool)
    rr = int(radius / RES)
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    disc = xx ** 2 + yy ** 2 <= rr ** 2
    cells = {(int((N1 * 1000 - y) / RES), int((x - E0 * 1000) / RES)) for x, y in pts[::2]}
    for r, c in cells:
        r0, r1, c0, c1 = max(0, r - rr), min(H, r + rr + 1), max(0, c - rr), min(W, c + rr + 1)
        m[r0:r1, c0:c1] |= disc[r0 - (r - rr):r1 - (r - rr), c0 - (c - rr):c1 - (c - rr)]
    return m


outer, detail = band(OUTER_M), band(DETAIL_M)
# minus the old territory (its own tiles are built already)
old = np.load(r"A:\blender_work\region_data\territory.npz")
old_tiles = {tuple(map(int, t)) for t in old["tiles"]}
for e, n in old_tiles:
    if E0 <= e < E1 and N0 <= n < N1:
        sl = (slice((N1 - n - 1) * 100, (N1 - n) * 100), slice((e - E0) * 100, (e - E0 + 1) * 100))
        outer[sl] = False; detail[sl] = False
tiles = sorted({(e, n) for e in range(E0, E1) for n in range(N0, N1)
                if outer[(N1 - n - 1) * 100:(N1 - n) * 100, (e - E0) * 100:(e - E0 + 1) * 100].any()})
detail_tiles = sorted({(e, n) for e, n in tiles
                       if detail[(N1 - n - 1) * 100:(N1 - n) * 100, (e - E0) * 100:(e - E0 + 1) * 100].any()})
print("E", E0, E1 - 1, "N", N0, N1 - 1, "tiles", len(tiles), "detail tiles", len(detail_tiles),
      "km² outer", round(outer.sum() * RES * RES / 1e6, 1), "detail", round(detail.sum() * RES * RES / 1e6, 1))
np.savez_compressed(os.path.join(D, "corridor.npz"), mask=outer, detail=detail, E0=E0, N1=N1, res=RES,
                    tiles=np.array(tiles), detail_tiles=np.array(detail_tiles), a4=pts.astype(np.float32))
json.dump([list(t) for t in tiles], open(os.path.join(D, "tiles_all.json"), "w"))
# chunks west to east, CHUNK tiles each; every chunk keeps the whole mask (prep reads only its tiles)
es = sorted({e for e, _ in tiles})
chunks, cur = [], []
for e in es:
    col_ = [t for t in tiles if t[0] == e]
    if cur and len(cur) + len(col_) > CHUNK:
        chunks.append(cur); cur = []
    cur += col_
chunks.append(cur)
for k, ch in enumerate(chunks):
    cd = os.path.join(D, f"chunk_{k}")
    os.makedirs(cd, exist_ok=True)
    np.savez_compressed(os.path.join(cd, "territory.npz"), mask=outer, detail=detail, E0=E0, N1=N1, res=RES,
                        grey=pts.astype(np.float32), tiles=np.array(ch))
    json.dump([list(t) for t in ch], open(os.path.join(cd, "territory_tiles.json"), "w"))
    print("chunk", k, "E", ch[0][0], "..", ch[-1][0], len(ch), "tiles")

