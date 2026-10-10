"""Köln on top of the A4 corridor: the centre (Messe/Deutz across the Rhine to the Hbf and on to Neumarkt) and the
road in from the corridor's west end (A4 / Kreuz Köln-Ost / B 55a). Grows a4/corridor.npz westwards (backup
corridor_before_koeln.npz) and splits the new tiles: those next to chunk 0 join it (chunk 0 is prepared again),
the city itself is chunk_4. Writes tiles_koeln.json (the tiles to download)."""
import json, math, os, shutil
import numpy as np

D = r"A:\blender_work\a4"
RES = 10
CENTRE = (50.930, 6.938, 50.953, 6.994)            # lat0, lon0, lat1, lon1: Neumarkt .. Messe
ROUTE_REFS = {"B 55a", "A 4", "A 3;A 4", "A 3"}     # the Stadtautobahn and the Kreuz Köln-Ost
ROUTE_LON = (6.975, 7.075)
DETAIL_M, OUTER_M = 300.0, 700.0
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


if not os.path.exists(os.path.join(D, "corridor_before_koeln.npz")):
    shutil.copy(os.path.join(D, "corridor.npz"), os.path.join(D, "corridor_before_koeln.npz"))
old = np.load(os.path.join(D, "corridor_before_koeln.npz"))
oE0, oN1 = int(old["E0"]), int(old["N1"])

# route points
pts = []
for e in json.load(open(os.path.join(D, "koeln_route.json")))["elements"]:
    t = e["tags"]
    if t.get("ref", "") not in ROUTE_REFS and not (t["highway"].endswith("_link") and t["highway"].startswith(("motorway", "trunk"))):
        continue
    g = e["geometry"]
    lat = np.array([p["lat"] for p in g]); lon = np.array([p["lon"] for p in g])
    if lon.max() < ROUTE_LON[0] or lon.min() > ROUTE_LON[1]:
        continue
    u = utm(lat, lon)
    for a, b in zip(u[:-1], u[1:]):
        n = max(1, int(np.hypot(*(b - a)) / 20))
        pts += [a + (b - a) * s for s in np.linspace(0, 1, n + 1)]
pts = np.array(pts)
cx = utm(np.array([CENTRE[0], CENTRE[0], CENTRE[2], CENTRE[2]]), np.array([CENTRE[1], CENTRE[3], CENTRE[1], CENTRE[3]]))
E0 = min(oE0, int(min(pts[:, 0].min() - OUTER_M, cx[:, 0].min()) // 1000))
N1 = max(oN1, int(max(pts[:, 1].max() + OUTER_M, cx[:, 1].max()) // 1000) + 1)
N0 = min(oN1 - old["mask"].shape[0] // 100, int(min(pts[:, 1].min() - OUTER_M, cx[:, 1].min()) // 1000))
E1 = oE0 + old["mask"].shape[1] // 100
H, W = (N1 - N0) * 100, (E1 - E0) * 100


def grow(m):
    out = np.zeros((H, W), bool)
    r0, c0 = (N1 - oN1) * 100, (oE0 - E0) * 100
    out[r0:r0 + m.shape[0], c0:c0 + m.shape[1]] = m
    return out


outer, detail = grow(old["mask"]), grow(old["detail"])


def stamp(m, P, radius):
    rr = int(radius / RES)
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    disc = xx ** 2 + yy ** 2 <= rr ** 2
    for r, c in {(int((N1 * 1000 - y) / RES), int((x - E0 * 1000) / RES)) for x, y in P}:
        a0, a1, b0, b1 = max(0, r - rr), min(H, r + rr + 1), max(0, c - rr), min(W, c + rr + 1)
        if a0 < a1 and b0 < b1:
            m[a0:a1, b0:b1] |= disc[a0 - (r - rr):a1 - (r - rr), b0 - (c - rr):b1 - (c - rr)]


new_outer, new_detail = np.zeros((H, W), bool), np.zeros((H, W), bool)
stamp(new_outer, pts, OUTER_M)
stamp(new_detail, pts, DETAIL_M)
# the centre, whole
yy, xx = np.mgrid[0:H, 0:W]
ux, uy = E0 * 1000 + (xx + 0.5) * RES, N1 * 1000 - (yy + 0.5) * RES
inside = (ux >= cx[:, 0].min()) & (ux <= cx[:, 0].max()) & (uy >= cx[:, 1].min()) & (uy <= cx[:, 1].max())
new_outer |= inside
new_detail |= inside
# not where tiles exist already (main region, corridor chunks 1-3); chunk 0's tiles get the new mask too
existing = {tuple(map(int, t)) for t in np.load(r"A:\blender_work\region_data\territory.npz")["tiles"]}
for k in (1, 2, 3):
    existing |= {tuple(map(int, t)) for t in np.load(os.path.join(D, f"chunk_{k}", "territory.npz"))["tiles"]}
for e, n in existing:
    if E0 <= e < E1 and N0 <= n < N1:
        sl = (slice((N1 - n - 1) * 100, (N1 - n) * 100), slice((e - E0) * 100, (e - E0 + 1) * 100))
        new_outer[sl] = False; new_detail[sl] = False
outer |= new_outer
detail |= new_detail


def tiles_of(m):
    return sorted({(e, n) for e in range(E0, E1) for n in range(N0, N1)
                   if m[(N1 - n - 1) * 100:(N1 - n) * 100, (e - E0) * 100:(e - E0 + 1) * 100].any()} - existing)


all_tiles = tiles_of(outer)
detail_tiles = [t for t in tiles_of(detail)]
chunk0_old = {tuple(map(int, t)) for t in np.load(os.path.join(D, "chunk_0", "territory.npz"))["tiles"]}
new_tiles = [t for t in all_tiles if t not in chunk0_old]
city = sorted(t for t in all_tiles if t[0] < 362)
chunk0 = sorted(t for t in all_tiles if t[0] >= 362)
old_detail = {tuple(map(int, t)) for t in old["detail_tiles"]}
np.savez_compressed(os.path.join(D, "corridor.npz"), mask=outer, detail=detail, E0=E0, N1=N1, res=RES,
                    tiles=np.array(sorted(set(map(tuple, old["tiles"].tolist())) | set(all_tiles))),
                    detail_tiles=np.array(sorted(old_detail | set(detail_tiles))), a4=old["a4"])
for k, ch in ((0, chunk0), (4, city)):
    cd = os.path.join(D, f"chunk_{k}")
    os.makedirs(cd, exist_ok=True)
    np.savez_compressed(os.path.join(cd, "territory.npz"), mask=outer, detail=detail, E0=E0, N1=N1, res=RES,
                        grey=pts.astype(np.float32), tiles=np.array(ch))
    json.dump([list(t) for t in ch], open(os.path.join(cd, "territory_tiles.json"), "w"))
json.dump([list(t) for t in new_tiles], open(os.path.join(D, "tiles_koeln.json"), "w"))
print("grid E", E0, E1 - 1, "N", N0, N1 - 1, "| new tiles", len(new_tiles), "| chunk 0", len(chunk0), "(was", len(chunk0_old),
      ") | chunk 4 (city)", len(city), "| new km2", round(new_outer.sum() * RES * RES / 1e6, 1))
