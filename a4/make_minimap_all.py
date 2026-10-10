"""The in-game map for the whole playable area: the main region (region_data/prepared) plus the A4 corridor chunks
(a4/chunk_K/prepared), in the format of region_data/make_minimap.py (MapMinimap.cs reads it unchanged).
One affine UTM -> local fit over the union (a few metres off at the far ends, under a pixel of the 4 m map).
Usage: python make_minimap_all.py OUT_DIR"""
import glob, json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

Image.MAX_IMAGE_PIXELS = None
OUT = sys.argv[1]
W_ = r"A:\blender_work"
PARTS = [os.path.join(W_, "region_data", "prepared")] + sorted(glob.glob(os.path.join(W_, "a4", "chunk_*", "prepared")))
ROADS = [os.path.join(W_, "region_data", "osm_roads.json")] + sorted(glob.glob(os.path.join(W_, "a4", "chunk_*", "osm_roads.json")))
RES = 4.0
metas = [json.load(open(os.path.join(p, "meta.json"))) for p in PARTS]
UX0 = min(m["utm_bbox"][0] for m in metas); UY0 = min(m["utm_bbox"][1] for m in metas)
UX1 = max(m["utm_bbox"][2] for m in metas); UY1 = max(m["utm_bbox"][3] for m in metas)
LAT0, LON0 = metas[0]["lat0"], metas[0]["lon0"]
W, H = int((UX1 - UX0) / RES), int((UY1 - UY0) / RES)
R = 6378137.0
E_A, E_F, K0, UTM_LON0 = 6378137.0, 1 / 298.257222101, 0.9996, math.radians(9)
E2 = E_F * (2 - E_F); EP2 = E2 / (1 - E2)


def utm_inv(x, y):
    e1 = (1 - np.sqrt(1 - E2)) / (1 + np.sqrt(1 - E2))
    mu = y / K0 / (E_A * (1 - E2/4 - 3*E2**2/64 - 5*E2**3/256))
    p1 = mu + (3*e1/2 - 27*e1**3/32) * np.sin(2*mu) + (21*e1**2/16 - 55*e1**4/32) * np.sin(4*mu) + (151*e1**3/96) * np.sin(6*mu) + (1097*e1**4/512) * np.sin(8*mu)
    N1 = E_A / np.sqrt(1 - E2 * np.sin(p1)**2); T1 = np.tan(p1)**2; C1 = EP2 * np.cos(p1)**2
    R1 = E_A * (1 - E2) / (1 - E2 * np.sin(p1)**2) ** 1.5; Dd = (x - 500000) / (N1 * K0)
    lat = p1 - (N1 * np.tan(p1) / R1) * (Dd*Dd/2 - (5 + 3*T1 + 10*C1 - 4*C1*C1 - 9*EP2) * Dd**4/24 + (61 + 90*T1 + 298*C1 + 45*T1*T1 - 252*EP2 - 3*C1*C1) * Dd**6/720)
    lon = UTM_LON0 + (Dd - (1 + 2*T1 + C1) * Dd**3/6 + (5 - 2*C1 + 28*T1 - 3*C1*C1 + 8*EP2 + 24*T1*T1) * Dd**5/120) / np.cos(p1)
    return np.degrees(lat), np.degrees(lon)


def tm(lat, lon):
    la, lo = np.radians(lat), np.radians(lon - LON0)
    b = np.sin(lo) * np.cos(la)
    return np.c_[0.5 * R * np.log((1 + b) / (1 - b)), R * (np.arctan(np.tan(la) / np.cos(lo)) - math.radians(LAT0))]


gx, gy = np.meshgrid(np.linspace(UX0, UX1, 41), np.linspace(UY0, UY1, 21))
gx, gy = gx.ravel(), gy.ravel()
dst = tm(*utm_inv(gx, gy))
AFF = np.linalg.lstsq(np.c_[gx - UX0, gy - UY0, np.ones_like(gx)], dst, rcond=None)[0]
err = float(np.abs(np.c_[gx - UX0, gy - UY0, np.ones_like(gx)] @ AFF - dst).max())
INV = np.linalg.inv(AFF[:2])
print("union E", UX0, UX1, "N", UY0, UY1, "image", W, H, "affine error m", round(err, 2), flush=True)


def local_to_px(xy):
    d = (np.asarray(xy, float) - AFF[2]) @ INV
    return np.c_[d[:, 0] / RES, (UY1 - UY0 - d[:, 1]) / RES]


# ---- orthophoto ----
img = Image.new("RGB", (W, H), (58, 66, 52))
side = int(1000 / RES)
seen = set()
for p in PARTS:
    for f in sorted(glob.glob(os.path.join(p, "ortho", "*.jpg"))):
        key = os.path.basename(f)[:-4]
        if key in seen:
            continue
        seen.add(key)
        e, n = map(int, key.split("_"))
        tile = Image.open(f).convert("RGB").resize((side, side), Image.LANCZOS)
        img.paste(tile, (int((e * 1000 - UX0) / RES), int((UY1 - (n + 1) * 1000) / RES)))
print("ortho tiles", len(seen), flush=True)

# ---- playable area: the region's outline and the corridor, outside dimmed, edge in pink ----
big = Image.new("L", (W, H), 0)
for path in (os.path.join(PARTS[0], "territory.npz"), os.path.join(W_, "a4", "corridor.npz")):
    t = np.load(path)
    m = Image.fromarray((t["mask"] * 255).astype(np.uint8))
    tres, E0, N1 = float(t["res"]), float(t["E0"]) * 1000, float(t["N1"]) * 1000
    mw, mh = m.size
    layer = Image.new("L", (W, H), 0)
    layer.paste(m.resize((int(mw * tres / RES), int(mh * tres / RES)), Image.NEAREST), (int((E0 - UX0) / RES), int((UY1 - N1) / RES)))
    big = Image.fromarray(np.maximum(np.asarray(big), np.asarray(layer)))
dim = Image.blend(img.convert("L").convert("RGB"), Image.new("RGB", (W, H), (20, 22, 26)), 0.55)
img = Image.composite(img, dim, big)
edge = big.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(5))
img.paste((255, 70, 160), mask=edge)
bigA = np.asarray(big)

# ---- roads (each OSM way once) + graph ----
MAIN = {"motorway", "trunk", "primary", "secondary", "primary_link", "secondary_link", "motorway_link", "trunk_link"}
CAR = MAIN | {"tertiary", "tertiary_link", "unclassified", "residential", "living_street", "service"}
draw = ImageDraw.Draw(img)
layers, nodes, index, edges, ways_seen = [], [], {}, [], set()


def node(p):
    k = (round(p["lat"], 7), round(p["lon"], 7))
    if k not in index:
        index[k] = len(nodes)
        nodes.append(k)
    return index[k]


for rf in ROADS:
    for e in json.load(open(rf, encoding="utf-8"))["elements"]:
        if e.get("id") in ways_seen:
            continue
        t = e.get("tags", {})
        hw = t.get("highway")
        g = e.get("geometry") or []
        if hw not in CAR or len(g) < 2:
            continue
        xy = tm(np.array([p["lat"] for p in g]), np.array([p["lon"] for p in g]))
        px = local_to_px(xy)
        ix, iy = px[:, 0].astype(int), px[:, 1].astype(int)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        if not ok.any() or not (bigA[iy[ok], ix[ok]] > 0).any():
            continue
        ways_seen.add(e.get("id"))
        motorway = hw.startswith("motorway")
        layers.append((2 if motorway else 1 if hw in MAIN else 0, hw == "service", [tuple(q) for q in px]))
        ids = [node(p) for p in g]
        oneway = t.get("oneway") == "yes" or t.get("junction") == "roundabout" or motorway
        through = hw in MAIN or hw in ("tertiary", "tertiary_link")
        for a, b in zip(ids[:-1], ids[1:]):
            edges.append((a, b, (1 if oneway else 0) + (2 if through else 0)))
FILL = {2: (255, 170, 60), 1: (250, 214, 120), 0: (235, 235, 230)}
for rank, service, pts in sorted(layers, key=lambda l: l[0]):
    wid = 6 if rank == 2 else 5 if rank == 1 else 2 if service else 3
    draw.line(pts, fill=(40, 40, 40), width=wid + 2)
for rank, service, pts in sorted(layers, key=lambda l: l[0]):
    wid = 6 if rank == 2 else 5 if rank == 1 else 2 if service else 3
    draw.line(pts, fill=FILL[rank], width=wid)
img.save(os.path.join(OUT, "minimap.jpg"), quality=88)

lat = np.array([k[0] for k in nodes]); lon = np.array([k[1] for k in nodes])
nxy = tm(lat, lon)
origin = json.load(open(os.path.join(W_, "unity_export", "region", "region.json")))["origin_local"]
json.dump({"res": RES, "width": W, "height": H, "utm_w": UX1 - UX0, "utm_h": UY1 - UY0, "origin": origin[:2],
           "aff": AFF[:2].ravel().tolist(), "aff2": AFF[2].tolist(), "inv": INV.ravel().tolist(),
           "nodes": np.round(nxy, 1).ravel().tolist(), "edges": np.array(edges, np.int64).ravel().tolist()},
          open(os.path.join(OUT, "minimap.json"), "w"))
print("MINIMAP_DONE roads", len(layers), "graph nodes", len(nodes), "edges", len(edges))
