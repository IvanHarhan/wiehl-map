"""The in-game map (MapMinimap.cs): orthophoto tiles shrunk to 4 m a pixel, roads drawn over them, everything outside
the user's territory dimmed with its outline in pink; plus the road graph for routes.
Run with system python: python make_minimap.py PREPARED_DIR OUT_DIR REGION_JSON (the export's region.json, for the origin)
Writes OUT_DIR/minimap.jpg and OUT_DIR/minimap.json (image placement + road graph, both in local metres, the frame
the Blender build and Unity export use; Unity = (Ox - x, z, Oy - y) * map scale around origin O)."""
import json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PREP, OUT = sys.argv[1], sys.argv[2]
REGION_JSON = sys.argv[3] if len(sys.argv) > 3 else None
RES = 4.0
meta = json.load(open(os.path.join(PREP, "meta.json")))
UX0, UY0, UX1, UY1 = meta["utm_bbox"]
AFF = np.array(json.loads(meta["affine"]) if isinstance(meta["affine"], str) else meta["affine"])
INV = np.linalg.inv(AFF[:2])
LAT0, LON0 = meta["lat0"], meta["lon0"]
W, H = int((UX1 - UX0) / RES), int((UY1 - UY0) / RES)
R = 6378137.0


def tm(lat, lon):
    la, lo = np.radians(lat), np.radians(lon - LON0)
    b = np.sin(lo) * np.cos(la)
    return np.c_[0.5 * R * np.log((1 + b) / (1 - b)), R * (np.arctan(np.tan(la) / np.cos(lo)) - math.radians(LAT0))]


def local_to_px(xy):
    d = (np.asarray(xy, float) - AFF[2]) @ INV
    return np.c_[d[:, 0] / RES, (UY1 - UY0 - d[:, 1]) / RES]


# ---- orthophoto ----
img = Image.new("RGB", (W, H), (58, 66, 52))
odir = os.path.join(PREP, "ortho")
n = 0
for f in os.listdir(odir):
    if not f.endswith(".jpg"):
        continue
    e, nn = map(int, f[:-4].split("_"))
    tile = Image.open(os.path.join(odir, f)).convert("RGB").resize((int(1000 / RES), int(1000 / RES)), Image.LANCZOS)
    img.paste(tile, (int((e * 1000 - UX0) / RES), int((UY1 - (nn + 1) * 1000) / RES)))
    n += 1
print("tiles", n, "image", W, H)

# ---- territory: outside dimmed and greyed, outline in pink ----
ter = np.load(os.path.join(PREP, "territory.npz"))
mask = Image.fromarray((ter["mask"] * 255).astype(np.uint8))
tres, E0, N1 = float(ter["res"]), float(ter["E0"]) * 1000, float(ter["N1"]) * 1000  # corner in km
big = Image.new("L", (W, H), 0)
mw, mh = mask.size
big.paste(mask.resize((int(mw * tres / RES), int(mh * tres / RES)), Image.NEAREST), (int((E0 - UX0) / RES), int((UY1 - N1) / RES)))
dim = Image.blend(img.convert("L").convert("RGB"), Image.new("RGB", (W, H), (20, 22, 26)), 0.55)
img = Image.composite(img, dim, big)
edge = big.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(5))
img.paste((255, 70, 160), mask=edge)

# ---- roads ----
MAIN = {"motorway", "trunk", "primary", "secondary", "primary_link", "secondary_link"}
CAR = MAIN | {"tertiary", "tertiary_link", "unclassified", "residential", "living_street", "service"}
roads = json.load(open(os.path.join(os.path.dirname(PREP), "osm_roads.json"), encoding="utf-8"))["elements"]
draw = ImageDraw.Draw(img)
layers = []
nodes, index, edges = [], {}, []


def node(p):
    k = (round(p["lat"], 7), round(p["lon"], 7))
    if k not in index:
        index[k] = len(nodes)
        nodes.append(k)
    return index[k]


for e in roads:
    t = e.get("tags", {})
    hw = t.get("highway")
    g = e.get("geometry") or []
    if hw not in CAR or len(g) < 2:
        continue
    xy = tm(np.array([p["lat"] for p in g]), np.array([p["lon"] for p in g]))
    px = local_to_px(xy)
    inside = [(0 <= x < W and 0 <= y < H and big.getpixel((int(x), int(y))) > 0) for x, y in px]
    if not any(inside):
        continue
    layers.append((hw in MAIN, hw == "service", [tuple(q) for q in px]))
    ids = [node(p) for p in g]
    oneway = t.get("oneway") == "yes" or t.get("junction") == "roundabout"
    through = hw in MAIN or hw in ("tertiary", "tertiary_link")  # the roads traffic keeps to
    for a, b in zip(ids[:-1], ids[1:]):
        edges.append((a, b, (1 if oneway else 0) + (2 if through else 0)))
for main, service, pts in sorted(layers, key=lambda l: l[0]):
    wid = 5 if main else 2 if service else 3
    draw.line(pts, fill=(40, 40, 40), width=wid + 2)
for main, service, pts in sorted(layers, key=lambda l: l[0]):
    wid = 5 if main else 2 if service else 3
    draw.line(pts, fill=(250, 214, 120) if main else (235, 235, 230), width=wid)
img.save(os.path.join(OUT, "minimap.jpg"), quality=88)

# ---- graph in local metres ----
lat = np.array([k[0] for k in nodes]); lon = np.array([k[1] for k in nodes])
nxy = tm(lat, lon)
origin = json.load(open(REGION_JSON))["origin_local"] if REGION_JSON else [0.0, 0.0, 0.0]
json.dump({"res": RES, "width": W, "height": H, "utm_w": UX1 - UX0, "utm_h": UY1 - UY0, "origin": origin[:2],
           "aff": AFF[:2].ravel().tolist(), "aff2": AFF[2].tolist(), "inv": INV.ravel().tolist(),
           "nodes": np.round(nxy, 1).ravel().tolist(), "edges": np.array(edges, np.int64).ravel().tolist()},
          open(os.path.join(OUT, "minimap.json"), "w"))
print("roads", len(layers), "graph nodes", len(nodes), "edges", len(edges))
