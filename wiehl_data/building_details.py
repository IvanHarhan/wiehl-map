"""Run from build_blend.py (exec) after the facade groups are built, before bm.to_mesh():
building roles from OSM (Rewe glass, supermarket/ice-rink halls, schools, churches), front doors,
garage doors, shop fronts, and window sills as real geometry.
Uses build_blend globals: bm, groups, zmin, zmax, bkind, funcs, V, FL, FT, FB, starts, vb, n_b, Z, col, P and face layers."""
import math
import numpy as np

sc = bpy.context.scene
SHOP_POI = {"restaurant", "cafe", "fast_food", "pub", "bank", "pharmacy", "bakery", "butcher", "clothes", "shoes",
            "florist", "optician", "hairdresser", "chemist", "dry_cleaning", "tailor", "sports", "kiosk", "seafood",
            "post_office", "alcohol", "supermarket", "yes", "convenience", "variety_store", "discount", "beverages",
            "mobile_phone", "electronics", "hardware", "doityourself", "books", "jewelry", "beauty", "ice_cream",
            "travel_agency", "department_store", "gift", "stationery", "pet", "toys",
            "gambling", "casino", "bookmaker", "lottery", "adult_gaming_centre"}
# Spielhallen, betting shops: dark sign with gold lettering, windows covered in frosted foil (as the law wants)
GAMBLING = {"gambling", "casino", "bookmaker", "adult_gaming_centre"}
ROAD_CLASSES = {"primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link", "residential",
                "unclassified", "living_street", "service", "pedestrian"}
N_DOOR_COLOURS, N_SIGN_COLOURS = 5, 6


def tm_arr(lat, lon):
    R = 6378137.0
    la, lo = np.radians(lat), np.radians(lon - sc["lon"])
    b = np.sin(lo) * np.cos(la)
    return np.c_[0.5 * R * np.log((1 + b) / (1 - b)), R * (np.arctan(np.tan(la) / np.cos(lo)) - math.radians(sc["lat"]))]


def in_poly(pts, poly):
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    hit = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
    return hit.sum(1) % 2 == 1


# ---------------- footprints, centroids, nearest road ----------------
foot = [[] for _ in range(n_b)]
for s, n, t, b in zip(starts, FL, FT, FB):
    if t == 2:
        foot[b].append(V[s:s + n, :2])
area = np.array([sum(abs(np.cross(p, np.roll(p, -1, axis=0)).sum()) / 2 for p in fp) for fp in foot])
cen = np.zeros((n_b, 2)); np.add.at(cen, vb, V[:, :2])
cen /= np.maximum(np.bincount(vb, minlength=n_b), 1)[:, None]


def building_at(pt, reach=15.0):
    near = np.nonzero(np.linalg.norm(cen - pt, axis=1) < 150)[0]
    for b in near:
        if any(in_poly(pt[None], p)[0] for p in foot[b]):
            return int(b)
    near = [b for b in near if np.linalg.norm(cen[b] - pt) < reach]
    return int(min(near, key=lambda b: np.linalg.norm(cen[b] - pt))) if near else -1


segs = []
for e in json.load(open(os.path.join(os.path.dirname(P), "osm_roads.json")))["elements"]:
    if e.get("tags", {}).get("highway") in ROAD_CLASSES and len(e.get("geometry", [])) > 1:
        xy = tm_arr(np.array([p["lat"] for p in e["geometry"]]), np.array([p["lon"] for p in e["geometry"]]))
        segs.append(np.c_[xy[:-1], xy[1:]])
segs = np.vstack(segs)
sa, sd = segs[:, :2], segs[:, 2:] - segs[:, :2]


def nearest_road(pts):
    out = np.empty_like(pts)
    for i in range(0, len(pts), 64):
        p = pts[i:i + 64, None, :]
        t = np.clip(((p - sa) * sd).sum(2) / np.maximum((sd * sd).sum(1), 1e-9), 0, 1)
        q = sa + sd * t[..., None]
        out[i:i + 64] = q[np.arange(len(p)), np.argmin(((q - p) ** 2).sum(2), axis=1)]
    return out


# ---------------- roles from OSM ----------------
kind2 = bkind.copy()
shop_b, notes = set(), []
extra = json.load(open(os.path.join(os.path.dirname(P), "osm_extra.json"), encoding="utf-8"))["elements"]


def osm_point(**tags):
    """the matching OSM node nearest the Wiehl origin (the region has several stations and town halls)"""
    pts = [tm_arr(np.array([e["lat"]]), np.array([e["lon"]]))[0] for e in extra
           if e["type"] == "node" and all(e.get("tags", {}).get(k) == v for k, v in tags.items())]
    return min(pts, key=lambda p: float(np.hypot(*p)))


def bld_near(xy, tol=3.0):
    """the building whose centroid is at xy — ids change whenever the map grows, the local frame does not"""
    d = np.linalg.norm(cen - np.array(xy), axis=1)
    if d.min() > tol:
        print("WARNING no building at", xy, "nearest", round(float(d.min()), 1))
    return int(d.argmin())


# checked by hand with the user on 2026-09-23 (the OSM points alone picked wrong ones).
# stores: glass front towards the station or the road, red band, kind of the other walls, sign text
if CHUNK:  # an A4 corridor chunk (a4/): none of Wiehl's hand-picked buildings are in it
    REWE = REWE_DRINKS = ALDI = JOBCENTER = OLD_TOWNHALL = BANK = None
    STORES = {}
else:
    REWE, REWE_DRINKS, ALDI = bld_near((28.28, -103.88)), bld_near((163.02, -126.94)), bld_near((-76.78, 411.99))
    STORES = {REWE: ("station", 1, 7, "REVO"), REWE_DRINKS: ("road", 1, 7, "REVO Getränke"), ALDI: ("parking", 0, 7, "ALDO")}
    JOBCENTER, OLD_TOWNHALL, BANK = bld_near((-81.18, 444.42)), bld_near((-27.33, 225.94)), bld_near((-69.9, 226.82))
    NEW_TOWNHALL = building_at(osm_point(amenity="townhall"))
    if NEW_TOWNHALL >= 0:
        STORES[NEW_TOWNHALL] = ("road", 0, 1, "RATHAUS")
HAND = set(STORES) | {b for b in (JOBCENTER, OLD_TOWNHALL, BANK) if b is not None}
try:
    station_pt = osm_point(railway="station")
except ValueError:  # an A4 chunk without a railway station
    station_pt = None
shop_name = {}  # building -> (OSM name, gambling?) of its first shop: the building's own sign colour
shop_pois = []  # every named shop: (building, point, name, gambling) — each gets its own sign over its own front
for e in extra:
    tg = e.get("tags", {})
    if e["type"] == "node":
        if tg.get("shop") in SHOP_POI or tg.get("amenity") in SHOP_POI or tg.get("leisure") in SHOP_POI:
            b = building_at(tm_arr(np.array([e["lat"]]), np.array([e["lon"]]))[0])
            if b >= 0 and b not in HAND:
                shop_b.add(b)
                kinds = {tg.get("amenity"), tg.get("leisure"), tg.get("shop")}
                gamble = "bet" if "bookmaker" in kinds else "hall" if kinds & GAMBLING else ""
                name = tg.get("name") or tg.get("brand") or ("Spielhalle" if gamble else "")
                if name and (b not in shop_name or gamble):
                    shop_name[b] = (name, gamble)
                if name:
                    shop_pois.append((b, tm_arr(np.array([e["lat"]]), np.array([e["lon"]]))[0], name, gamble))
    elif e["type"] == "way" and "geometry" in e:
        poly = tm_arr(np.array([p["lat"] for p in e["geometry"]]), np.array([p["lon"] for p in e["geometry"]]))
        inside = np.nonzero(in_poly(cen, poly))[0]
        if tg.get("leisure") in ("ice_rink", "sports_centre") and tg.get("building"):
            role = 5
        elif tg.get("building") == "church":
            role = 3
        elif tg.get("amenity") == "school" or tg.get("building") == "school":
            role = 6
        else:
            continue
        for b in inside:
            if kind2[b] != 2 and zmax[b] - zmin[b] > 4:
                kind2[b] = role
        notes.append(f"{tg.get('name', role)} -> {list(inside)}")
for b, (_, _, other, text) in STORES.items():
    kind2[b] = other
    notes.append(f"{text} -> {b} ({area[b]:.0f} m2)")
if not CHUNK:
    kind2[JOBCENTER], kind2[OLD_TOWNHALL] = 1, 8
    shop_b.add(BANK)

# only what is surely lived in gets house windows (the user, 2026-09-24): ALKIS dwellings (31001_1xxx) big enough to
# live in. Business buildings keep windows where OSM puts a shop, office or amenity in them, or when they stand
# 2+ storeys in a town; barns, depots, workshops, garages, canopies and unknown codes get plain walls, big ones a
# hall's clerestory
poi_b = set(shop_b)
for e in extra:
    tg = e.get("tags", {})
    if not (tg.get("shop") or tg.get("office") or tg.get("amenity") or tg.get("craft")):
        continue
    if e["type"] == "node":
        b = building_at(tm_arr(np.array([e["lat"]]), np.array([e["lon"]]))[0], reach=0.0)
        if b >= 0:
            poi_b.add(b)
    elif e["type"] == "way" and "geometry" in e and tg.get("building"):
        poly = tm_arr(np.array([p["lat"] for p in e["geometry"]]), np.array([p["lon"] for p in e["geometry"]]))
        poi_b.update(np.nonzero(in_poly(cen, poly))[0].tolist())
DWELLING_NOT = ("31001_1312", "31001_1313")  # garden houses
hb_ = zmax - zmin
in_towns = np.isin(np.floor((((cen - AFF[2]) @ INV)[:, 0] + UX0) / 1000).astype(int) * 10000
                   + np.floor((((cen - AFF[2]) @ INV)[:, 1] + UY0) / 1000).astype(int),
                   [int(k[:3]) * 10000 + int(k[4:]) for k in meta.get("detail", [])]) if REGION else np.ones(n_b, bool)
n_plain = n_hall = 0
for b in range(n_b):
    if b in HAND or kind2[b] in (3, 4, 5, 6, 8):
        continue
    f = funcs[b]
    if f.startswith("31001_1") and f not in DWELLING_NOT:
        if area[b] < 35 or hb_[b] < 3.5:
            kind2[b] = 2
            n_plain += 1
        continue
    if f.startswith("31001_3") and hb_[b] >= 3.5 and area[b] >= 35:
        continue  # public buildings: offices, kindergartens, homes
    if (b in poi_b and hb_[b] >= 3.0) or (f == "31001_2000" and in_towns[b] and hb_[b] >= 7.0 and area[b] < 2000):
        kind2[b] = 1
        continue
    kind2[b] = 5 if area[b] > 400 and hb_[b] >= 5.0 else 2
    n_hall += kind2[b] == 5
    n_plain += kind2[b] == 2
print("KINDS plain", n_plain, "halls", n_hall, "business with windows", int((kind2 == 1).sum()), "houses", int((kind2 == 0).sum()))

bdir = nearest_road(cen)
rng_d = np.random.default_rng(5)
door_col = rng_d.integers(0, N_DOOR_COLOURS, n_b)
sign_col = rng_d.integers(0, N_SIGN_COLOURS, n_b)
if BANK is not None:
    sign_col[BANK] = 0  # red
# the chains' own colours on the sign band (shader palette: 0 red 1 blue 2 dark 3 green 4 orange 5 white)
BRAND_COLOUR = [(("kaufland", "rewe", "sparkasse", "rossmann", "penny", "kik", "apotheke", "admiral", "takko", "vodafone",
                  "deichmann", "mediamarkt", "media markt"), 0),
                (("aldi", "lidl", "action", "volksbank", "raiffeisen", "tipico", "bet3000", "telekom", "o2", "fressnapf",
                  "euronics", "hornbach", "obi"), 1),
                (("edeka", "netto", "sportwetten", "tedi", "mcdonald", "burger king", "shell", "aral"), 4),
                (("dm", "bäckerei", "baeckerei", "bakery", "müller", "c&a", "h&m"), 5),
                (("grün", "gruen", "blume", "bio", "apo"), 3)]


def brand_colour(name, gamble):
    low = name.lower()
    for keys, colour in BRAND_COLOUR:
        if any(low == k or low.startswith(k + " ") or f" {k}" in f" {low}" for k in keys):
            return colour
    return 2 if gamble else None


for b, (nm, gamble) in shop_name.items():
    c = brand_colour(nm, gamble)
    if c is not None:
        sign_col[b] = c

# ---------------- per facade: doors, shop fronts, sills ----------------
info = []
for fs in groups.values():
    b = fs[0][l_b]
    n2 = Vector((fs[0].normal.x, fs[0].normal.y, 0)).normalized()
    vs = {v for f in fs for v in f.verts}
    c = np.array([sum(v.co.x for v in vs) / len(vs), sum(v.co.y for v in vs) / len(vs)])
    to_road = bdir[b] - c
    dist = float(np.linalg.norm(to_road))
    facing = float((to_road / max(dist, 1e-6)) @ np.array((n2.x, n2.y))) if dist > 0.5 else 0.0
    info.append((fs, b, n2, facing, dist, fs[0][l_wlen], fs[0][l_wbot], fs[0][l_weave]))

best = {}
for i, (fs, b, n2, facing, dist, wlen, wbot, weave) in enumerate(info):
    if wlen >= 2.5 and wbot < 1.5 and facing > 0.2 and (b not in best or facing > info[best[b]][3]):
        best[b] = i

# stores: which facades are glass, plus Rewe's second entrance on the side facing its car park
parks = [tm_arr(np.array([p["lat"] for p in e["geometry"]]), np.array([p["lon"] for p in e["geometry"]])).mean(0)
         for e in extra if e["type"] == "way" and e.get("tags", {}).get("amenity") == "parking"]
glass_dir = {}
for b, (towards, *_) in STORES.items():
    target = {"station": station_pt, "road": bdir[b]}.get(towards)
    if target is None:
        target = min(parks, key=lambda c: np.linalg.norm(c - cen[b]))
    v = target - cen[b]
    glass_dir[b] = v / max(np.linalg.norm(v), 1e-6)


def faces_dir(i, d):
    return float(np.array((info[i][2].x, info[i][2].y)) @ d)


extra_door = {}
if REWE is not None:
    park = min(parks, key=lambda c: np.linalg.norm(c - cen[REWE]))
    to_park = (park - cen[REWE]) / np.linalg.norm(park - cen[REWE])
    side = [i for i, x in enumerate(info) if x[1] == REWE and x[5] >= 6 and faces_dir(i, glass_dir[REWE]) <= 0.5]
    if side:
        extra_door[REWE] = max(side, key=lambda i: faces_dir(i, to_park))
glass_faces, door_at = {}, {}

sill_v, sill_f = [], []


def add_box(p0, ux, uy, uz):
    """box from corner p0 spanned by vectors ux, uy, uz (numpy)"""
    base_i = len(sill_v)
    for k in range(8):
        sill_v.append(tuple(p0 + ux * (k & 1) + uy * ((k >> 1) & 1) + uz * ((k >> 2) & 1)))
    for q in ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)):
        sill_f.append(tuple(base_i + j for j in q))


# each named shop onto the facade of its building nearest to its OSM point, at the point's place along the wall:
# several shops in one block (Wiehl's Hauptstraße) get a front and a sign each
def facade_frame(i):
    fs, b, n2 = info[i][:3]
    tan = Vector((-n2.y, n2.x, 0))
    vs = {v for f in fs for v in f.verts}
    return np.array(tan)[:2], min(v.co.dot(tan) for v in vs), np.array(n2)[:2], sum(v.co.dot(n2) for v in vs) / len(vs)


faces_of_b = {}
for i, x in enumerate(info):
    if x[5] >= 3.0 and x[6] < 1.5:
        faces_of_b.setdefault(x[1], []).append(i)
poi_on_face = {}  # facade -> [(u along it, name, gambling)]
for b, pt, nm, gamble in shop_pois:
    best_i, best_s = None, 1e9
    for i in faces_of_b.get(b, ()):
        t2, umin, n2_, d_plane = facade_frame(i)
        u = float(pt @ t2 - umin)
        off = max(0.0, -u, u - info[i][5])
        s = abs(float(pt @ n2_) - d_plane) + 2.0 * off - (3.0 if info[i][3] > 0.2 else 0.0)
        if s < best_s:
            best_i, best_s = i, s
    if best_i is not None:
        u = float(np.clip(pt @ facade_frame(best_i)[0] - facade_frame(best_i)[1], 0.0, info[best_i][5]))
        poi_on_face.setdefault(best_i, []).append((u, nm, gamble))
face_scol = {}
for i, lst in poi_on_face.items():
    c = brand_colour(lst[0][1], lst[0][2])
    if c is not None:
        face_scol[i] = c

n_doors = n_garages = n_shops = 0
for i, (fs, b, n2, facing, dist, wlen, wbot, weave) in enumerate(info):
    k = int(kind2[b])
    glass = b in glass_dir and faces_dir(i, glass_dir[b]) > 0.5
    if glass:
        k = 4
        glass_faces.setdefault(b, []).append(i)
    shop = (b in shop_b and k in (0, 1) and facing > 0.4 and dist < 30 and wbot < 1.5 and wlen >= 3) \
        or (i in poi_on_face and k in (0, 1, 7))
    spacing = 3.4 if k in (0, 8) else 4.5
    nwin = max(1, math.floor(wlen / spacing))
    bay = wlen / nwin
    door_u, door_t = 0.0, 0
    if extra_door.get(b) == i:
        door_u, door_t = wlen / 2, 3
    elif best.get(b) == i and not shop and k not in (4, 7):
        if k == 2 and funcs[b] == "31001_2460":
            if wlen >= 2.6:
                door_u, door_t = wlen / 2, 2
                n_garages += 1
        elif k != 2:
            door_u, door_t = (nwin // 2 + 0.5) * bay, 1
            n_doors += 1
    n_shops += shop
    if door_t:
        door_at[i] = door_u
    for f in fs:
        f[l_kind] = k
        f[l_door_u], f[l_door_t], f[l_shop] = door_u, door_t, float(shop)
        f[l_dcol], f[l_scol] = float(door_col[b]), float(face_scol.get(i, sign_col[b]))
        f[l_band] = float(STORES[b][1]) if glass else 0.0
        f[l_white] = 1.0 if b == JOBCENTER else 0.0

    # sills under the painted windows (same rules as wall_material)
    if k not in (0, 1, 8) or zmax[b] - zmin[b] <= 4 or wlen <= 1.8:
        continue
    tan = Vector((-n2.y, n2.x, 0))
    if Z.cross(fs[0].normal).normalized().dot(tan) < 0:
        tan = -tan
    base = Vector((0, 0, zmin[b]))
    vs = {v for f in fs for v in f.verts}
    umin = min((v.co - base).dot(tan) for v in vs)
    d_plane = sum(v.co.dot(n2) for v in vs) / len(vs)
    polys = [np.array([((v.co - base).dot(tan) - umin, v.co.z - zmin[b]) for v in f.verts]) for f in fs]
    tn, nn_ = np.array(tan), np.array(n2)
    for fl in np.arange(0, weave + 0.3 - 3.0 + 1e-6, 3.0):
        if fl <= wbot - 0.3 or (shop and fl < wbot + 3.0):
            continue
        for j in range(nwin):
            uc = (j + 0.5) * bay
            if door_t == 1 and fl < wbot + 3.0 and abs(uc - door_u) < bay / 2:
                continue
            probe = np.array([[uc, fl + 0.85], [uc, fl + 2.4]])
            if not any(in_poly(probe, p).all() for p in polys):
                continue
            half = (0.24 * bay + 0.06)
            p0 = tn * (uc + umin - half) + nn_ * (d_plane - 0.02)
            p0 = np.array((p0[0], p0[1], zmin[b] + fl + 0.72))
            add_box(p0, tn * 2 * half, nn_ * 0.15, np.array((0, 0, 0.06)))

if sill_f:
    sme = bpy.data.meshes.new("NRW_WindowSills")
    sme.from_pydata(sill_v, [], sill_f)
    sme.validate()
    sm = bpy.data.materials.new("NRW_Sill")
    sm.use_nodes = True
    sm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.62, 0.62, 0.6, 1)
    sm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.5
    sme.materials.append(sm)
    col.objects.link(bpy.data.objects.new("NRW_WindowSills", sme))

# ---------------- signs: lettering and logo panels on the facades ----------------
signs = bpy.data.collections.new("NRW_Signs")
col.children.link(signs)


def load_font(*paths):
    for p in paths:
        if os.path.exists(p):
            return bpy.data.fonts.load(p, check_existing=True)
    return None


FONT_BOLD = load_font(r"C:\Windows\Fonts\ariblk.ttf", r"C:\Windows\Fonts\arialbd.ttf")
FONT_SANS = load_font(r"C:\Windows\Fonts\arial.ttf")
FONT_SERIF = load_font(r"C:\Windows\Fonts\georgia.ttf", r"C:\Windows\Fonts\times.ttf")


def flat_mat(name, rgb, image=None):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = 0.35
    if image is not None:
        t = m.node_tree.nodes.new("ShaderNodeTexImage")
        t.image = image
        m.node_tree.links.new(t.outputs["Color"], b.inputs["Base Color"])
    return m


WHITE, DARK, RED = flat_mat("NRW_SignWhite", (0.95, 0.95, 0.95)), flat_mat("NRW_SignDark", (0.06, 0.06, 0.07)), \
    flat_mat("NRW_SignRed", (0.7, 0.02, 0.03))


def facade_point(i, u, z, out):
    fs, b, n2 = info[i][:3]
    tan = Vector((-n2.y, n2.x, 0))
    vs = {v for f in fs for v in f.verts}
    umin = min(v.co.dot(tan) for v in vs)
    d_plane = sum(v.co.dot(n2) for v in vs) / len(vs)
    p = tan * (u + umin) + n2 * (d_plane + out)
    return Vector((p.x, p.y, z)), n2


def put_text(text, i, u, z, size, font, mats, char_mats=None, align="CENTER", out=0.06):
    cu = bpy.data.curves.new("Sign_" + text, "FONT")
    cu.body, cu.size, cu.extrude = text, size, 0.03
    cu.align_x, cu.align_y = align, "CENTER"
    if font:
        cu.font = font
    for m in mats:
        cu.materials.append(m)
    for ci, mi in enumerate(char_mats or []):
        cu.body_format[ci].material_index = mi
    ob = bpy.data.objects.new("Sign_" + text, cu)
    p, n2 = facade_point(i, u, z, out)
    ob.location = p
    # text faces +Z and reads along +X: stand it up and turn it so it faces the facade normal
    ob.rotation_euler = (math.pi / 2, 0, math.atan2(n2.x, -n2.y))
    signs.objects.link(ob)
    return ob


def put_panel(name, i, u, z, w, h, mat, out=0.05):
    c, n2 = facade_point(i, u, z, out)
    t = Vector((-n2.y, n2.x, 0))
    corners = [c - t * w / 2 - Vector((0, 0, h / 2)), c + t * w / 2 - Vector((0, 0, h / 2)),
               c + t * w / 2 + Vector((0, 0, h / 2)), c - t * w / 2 + Vector((0, 0, h / 2))]
    me_p = bpy.data.meshes.new(name)
    me_p.from_pydata([tuple(p) for p in corners], [], [(0, 1, 2, 3)])
    me_p.uv_layers.new(name="UVMap").data.foreach_set("uv", [0, 0, 1, 0, 1, 1, 0, 1])
    me_p.materials.append(mat)
    signs.objects.link(bpy.data.objects.new(name, me_p))


def logo_stripes():
    """ALDO-style logo: navy square, thin light frame, four diagonal colour stripes, room for the name below"""
    S = 512
    yy, xx = (np.mgrid[0:S, 0:S] + 0.5) / S
    yy = 1 - yy
    rgba = np.zeros((S, S, 4), np.float32)
    rgba[...] = (0.01, 0.07, 0.3, 1)
    edge = (np.minimum(np.minimum(xx, 1 - xx), np.minimum(yy, 1 - yy)) < 0.035)
    rgba[edge] = (0.35, 0.7, 0.95, 1)
    diag = xx - yy * 0.9
    upper = (yy > 0.33) & (yy < 0.92) & (xx > 0.08) & (xx < 0.92)
    for k, c in enumerate(((0.3, 0.7, 0.95), (1.0, 0.8, 0.0), (1.0, 0.5, 0.05), (0.9, 0.15, 0.1))):
        m = upper & (np.abs(diag - (-0.35 + 0.17 * k)) < 0.055)
        rgba[m] = (*c, 1)
    img = bpy.data.images.new("nrw_logo_aldo", S, S, alpha=True)
    img.pixels.foreach_set(np.flipud(rgba).ravel())
    img.filepath_raw = os.path.join(TEX, "nrw_logo_aldo.png")
    img.file_format = "PNG"
    img.save()
    return img


bvh_b = BVHTree.FromBMesh(bm)


def unblocked(i):
    """no other wall of the model stands in front of this facade (stepped buildings hide their inner facades)"""
    wlen, wbot, weave = info[i][5:8]
    b = info[i][1]
    for frac in (0.3, 0.5, 0.7):
        p, n2 = facade_point(i, wlen * frac, zmin[b] + (wbot + weave) / 2, 0.05)
        loc, nor, _, dist = bvh_b.ray_cast(p, n2, 200.0)
        # a wall close in front hides the facade; a wall hit from behind means the facade is inside an attached building
        if loc is not None and (dist < 40 or nor.dot(n2) > 0):
            return False
    return True


def widest(ids):
    return max([i for i in ids if unblocked(i)] or ids, key=lambda i: info[i][5])


n_signs = 0
for b, (_, band_on, _, text) in STORES.items():
    if b not in glass_faces or b == REWE:  # rewe.py signs Rewe on its own shop fronts
        continue
    i = widest(glass_faces[b])
    wlen, wbot, weave = info[i][5:8]
    if text == "ALDO":
        top = zmin[b] + weave - 0.35
        put_panel("Logo_ALDO", i, wlen / 2, top - 1.3, 2.6, 2.6, flat_mat("NRW_LogoAldo", (1, 1, 1), logo_stripes()))
        put_text("ALDO", i, wlen / 2, top - 2.15, 0.62, FONT_BOLD, [WHITE], out=0.08)
    elif band_on:
        put_text(text, i, wlen / 2, zmin[b] + weave - 0.55, 1.35 if len(text) < 6 else 0.9, FONT_BOLD, [WHITE])
    else:
        put_text(text, i, wlen / 2, zmin[b] + weave - 0.55, 0.75, FONT_SANS, [WHITE])
    n_signs += 1

# every named shop its own name on the sign band over its shop front; Spielhallen and betting shops in gold on dark,
# the shop windows behind frosted foil with the word on it
GOLD = flat_mat("NRW_SignGold", (1.0, 0.72, 0.12))
FOIL = flat_mat("NRW_SpielhalleFoil", (0.62, 0.58, 0.52))
TEXT_ON = {0: WHITE, 1: WHITE, 2: WHITE, 3: WHITE, 4: DARK, 5: DARK}
n_named = 0
for i, lst in poi_on_face.items():
    b, wlen, wbot = info[i][1], info[i][5], info[i][6]
    lst = sorted(lst)
    colour = int(face_scol.get(i, sign_col[b]))
    for j, (u, nm, gamble) in enumerate(lst):
        # its stretch of the wall: halfway to the neighbours' points, and the text fitted into it
        lo = (lst[j - 1][0] + u) / 2 if j else 0.0
        hi = (lst[j + 1][0] + u) / 2 if j + 1 < len(lst) else wlen
        span = hi - lo - 0.4
        text = nm.upper() if gamble else nm
        size = min(0.42, 0.9 * span / max(len(text) * 0.62, 1.0))
        if size < 0.14:
            continue
        half = len(text) * 0.31 * size
        uc = float(np.clip(u, lo + 0.2 + half, hi - 0.2 - half)) if hi - lo > 2 * half + 0.4 else (lo + hi) / 2
        put_text(text, i, uc, zmin[b] + wbot + 3.325, size, FONT_BOLD, [GOLD if gamble else TEXT_ON[colour]], out=0.08)
        if gamble:
            put_panel(f"Foil_{b}_{i}_{j}", i, (lo + hi) / 2, zmin[b] + wbot + 1.6, max(hi - lo - 0.7, 0.8), 2.3, FOIL, out=0.03)
            word = "SPORTWETTEN" if gamble == "bet" else "SPIELHALLE"
            if text != word:
                put_text(word, i, (lo + hi) / 2, zmin[b] + wbot + 2.5, min(0.3, 0.9 * span / 7.0), FONT_BOLD, [GOLD], out=0.09)
        n_named += 1
print("SHOP_SIGNS", n_named, "of", len(shop_pois), "named shops,", sum(1 for p in shop_pois if p[3]), "gambling")


def face_of(b):
    """the open, street-facing facade of a building for its lettering"""
    own = [i for i, x in enumerate(info) if x[1] == b and x[5] >= 4]
    return max([i for i in own if unblocked(i)] or own, key=lambda i: info[i][3])
if not CHUNK:
    i = face_of(JOBCENTER)
    put_text("jobcenter", i, info[i][5] / 2, zmin[JOBCENTER] + info[i][7] - 1.3, 1.4, FONT_BOLD, [RED, DARK],
             char_mats=[0, 0, 0, 1, 1, 1, 1, 1, 1])
    shop_faces = [j for j, x in enumerate(info) if x[1] == BANK and x[0][0][l_shop] > 0.5]
    i = widest(shop_faces) if shop_faces else face_of(BANK)
    zb = zmin[BANK] + info[i][6] + 3.325
    put_text("S", i, 0.8, zb - 0.03, 0.62, FONT_BOLD, [WHITE])
    dot = bpy.data.objects.new("Sign_S_dot", bpy.data.meshes.new("Sign_S_dot"))
    bmd = bmesh.new()
    bmesh.ops.create_cone(bmd, cap_ends=True, segments=16, radius1=0.07, radius2=0.07, depth=0.03)
    bmd.to_mesh(dot.data); bmd.free()
    dot.data.materials.append(WHITE)
    p_dot, n_dot = facade_point(i, 0.8, zb + 0.34, 0.075)
    dot.location, dot.rotation_euler = p_dot, (math.pi / 2, 0, math.atan2(n_dot.x, -n_dot.y))
    signs.objects.link(dot)
    put_text("Sparkasse", i, 1.25, zb, 0.5, FONT_BOLD, [WHITE], align="LEFT")
    i = face_of(OLD_TOWNHALL)
    put_text("RATHAUS", i, door_at.get(i, info[i][5] / 2), zmin[OLD_TOWNHALL] + info[i][6] + 2.75, 0.55, FONT_SERIF, [DARK])
    n_signs += 3
for nte in notes:
    print("ROLE", nte)
print("DETAILS doors", n_doors, "garage_doors", n_garages, "shop_facades", n_shops, "sills", len(sill_f) // 6,
      "kinds", np.bincount(kind2.astype(int), minlength=7).tolist())
