"""Run from build_blend.py (exec) after rewe.py: buildings the user asked for, from Google Street View and place photos
(September 2023 / 2026) as modelling reference only — no photo goes into a texture.
  home     Tannhäuserstraße 31, Wiehl: 1960s block of flats, warm light render, brown plinth, stacked balconies with
           brown board parapets and terracotta side walls
  station  Gummersbach-Dieringhausen station: red masonry, canopy over the front, clock high on the facade
  pavilion Dieringhauser Str. 1: one-storey shop pavilion, "PIZZA AMORE" and "Spielhalle" over the shop windows
  action   the Action store by the station, with its sign and a pylon at the road
  forum    Forum Gummersbach: dark brick, glass entrance under a lit canopy, flower logo and lettering
Each facade is picked from the LoD2 walls; colours and styles go into face attributes (gcol, fstyle) that
export_region.py turns into facade-shader data, the extra geometry into the NRW_Landmarks collection.
Uses globals: tm, ground, MeshBuilder, box, simple_material, signs, FONT_BOLD, FONT_SANS."""
L_OUT = 0.04
lm_col = bpy.data.collections.new("NRW_Landmarks")
bpy.data.collections["NRW"].children.link(lm_col)

me_l = bpy.data.objects["NRW_Buildings"].data
nf = len(me_l.polygons)
l_bid = np.empty(nf, np.int32); me_l.attributes["bid"].data.foreach_get("value", l_bid)
l_ctr = np.empty(nf * 3, np.float32); me_l.polygons.foreach_get("center", l_ctr); l_ctr = l_ctr.reshape(-1, 3)
l_nrm = np.empty(nf * 3, np.float32); me_l.polygons.foreach_get("normal", l_nrm); l_nrm = l_nrm.reshape(-1, 3)
l_area = np.empty(nf, np.float32); me_l.polygons.foreach_get("area", l_area)
l_mat = np.empty(nf, np.int32); me_l.polygons.foreach_get("material_index", l_mat)
l_vco = np.empty(len(me_l.vertices) * 3, np.float32); me_l.vertices.foreach_get("co", l_vco); l_vco = l_vco.reshape(-1, 3)
l_gcol = np.empty(nf * 4, np.float32); me_l.attributes["gcol"].data.foreach_get("color", l_gcol); l_gcol = l_gcol.reshape(-1, 4)
l_kind = np.empty(nf * 3, np.float32); me_l.attributes["w_kind"].data.foreach_get("vector", l_kind); l_kind = l_kind.reshape(-1, 3)
l_look = np.empty(nf * 3, np.float32); me_l.attributes["w_look"].data.foreach_get("vector", l_look); l_look = l_look.reshape(-1, 3)
l_style = np.full(nf, -1.0, np.float32)
walls_l = (l_mat == 1) & (np.abs(l_nrm[:, 2]) < 0.3)


def building_at_ll(lat, lon):
    """the LoD2 building under a lat/lon (the one whose wall faces surround it, else the nearest)"""
    p = tm(np.array([lat]), np.array([lon]))[0]
    near = np.unique(l_bid[walls_l & (np.hypot(*(l_ctr[:, :2] - p).T) < 90)])
    best, best_d = None, 1e9
    for b in near:
        f = walls_l & (l_bid == b)
        xy = l_ctr[f, :2]
        inside = (xy[:, 0].min() <= p[0] <= xy[:, 0].max()) and (xy[:, 1].min() <= p[1] <= xy[:, 1].max())
        d = float(np.hypot(*(xy.mean(0) - p))) - (1000 if inside else 0)
        if d < best_d:
            best, best_d = int(b), d
    return best, p


def facade(b, target, min_len=4.0):
    """the wall of building b facing a point: unit normal, tangent, u range, bottom and top z, face ids"""
    faces = np.nonzero(walls_l & (l_bid == b) & (l_area > 3.0))[0]
    best, score = None, -1e9
    for i in faces:
        n = l_nrm[i, :2] / max(np.linalg.norm(l_nrm[i, :2]), 1e-6)
        to = target - l_ctr[i, :2]
        s = float(n @ (to / max(np.linalg.norm(to), 1e-6))) * 50 - np.linalg.norm(to) * 0.1 + l_area[i] * 0.02
        if s > score:
            best, score = i, s
    n = l_nrm[best, :2] / np.linalg.norm(l_nrm[best, :2])
    t = np.array((-n[1], n[0]))
    d = float(l_ctr[best, :2] @ n)
    same = [i for i in faces if l_nrm[i, :2] @ n > 0.98 and abs(l_ctr[i, :2] @ n - d) < 0.3]
    vs = np.unique(np.concatenate([list(me_l.polygons[i].vertices) for i in same]))
    us = l_vco[vs, :2] @ t
    return dict(n=n, t=t, d=d, u0=float(us.min()), u1=float(us.max()), z0=float(l_vco[vs, 2].min()),
                z1=float(l_vco[vs, 2].max()), faces=same)


def fp(f, u, w, z):
    """point on facade f: u along it, w out of the wall, z absolute"""
    xy = f["t"] * u + f["n"] * (f["d"] + w)
    return np.array((xy[0], xy[1], z))


def fbox(mb, f, u0, u1, w0, w1, z0, z1, mat):
    t3, n3 = np.array((*f["t"], 0.0)), np.array((*f["n"], 0.0))
    box(mb, fp(f, (u0 + u1) / 2, (w0 + w1) / 2, (z0 + z1) / 2), t3 * (u1 - u0) / 2, n3 * (w1 - w0) / 2,
        np.array((0, 0, (z1 - z0) / 2)), mat)


def text_on(f, text, u, z, size, font, mat, out=0.06, align="CENTER"):
    cu = bpy.data.curves.new("Sign_" + text, "FONT")
    cu.body, cu.size, cu.extrude = text, size, 0.03
    cu.align_x, cu.align_y = align, "CENTER"
    cu.font = font
    cu.materials.append(mat)
    ob = bpy.data.objects.new("Sign_" + text, cu)
    ob.location = fp(f, u, out, z)
    ob.rotation_euler = (math.pi / 2, 0, math.atan2(f["n"][0], -f["n"][1]))
    signs.objects.link(ob)
    return ob


def paint(b, rgb=None, style=None, kind=None, shop_faces=(), sign_colour=None):
    f = l_bid == b
    if rgb is not None:
        l_gcol[f] = (*rgb, 1.0)
        l_look[f, 1] = 0.0
    if style is not None:
        l_style[f] = style
    if kind is not None:
        l_kind[f, 0] = kind
    for i in shop_faces:
        l_kind[i, 1] = 1.0
    if sign_colour is not None and "scol" in me_l.attributes:
        sc_ = np.empty(nf, np.float32); me_l.attributes["scol"].data.foreach_get("value", sc_)
        sc_[list(shop_faces)] = sign_colour
        me_l.attributes["scol"].data.foreach_set("value", sc_)


def finish(mb, name, mats):
    ob = mb.build(name, "UVMap", mats)
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    lm_col.objects.link(ob)


done = []
M_BROWN = simple_material("NRW_LmBrown", (0.16, 0.07, 0.03), 0.8)
M_TERRA = simple_material("NRW_LmTerracotta", (0.36, 0.13, 0.05), 0.8)
M_SLAB = simple_material("NRW_LmConcrete", (0.45, 0.44, 0.42), 0.9)
M_METAL = simple_material("NRW_LmMetal", (0.09, 0.095, 0.1), 0.4, 0.8)
M_WHITE = simple_material("NRW_LmWhite", (0.8, 0.8, 0.78), 0.5)
M_GLASS = simple_material("NRW_LmGlass", (0.03, 0.045, 0.055), 0.05, 0.3)
M_WARM = simple_material("NRW_LmWarmLight", (1.0, 0.8, 0.5), 0.5)
M_WARM.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (1.0, 0.75, 0.45, 1)
M_WARM.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 3.0
SIGN_RED, SIGN_WHITE = bpy.data.materials["NRW_SignRed"], bpy.data.materials["NRW_SignWhite"]
M_ACTION = simple_material("NRW_LmActionBlue", (0.0, 0.08, 0.45), 0.4)
PETALS = [simple_material(f"NRW_LmPetal{i}", c, 0.4) for i, c in
          enumerate([(0.45, 0.62, 0.2), (0.85, 0.72, 0.25), (0.8, 0.45, 0.5), (0.55, 0.45, 0.75), (0.7, 0.8, 0.35), (0.9, 0.6, 0.55)])]

# ---- home: Tannhäuserstraße 31 ----
b, p = building_at_ll(50.9580615, 7.5429641)
if b is not None:
    paint(b, rgb=(0.6, 0.57, 0.5), style=0.0, kind=0.0)
    mb = MeshBuilder()
    walls_b = np.nonzero(walls_l & (l_bid == b) & (l_area > 1.0))[0]
    zb = l_vco[np.unique(np.concatenate([list(me_l.polygons[i].vertices) for i in walls_b])), 2]
    base, eave = float(zb.min()), float(np.percentile(zb, 75))
    # brown plinth along every wall, a hand's breadth proud of the render
    for i in walls_b:
        f = {"n": l_nrm[i, :2] / np.linalg.norm(l_nrm[i, :2])}
        f["t"] = np.array((-f["n"][1], f["n"][0]))
        f["d"] = float(l_ctr[i, :2] @ f["n"])
        vs = list(me_l.polygons[i].vertices)
        us = l_vco[vs, :2] @ f["t"]
        g0 = float(ground(fp(f, float(us.mean()), 0, 0)[None, :2])[0])
        fbox(mb, f, float(us.min()), float(us.max()), -0.05, 0.05, g0 - 0.3, g0 + 0.9, 0)
    # balconies: two stacks on each long side, one per storey above the ground floor
    long_sides = sorted(walls_b, key=lambda i: -l_area[i])[:2]
    for i in long_sides:
        f = facade(b, l_ctr[i, :2] + l_nrm[i, :2] * 5)
        L = f["u1"] - f["u0"]
        g0 = float(ground(fp(f, (f["u0"] + f["u1"]) / 2, 0, 0)[None, :2])[0])
        storeys = max(1, int((eave - g0 - 0.3) / 2.8))
        for uc in (f["u0"] + 0.25 * L, f["u0"] + 0.75 * L):
            for k in range(storeys):
                z = g0 + 1.5 + k * 2.8  # raised ground floor, then a storey each
                if z > eave - 1.2:
                    break
                fbox(mb, f, uc - 1.4, uc + 1.4, 0, 1.4, z - 0.15, z, 2)                  # slab
                fbox(mb, f, uc - 1.4, uc + 1.4, 1.35, 1.42, z, z + 1.0, 0)               # board parapet
                for s in (-1, 1):
                    fbox(mb, f, uc + s * 1.4 - 0.06, uc + s * 1.4 + 0.06, 0, 1.4, z - 0.15, z + 2.6, 1)  # side walls
    finish(mb, "NRW_Lm_Home", [M_BROWN, M_TERRA, M_SLAB])
    done.append(("home", b))

# ---- Dieringhausen station ----
b, p = building_at_ll(50.98592, 7.5439)
road = tm(np.array([50.98530]), np.array([7.54330]))[0]  # Dieringhauser Straße in front
if b is not None:
    paint(b, style=3.0, kind=8.0)
    f = facade(b, road)
    mb = MeshBuilder()
    L = f["u1"] - f["u0"]
    g0 = float(ground(fp(f, (f["u0"] + f["u1"]) / 2, 0, 0)[None, :2])[0])
    fbox(mb, f, f["u0"] + 0.15 * L, f["u1"] - 0.15 * L, 0, 1.8, g0 + 3.4, g0 + 3.6, 0)          # canopy
    for s in (0.2, 0.5, 0.8):
        fbox(mb, f, f["u0"] + s * L - 0.05, f["u0"] + s * L + 0.05, 1.7, 1.78, g0, g0 + 3.4, 0)  # its posts
    # clock high on the front: white face, dark ring and hands
    zc = min(f["z1"] - 1.2, g0 + 8.5)
    uc = (f["u0"] + f["u1"]) / 2
    ring = [fp(f, uc + 0.75 * math.cos(a), 0.08, zc + 0.75 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 25)[:-1]]
    face = [fp(f, uc + 0.65 * math.cos(a), 0.1, zc + 0.65 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 25)[:-1]]
    mb.face(ring, 0)  # counter-clockwise as seen from the street: faces out
    mb.face(face, 1)
    fbox(mb, f, uc - 0.03, uc + 0.03, 0.1, 0.13, zc, zc + 0.5, 0)
    fbox(mb, f, uc, uc + 0.38, 0.1, 0.13, zc - 0.03, zc + 0.03, 0)
    finish(mb, "NRW_Lm_Station", [M_METAL, M_WHITE])
    done.append(("station", b))

# ---- the pavilion: PIZZA AMORE + Spielhalle ----
b, p = building_at_ll(50.984866, 7.5427445)
if b is not None:
    f = facade(b, road)
    paint(b, rgb=(0.7, 0.7, 0.68), style=0.0, kind=1.0, shop_faces=f["faces"], sign_colour=5.0)
    L = f["u1"] - f["u0"]
    g0 = float(ground(fp(f, (f["u0"] + f["u1"]) / 2, 0, 0)[None, :2])[0])
    rel = g0 + 3.32  # the sign band of the shader's shop front
    # read along the facade from the viewer: +u is to their right
    text_on(f, "PIZZA AMORE", f["u0"] + 0.28 * L, rel, 0.42, FONT_BOLD, SIGN_RED)
    # the Spielhalle half like every other one (building_details GOLD / FOIL): dark sign, gold letters, frosted windows
    text_on(f, "SPIELHALLE", f["u0"] + 0.74 * L, rel, 0.45, FONT_BOLD, GOLD, out=0.2)
    mb = MeshBuilder()
    fbox(mb, f, f["u0"] - 0.3, f["u1"] + 0.3, 0, 1.2, f["z1"] - 0.15, f["z1"], 0)  # the roof's overhang
    fbox(mb, f, f["u0"] + 0.52 * L, f["u1"] - 0.2, 0.05, 0.15, rel - 0.3, rel + 0.3, 0)    # dark sign panel
    fbox(mb, f, f["u0"] + 0.52 * L, f["u1"] - 0.4, 0.01, 0.04, g0 + 0.4, g0 + 2.8, 1)     # foil over the glass
    finish(mb, "NRW_Lm_Pavilion", [M_METAL, FOIL])
    done.append(("pavilion", b))

# ---- Action ----
b, p = building_at_ll(50.98458, 7.54231)
if b is not None:
    f = facade(b, road)
    paint(b, rgb=(0.72, 0.71, 0.66), style=0.0, kind=7.0)
    g0 = float(ground(fp(f, (f["u0"] + f["u1"]) / 2, 0, 0)[None, :2])[0])
    uc = (f["u0"] + f["u1"]) / 2
    mb = MeshBuilder()
    fbox(mb, f, uc - 2.6, uc + 2.6, 0, 0.08, g0 + 3.0, g0 + 4.2, 1)                       # white sign panel
    fbox(mb, f, f["u0"] + 1, f["u0"] + 1.5, 9, 9.5, g0 - 0.3, g0 + 5.0, 0)                 # pylon by the road
    fbox(mb, f, f["u0"] + 0.6, f["u0"] + 1.9, 9.02, 9.48, g0 + 3.4, g0 + 4.9, 1)
    finish(mb, "NRW_Lm_Action", [M_METAL, M_WHITE])
    text_on(f, "ACTION", uc, g0 + 3.6, 0.8, FONT_BOLD, M_ACTION, out=0.12)
    done.append(("action", b))

# ---- Forum Gummersbach ----
b, p = building_at_ll(51.02400, 7.56560)
bus = tm(np.array([51.02305]), np.array([7.56652]))[0]  # the bus station south of it
if b is not None:
    paint(b, style=3.0, kind=1.0)
    f = facade(b, bus, 20.0)
    g0 = float(ground(fp(f, (f["u0"] + f["u1"]) / 2, 0, 0)[None, :2])[0])
    uc = (f["u0"] + f["u1"]) / 2
    mb = MeshBuilder()
    fbox(mb, f, uc - 7, uc + 7, -0.1, 0.1, g0 - 0.2, g0 + 5.5, 1)           # glass entrance front
    for k in range(8):
        u = uc - 7 + k * 2.0
        fbox(mb, f, u - 0.06, u + 0.06, 0.1, 0.18, g0, g0 + 5.5, 0)          # mullions
    fbox(mb, f, uc - 8, uc + 8, 0, 3.5, g0 + 5.5, g0 + 6.1, 0)               # canopy
    fbox(mb, f, uc - 7.6, uc + 7.6, 0.1, 3.3, g0 + 5.47, g0 + 5.5, 2)        # its lit soffit
    # the flower logo: six pastel hexagons round a centre, above the lettering
    lz, lu = g0 + 12.0, uc
    for k in range(6):
        a = k * math.pi / 3
        cu_, cz = lu + 1.25 * math.cos(a), lz + 1.25 * math.sin(a)
        hexa = [fp(f, cu_ + 0.7 * math.cos(a2), 0.12, cz + 0.7 * math.sin(a2)) for a2 in np.linspace(0, 2 * math.pi, 7)[:-1]]
        mb.face(hexa, 3 + k)
    finish(mb, "NRW_Lm_Forum", [M_METAL, M_GLASS, M_WARM] + PETALS)
    text_on(f, "forum", uc, g0 + 9.2, 2.0, FONT_SANS, SIGN_WHITE, out=0.1)
    text_on(f, "gummersbach", uc, g0 + 7.8, 0.9, FONT_SANS, SIGN_WHITE, out=0.1)
    done.append(("forum", b))

# ---- from the architecture analysis (September 2026): ALDI Weiherplatz, Kaufland Dieringhausen, ZOB Gummersbach ----
def hexlin(h):
    c = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0
    return tuple(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


def smallest_at_ll(lat, lon, max_h):
    """the LoD2 building no taller than max_h with the smallest footprint box around a lat/lon: picks a canopy out
    of the block it fronts"""
    p = tm(np.array([lat]), np.array([lon]))[0]
    best, best_a = None, 1e18
    for b in np.unique(l_bid[np.hypot(*(l_ctr[:, :2] - p).T) < 150]):
        vs = np.unique(np.concatenate([list(me_l.polygons[i].vertices) for i in np.nonzero(l_bid == b)[0]]))
        lo, hi = l_vco[vs, :2].min(0), l_vco[vs, :2].max(0)
        a = float(np.prod(hi - lo))
        if l_vco[vs, 2].max() - l_vco[vs, 2].min() > max_h:
            continue
        if (lo <= p).all() and (p <= hi).all() and a < best_a:
            best, best_a = int(b), a
    return best


def open_canopy(mb, b, post_mat, soffit_mat):
    """an ALKIS canopy comes as a closed box: its walls go (at the end of this file), the roof stands on posts"""
    roofs = np.nonzero((l_bid == b) & (l_mat != 1) & (l_nrm[:, 2] > 0.3))[0]
    for i in roofs:
        vs = list(me_l.polygons[i].vertices)
        mb.face([l_vco[v] - (0, 0, 0.05) for v in vs[::-1]], soffit_mat)   # underside
    vs = np.unique(np.concatenate([list(me_l.polygons[i].vertices) for i in roofs])) if len(roofs) else []
    seen = set()
    for v in vs:
        k = tuple(np.round(l_vco[v, :2], 0))
        if k in seen:
            continue
        seen.add(k)
        g = float(ground(l_vco[v, :2][None])[0])
        top = float(l_vco[v, 2]) - 0.05
        if top - g > 1.5:
            box(mb, np.array((*l_vco[v, :2], (g + top) / 2)), np.array((0.08, 0, 0)), np.array((0, 0.08, 0)),
                np.array((0, 0, (top - g) / 2)), post_mat)


drop_b, drop_walls_b = [], []
M_DARK = simple_material("NRW_LmDarkBand", hexlin("555652"), 0.5, 0.3)
M_ALDI = simple_material("NRW_LmAldiBlue", hexlin("0050AA"), 0.4)
M_ORANGE = simple_material("NRW_LmAldiOrange", hexlin("F39200"), 0.4)
M_GOLD = simple_material("NRW_LmZobGold", hexlin("C09B63"), 0.35, 0.7)
M_GOLD_IN = simple_material("NRW_LmZobSoffit", hexlin("C9A86B"), 0.5, 0.3)
M_COLUMN = simple_material("NRW_LmZobColumn", hexlin("30302F"), 0.5, 0.3)
M_GLASS2 = simple_material("NRW_LmShopGlass", hexlin("677576"), 0.05, 0.3)
M_VB_BLUE = simple_material("NRW_LmVolksbankBlue", hexlin("0066B3"), 0.4)
M_VB_ORANGE = simple_material("NRW_LmVolksbankOrange", hexlin("FF6600"), 0.4)

# ---- ALDI SÜD, Weiherplatz: one unit of the long retail block; entrance, dark band and the square sign face the car park
b, _ = building_at_ll(50.95099, 7.54778)
park = tm(np.array([50.95102]), np.array([7.54841]))[0]
if b is not None:
    paint(b, rgb=hexlin("D8D8D3"))
    f = facade(b, park)
    g0 = float(ground(fp(f, (f["u0"] + f["u1"]) / 2, 0, 0)[None, :2])[0])
    ua = float(np.clip(tm(np.array([50.95099]), np.array([7.54778]))[0] @ f["t"], f["u0"] + 9, f["u1"] - 9))
    top = min(f["z1"], g0 + 6.0)
    mb = MeshBuilder()
    fbox(mb, f, ua - 15, ua + 15, 0, 0.12, top - 0.9, top, 0)                        # dark band along the unit
    fbox(mb, f, ua - 3, ua + 3, 0, 0.1, g0 - 0.2, g0 + 3.0, 1)                        # glass entrance
    fbox(mb, f, ua - 3.2, ua + 3.2, 0.1, 0.18, g0 + 3.0, g0 + 3.2, 0)                 # its dark head
    fbox(mb, f, ua - 1.0, ua + 1.0, 0.12, 0.22, top - 0.3, top + 1.7, 2)              # the 2 x 2 m sign, high on the band
    for k, z in enumerate((top + 0.05, top + 0.2, top + 0.35)):
        fbox(mb, f, ua - 0.85, ua + 0.85, 0.22, 0.24, z, z + 0.08, 3 if k != 1 else 4)  # its stripes
    c = smallest_at_ll(50.95130, 7.54784, 8.0)                                            # the long canopy in front
    if c is not None and c != b:
        open_canopy(mb, c, 0, 0)
        drop_walls_b.append(c)
    finish(mb, "NRW_Lm_Aldi", [M_DARK, M_GLASS2, M_ALDI, M_ORANGE, SIGN_RED])
    text_on(f, "ALDI", ua, top + 1.1, 0.62, FONT_BOLD, SIGN_WHITE, out=0.25)
    text_on(f, "SÜD", ua, top + 0.6, 0.3, FONT_BOLD, SIGN_WHITE, out=0.25)
    done.append(("aldi", b))

# ---- Kaufland, Dieringhauser Straße 16: a very wide low box, glass portal and red sign towards the car park ----
b, _ = building_at_ll(50.98313, 7.53025)
park = tm(np.array([50.98327]), np.array([7.52917]))[0]
if b is not None:
    paint(b, rgb=hexlin("D8D8D3"), kind=7.0)
    f = facade(b, park, 20.0)
    g0 = float(ground(fp(f, (f["u0"] + f["u1"]) / 2, 0, 0)[None, :2])[0])
    uc = (f["u0"] + f["u1"]) / 2
    top = f["z1"]
    mb = MeshBuilder()
    fbox(mb, f, f["u0"], f["u1"], 0, 0.12, top - 1.0, top, 0)                        # dark metal band
    fbox(mb, f, f["u0"], f["u1"], 0, 0.06, g0 - 0.2, g0 + 0.8, 3)                     # grey base
    fbox(mb, f, uc - 5, uc + 5, 0, 0.1, g0 - 0.2, g0 + 3.6, 1)                        # glass portal
    for k in range(6):
        fbox(mb, f, uc - 5 + 2 * k - 0.05, uc - 5 + 2 * k + 0.05, 0.1, 0.16, g0, g0 + 3.6, 0)
    fbox(mb, f, uc - 6, uc + 6, 0, 2.5, g0 + 3.6, g0 + 4.0, 0)                        # portal canopy
    fbox(mb, f, uc - 3.8, uc + 3.8, 0.12, 0.22, top - 1.9, top - 0.1, 2)              # red sign panel
    # (2026-09-27, "more like the real one") the red logo square beside the lettering, a red edge on the portal
    # canopy, a lettered pylon by the road and trolley shelters in the car park
    fbox(mb, f, uc + 4.1, uc + 6.3, 0.12, 0.22, top - 2.1, top + 0.1, 2)
    fbox(mb, f, uc - 6, uc + 6, 2.45, 2.6, g0 + 3.55, g0 + 4.05, 2)
    rp = np.asarray(bdir[b], float)
    u_dir = (park - rp) / max(np.linalg.norm(park - rp), 1e-6)
    pc = rp + u_dir * 7.0
    n_p, t_p = -u_dir, np.array((u_dir[1], -u_dir[0]))
    gp = float(ground(pc[None])[0])
    box(mb, np.array((*pc, gp + 4.5)), np.array((*t_p, 0.0)) * 1.3, np.array((*n_p, 0.0)) * 0.25, np.array((0, 0, 4.5)), 2)
    box(mb, np.array((*pc, gp + 3.0)), np.array((*t_p, 0.0)) * 1.2, np.array((*n_p, 0.0)) * 0.27, np.array((0, 0, 1.6)), 4)
    for side in (1.0, -1.0):
        fpyl = dict(n=n_p * side, t=t_p * side, d=float(pc @ (n_p * side)) + 0.27)
        text_on(fpyl, "Kaufland", float(pc @ (t_p * side)), gp + 7.7, 0.5, FONT_BOLD, SIGN_WHITE, out=0.03)
        text_on(fpyl, "K", float(pc @ (t_p * side)), gp + 6.2, 1.1, FONT_BOLD, SIGN_WHITE, out=0.03)
    for s_u in (-14.0, 14.0):
        sp = park + f["t"] * s_u
        gs = float(ground(sp[None])[0])
        t3, n3 = np.array((*f["t"], 0.0)), np.array((*f["n"], 0.0))
        for du, dn in ((-3, -1.1), (3, -1.1), (-3, 1.1), (3, 1.1)):
            box(mb, np.array((*(sp + f["t"] * du + f["n"] * dn), gs + 1.15)), t3 * 0.04, n3 * 0.04, np.array((0, 0, 1.15)), 5)
        box(mb, np.array((*sp, gs + 2.35)), t3 * 3.2, n3 * 1.3, np.array((0, 0, 0.06)), 3)
    finish(mb, "NRW_Lm_Kaufland", [M_DARK, M_GLASS2, SIGN_RED, M_SLAB, M_WHITE, M_METAL])
    text_on(f, "Kaufland", uc, top - 1.0, 1.0, FONT_BOLD, SIGN_WHITE, out=0.26)
    text_on(f, "K", uc + 5.2, top - 1.0, 1.7, FONT_BOLD, SIGN_WHITE, out=0.26)
    done.append(("kaufland", b))

# ---- Volksbank Oberberg, Bahnhofstraße 3 by the Rathaus: an office block with window ribbons and the blue lettering
b, p = building_at_ll(50.94827, 7.54863)
if b is not None:
    paint(b, rgb=hexlin("E4E3DE"), style=0.0, kind=6.0)
    f = facade(b, nearest_road(p[None])[0])
    g0 = float(ground(fp(f, (f["u0"] + f["u1"]) / 2, 0, 0)[None, :2])[0])
    uc = (f["u0"] + f["u1"]) / 2
    top = f["z1"]
    mb = MeshBuilder()
    fbox(mb, f, f["u0"], f["u1"], 0, 0.15, top - 0.6, top, 0)                  # dark attic band
    fbox(mb, f, uc - 3, uc + 3, 0, 0.1, g0 - 0.2, g0 + 3.0, 1)                  # glass entrance
    fbox(mb, f, uc - 3.5, uc + 3.5, 0, 1.8, g0 + 3.0, g0 + 3.3, 0)              # its canopy
    fbox(mb, f, uc - 4.6, uc - 3.4, 0.15, 0.25, top - 1.9, top - 0.7, 3)        # the logo square
    finish(mb, "NRW_Lm_Volksbank", [M_DARK, M_GLASS2, M_VB_BLUE, M_VB_ORANGE])
    text_on(f, "Volksbank", uc + 1.0, top - 1.3, 1.1, FONT_BOLD, M_VB_BLUE, out=0.2)
    done.append(("volksbank", b))

# ---- ZOB Gummersbach: the folded gold canopy (~130 m) on tilted black columns, in place of LoD2's closed box ----
c = smallest_at_ll(51.02311, 7.56640, 12.0)
if c is not None:
    fi = np.nonzero(l_bid == c)[0]
    vs = np.unique(np.concatenate([list(me_l.polygons[i].vertices) for i in fi]))
    xy = l_vco[vs, :2]
    ctr = xy.mean(0)
    ax_ = np.linalg.svd(xy - ctr, full_matrices=False)[2][0]
    side = np.array((-ax_[1], ax_[0]))
    s, w = (xy - ctr) @ ax_, (xy - ctr) @ side
    N = 13
    edges = np.linspace(s.min() + 2, s.max() - 2, N + 1)
    L, R, C = [], [], []
    for k, sk in enumerate(edges):
        near = np.abs(s - sk) < (edges[1] - edges[0])
        mid = float(np.median(w[near])) if near.any() else 0.0
        half = float(np.clip((w[near].max() - w[near].min()) / 2 if near.any() else 8, 6, 10))
        cp = ctr + ax_ * sk + side * mid
        gz = float(ground(cp[None])[0])
        up, dn = gz + 7.4, gz + 5.9
        zl, zr = (up, dn) if k % 2 == 0 else (dn, up)
        L.append(np.array((*(cp + side * half), zl)))
        R.append(np.array((*(cp - side * half), zr)))
        C.append((cp, gz, half))
    mb = MeshBuilder()
    TH = np.array((0, 0, 0.4))

    def tri(a, b_, c_, mat, want_up):
        n = np.cross(b_ - a, c_ - a)
        mb.face([a, b_, c_] if (n[2] > 0) == want_up else [a, c_, b_], mat)

    def quad_out(p, out_dir, mat):
        n = np.cross(p[1] - p[0], p[3] - p[0])
        mb.face(p if n[:2] @ out_dir >= 0 else p[::-1], mat)

    for k in range(N):
        for a, b_, c_ in ((L[k], R[k], R[k + 1]), (L[k], R[k + 1], L[k + 1])):
            tri(a, b_, c_, 0, True)
            tri(a - TH, b_ - TH, c_ - TH, 1, False)
        for P_, sgn in ((L, 1), (R, -1)):
            quad_out([P_[k], P_[k + 1], P_[k + 1] - TH, P_[k] - TH], side * sgn, 0)
    for P2, sgn in (((L[0], R[0]), -1), ((L[N], R[N]), 1)):
        quad_out([P2[0], P2[1], P2[1] - TH, P2[0] - TH], ax_ * sgn, 0)
    # columns in V pairs: from near the kerb line out to under both edges
    for k in range(1, N, 2):
        cp, gz, half = C[k]
        for sgn, P_ in ((1, L), (-1, R)):
            base = np.array((*(cp + side * sgn * 1.2), gz - 0.3))
            topp = np.array((*(cp + side * sgn * (half - 2.5)), P_[k][2] - 0.4 - 0.3))
            box(mb, (base + topp) / 2, np.array((*ax_ * 0.3, 0)), np.array((*side * 0.3, 0)), (topp - base) / 2, 2)
    finish(mb, "NRW_Lm_ZOB", [M_GOLD, M_GOLD_IN, M_COLUMN])
    drop_b.append(c)
    done.append(("zob", c))

me_l.attributes["gcol"].data.foreach_set("color", l_gcol.ravel())
me_l.attributes["w_kind"].data.foreach_set("vector", l_kind.ravel())
me_l.attributes["w_look"].data.foreach_set("vector", l_look.ravel())
a = me_l.attributes.get("fstyle") or me_l.attributes.new("fstyle", "FLOAT", "FACE")
a.data.foreach_set("value", l_style)
if drop_b or drop_walls_b:
    import bmesh
    bm_l = bmesh.new()
    bm_l.from_mesh(me_l)
    lay = bm_l.faces.layers.int["bid"]
    dead = [f_ for f_ in bm_l.faces if f_[lay] in drop_b or (f_[lay] in drop_walls_b and abs(f_.normal.z) < 0.3)]
    bmesh.ops.delete(bm_l, geom=dead, context="FACES")
    bm_l.to_mesh(me_l)
    bm_l.free()
    print("LANDMARKS dropped", drop_b, "walls of", drop_walls_b, len(dead), "faces")
print("LANDMARKS", done)
