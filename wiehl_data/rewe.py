"""Run from build_blend.py (exec) after busbf.py: Rewe (building REWE from building_details), reworked from street photos.
The generic store rule made every facade towards the station glass. The photos show two shop fronts instead —
the west one on Bahnhofstraße with the café terrace and the east entrance by the car park — each glass on the
ground floor under a dark anthracite upper storey that sticks out ~1.8 m. The tall south-west part is a red
three-storey block with windows; the rest are plain white store walls. A glass stair tower stands at the
north-west corner.
Uses globals: REWE, offset, ground, box, MeshBuilder, simple_material, signs, FONT_BOLD, FONT_SANS."""
REWE_B = REWE
OVERHANG, GLASS_H, TOWER = 1.8, 3.6, 4.5

me_b = bpy.data.objects["NRW_Buildings"].data
n_f = len(me_b.polygons)


def face_attr(name, width, prop):
    a = np.empty(n_f * width, np.float32)
    me_b.attributes[name].data.foreach_get(prop, a)
    return a.reshape(n_f, width) if width > 1 else a


bid = np.empty(n_f, np.int32)
me_b.attributes["bid"].data.foreach_get("value", bid)
nrm = np.empty(n_f * 3, np.float32); me_b.polygons.foreach_get("normal", nrm); nrm = nrm.reshape(-1, 3)
ctr = np.empty(n_f * 3, np.float32); me_b.polygons.foreach_get("center", ctr); ctr = ctr.reshape(-1, 3)
vco = np.empty(len(me_b.vertices) * 3, np.float32); me_b.vertices.foreach_get("co", vco); vco = vco.reshape(-1, 3)
w_kind, w_look, gcol = face_attr("w_kind", 3, "vector"), face_attr("w_look", 3, "vector"), face_attr("gcol", 4, "color")

walls = [int(i) for i in np.nonzero((bid == REWE_B) & (np.abs(nrm[:, 2]) < 0.3))[0]]
fverts = {i: list(me_b.polygons[i].vertices) for i in walls}
ztop = {i: vco[fverts[i], 2].max() for i in walls}
# LoD2 draws the jutting upper storey of the west front's south half as a 1.5-2 m wedge standing on the ground
# (x < 6.9, y -97..-76); it goes, and the upper storey below is built out over the glass instead
wedge_top = 0.0
wedge = []
for i in np.nonzero((bid == REWE_B) & (ctr[:, 1] > -97.6) & (ctr[:, 1] < -75.9))[0]:
    vs = list(me_b.polygons[i].vertices)
    if vco[vs, 0].max() < 6.95 and not (nrm[i, 0] < -0.8 and vco[vs, 0].min() >= 5.35):
        wedge.append(int(i))
        wedge_top = max(wedge_top, float(vco[vs, 2].max()))
west = [i for i in walls if nrm[i, 0] < -0.8 and ctr[i, 0] < 16 and ctr[i, 1] > -100 and i not in wedge]
east = [i for i in walls if nrm[i, 0] > 0.8 and ctr[i, 0] > 64 and ctr[i, 1] > -86]
red = [i for i in walls if 8 < ctr[i, 0] < 50 and -129 < ctr[i, 1] < -109 and ztop[i] > 196.5 + offset]
for i in walls:
    w_kind[i] = (7, 0, 0)
for i in west + east:
    w_kind[i] = (4, 0, 0)
for i in red:
    w_kind[i] = (0, 0, 0)
    w_look[i, 1] = 0.0  # no brick
    gcol[i] = (0.42, 0.035, 0.03, 1.0)
me_b.attributes["w_kind"].data.foreach_set("vector", w_kind.ravel())
me_b.attributes["w_look"].data.foreach_set("vector", w_look.ravel())
me_b.attributes["gcol"].data.foreach_set("color", gcol.ravel())

mb = MeshBuilder()
CLAD, SOFFIT, TGLASS, FRAME, FASCIA, CLOTH, FURN, PLANTER, HEDGE = range(9)
SIGN_RED, SIGN_WHITE = bpy.data.materials["NRW_SignRed"], bpy.data.materials["NRW_SignWhite"]
mats = [simple_material("NRW_ReweClad", (0.03, 0.032, 0.035), 0.4, 0.5),
        simple_material("NRW_ReweSoffit", (0.1, 0.1, 0.11), 0.6),
        simple_material("NRW_ReweTowerGlass", (0.02, 0.04, 0.06), 0.05, 0.6),
        simple_material("NRW_ReweFrame", (0.05, 0.055, 0.06), 0.4, 0.7),
        simple_material("NRW_ReweFascia", (0.6, 0.61, 0.62), 0.5),
        simple_material("NRW_ReweUmbrella", (0.45, 0.02, 0.02), 0.85),
        simple_material("NRW_ReweFurniture", (0.06, 0.06, 0.065), 0.5, 0.6),
        simple_material("NRW_RewePlanter", (0.3, 0.3, 0.29), 0.9),
        simple_material("NRW_ReweHedge", (0.04, 0.12, 0.03), 0.9)]
Z3 = np.array((0.0, 0.0, 1.0))


class Front:
    """a shop front: facade plane from its wall faces; u runs along it, w points out of the building"""
    def __init__(self, ids):
        n = nrm[ids, :2].mean(0)
        self.n = n / np.linalg.norm(n)
        self.t = np.array((-self.n[1], self.n[0]))
        vs = sorted({v for i in ids for v in fverts[i]})
        us = vco[vs, :2] @ self.t
        self.u0, self.u1 = us.min(), us.max()
        self.d = float(np.median(ctr[ids, :2] @ self.n))
        self.top = max(ztop[i] for i in ids)
        self.g = float(ground(self.xy((self.u0 + self.u1) / 2, 0)[None])[0])

    def xy(self, u, w):
        return self.t * u + self.n * (self.d + w)

    def p(self, u, w, z):
        return np.array((*self.xy(u, w), z))

    def box(self, u0, u1, w0, w1, z0, z1, mat):
        t3, n3 = np.array((*self.t, 0)), np.array((*self.n, 0))
        box(mb, self.p((u0 + u1) / 2, (w0 + w1) / 2, (z0 + z1) / 2),
            t3 * (u1 - u0) / 2, n3 * (w1 - w0) / 2, Z3 * (z1 - z0) / 2, mat)

    def upper(self, u0, u1, top=None):
        """the anthracite upper storey, pushed out over the glass"""
        z0, z1 = self.g + GLASS_H, max(self.top, top or 0.0) + 0.5
        self.box(u0, u1, -0.3, OVERHANG, z0, z1, CLAD)
        self.box(u0 + 0.05, u1 - 0.05, -0.2, OVERHANG - 0.05, z0 - 0.03, z0, SOFFIT)
        return (z0 + z1) / 2

    def sign(self, text, u, z, size, font, mat, align="CENTER"):
        cu = bpy.data.curves.new("Sign_" + text, "FONT")
        cu.body, cu.size, cu.extrude = text, size, 0.04
        cu.align_x, cu.align_y = align, "CENTER"
        cu.font = font
        cu.materials.append(mat)
        ob = bpy.data.objects.new("Sign_" + text, cu)
        ob.location = self.p(u, OVERHANG + 0.05, z)
        ob.rotation_euler = (math.pi / 2, 0, math.atan2(self.n[0], -self.n[1]))
        signs.objects.link(ob)


def umbrella(f, u, w, g):
    c = f.p(u, w, g)
    rim = [c + np.array((1.5 * math.cos(a), 1.5 * math.sin(a), 2.25)) for a in np.linspace(0, 2 * math.pi, 9)[:-1]]
    apex = c + Z3 * 2.75
    for k in range(8):
        mb.face([rim[k], rim[(k + 1) % 8], apex], CLOTH)
        mb.face([rim[(k + 1) % 8], rim[k], apex - Z3 * 0.02], CLOTH)
    box(mb, c + Z3 * 1.37, np.array((0.03, 0, 0)), np.array((0, 0.03, 0)), Z3 * 1.4, FURN)
    box(mb, c + Z3 * 0.74, np.array((0.42, 0, 0)), np.array((0, 0.42, 0)), Z3 * 0.02, FURN)
    for side in (-1, 1):
        s = c + np.array((0, side * 0.75, 0))
        box(mb, s + Z3 * 0.45, np.array((0.22, 0, 0)), np.array((0, 0.22, 0)), Z3 * 0.02, FURN)
        box(mb, s + np.array((0, side * 0.21, 0.68)), np.array((0.22, 0, 0)), np.array((0, 0.015, 0)), Z3 * 0.22, FURN)
        for dx in (-0.18, 0.18):
            box(mb, s + np.array((dx, 0, 0.22)), np.array((0.015, 0, 0)), np.array((0, 0.015, 0)), Z3 * 0.23, FURN)


# ---- west front: glass stair tower at the north end, anthracite upper storey, sign, café terrace ----
wf = Front(west)
north_is_u0 = wf.p(wf.u0, 0, 0)[1] > wf.p(wf.u1, 0, 0)[1]
tu0, tu1 = (wf.u0 - 0.3, wf.u0 + TOWER) if north_is_u0 else (wf.u1 - TOWER, wf.u1 + 0.3)
tz0, tz1 = wf.g - 0.2, wf.top + 1.2
wf.box(tu0, tu1, -0.3, TOWER - 0.3, tz0, tz1, TGLASS)
for u in np.arange(tu0, tu1 + 0.01, (tu1 - tu0) / 3):
    wf.box(u - 0.04, u + 0.04, TOWER - 0.34, TOWER - 0.26, tz0, tz1, FRAME)
for w in np.linspace(-0.3, TOWER - 0.3, 4):
    for u in (tu0, tu1):
        wf.box(u - 0.04, u + 0.04, w - 0.04, w + 0.04, tz0, tz1, FRAME)
for z in np.arange(wf.g + 3.5, tz1 - 0.5, 3.5):
    wf.box(tu0 - 0.04, tu1 + 0.04, -0.3, TOWER - 0.26, z - 0.06, z + 0.06, FRAME)
wf.box(tu0 - 0.6, tu1 + 0.6, -0.3, TOWER + 0.3, tz1, tz1 + 0.25, FASCIA)

su0, su1 = (tu1, wf.u1 + 0.2) if north_is_u0 else (wf.u0 - 0.2, tu0)
zs = wf.upper(su0, su1, wedge_top)
mid = (su0 + su1) / 2
wf.sign("REVO", mid - 1.5, zs, 1.6, FONT_BOLD, SIGN_RED)
wf.sign("am besten", mid + 2.2, zs - 0.2, 0.55, FONT_SANS, SIGN_WHITE, align="LEFT")

step = 1 if north_is_u0 else -1
start = tu1 + 1.5 if north_is_u0 else tu0 - 1.5
for k in range(4):
    u = start + step * (2.0 + 4.0 * k)
    umbrella(wf, u, 5.0, float(ground(wf.xy(u, 5.0)[None])[0]))
pu0, pu1 = sorted((start, start + step * 16.5))
wf.box(pu0, pu1, 8.0, 9.0, wf.g - 0.2, wf.g + 0.6, PLANTER)
wf.box(pu0 + 0.1, pu1 - 0.1, 8.1, 8.9, wf.g + 0.6, wf.g + 1.0, HEDGE)

# ---- east front by the car park: upper storey over the glass, sign near the north corner ----
ef = Front(east)
zs = ef.upper(ef.u0 - 0.2, ef.u1 + 0.2)
north_u = ef.u1 if ef.p(ef.u1, 0, 0)[1] > ef.p(ef.u0, 0, 0)[1] else ef.u0
ef.sign("REVO", north_u - 5.5 * np.sign(north_u - (ef.u0 + ef.u1) / 2), zs, 1.5, FONT_BOLD, SIGN_RED)

mb.build("NRW_Rewe", "UVMap", mats)

bm_r = bmesh.new()
bm_r.from_mesh(me_b)
bm_r.faces.ensure_lookup_table()
bmesh.ops.delete(bm_r, geom=[bm_r.faces[i] for i in wedge], context="FACES_ONLY")
bm_r.to_mesh(me_b)
bm_r.free()
print("REWE west", len(west), "east", len(east), "red", len(red), "wedge", len(wedge), "faces", len(mb.f))
