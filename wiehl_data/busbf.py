"""Run from build_blend.py (exec) after build_props.py: the bus station (Busbahnhof), modelled from street photos.
LoD2 stores its canopies as solid 3.7 m blocks, so build_blend drops those buildings (BUSBF_BLOCKS) and this builds
them from their footprints instead: a flat canopy on posts over a kiosk with louvred aluminium walls and
teal-framed glass waiting cabins at both ends, and two small shelters with benches on the island.
Uses globals: P, offset, BUSBF_BLOCKS, ground, MeshBuilder, box, simple_material, GLASS_P."""
import math
import numpy as np

Z3 = np.array((0.0, 0.0, 1.0))
# the earlier parts reuse names like V, so read the LoD2 arrays again
BV = np.load(os.path.join(P, "bld_verts.npy"))
BV[:, 2] += offset
BFL = np.load(os.path.join(P, "bld_face_len.npy"))
BFT = np.load(os.path.join(P, "bld_face_type.npy"))
BFB = np.load(os.path.join(P, "bld_face_bld.npy"))
BST = np.r_[0, np.cumsum(BFL)[:-1]]


def roof_frame(b):
    """largest flat roof face of building b -> centre (x, y), unit long axis, unit short axis (pointing north),
    half length, half width, roof z"""
    best = None
    for s, n, t, bb in zip(BST, BFL, BFT, BFB):
        if bb != b or t != 0:
            continue
        q = BV[s:s + n]
        area = abs(np.cross(q[1, :2] - q[0, :2], q[2, :2] - q[0, :2]))
        if best is None or area > best[0]:
            best = (area, q)
    q = best[1]
    e1, e2 = q[1, :2] - q[0, :2], q[2, :2] - q[1, :2]
    long_e, short_e = (e1, e2) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, e1)
    u = long_e / np.linalg.norm(long_e)
    v = np.array((-u[1], u[0]))
    if v[1] < 0:
        v = -v
    return q[:, :2].mean(0), u, v, np.linalg.norm(long_e) / 2, np.linalg.norm(short_e) / 2, q[:, 2].max()


mb = MeshBuilder()
FASCIA, SOFFIT, POST, LOUVRE, TEAL, WHITE, DOOR, GLASS, BENCH, SIGN, BACKING = range(11)
mats = [simple_material("NRW_BusbfFascia", (0.58, 0.6, 0.61), 0.5, 0.3),
        simple_material("NRW_BusbfSoffit", (0.19, 0.2, 0.21), 0.45, 0.6),
        simple_material("NRW_BusbfPost", (0.16, 0.17, 0.17), 0.4, 0.8),
        simple_material("NRW_BusbfLouvre", (0.35, 0.38, 0.39), 0.35, 0.8),
        simple_material("NRW_BusbfTeal", (0.03, 0.2, 0.2), 0.4, 0.2),
        simple_material("NRW_BusbfWhite", (0.68, 0.68, 0.66), 0.5),
        simple_material("NRW_BusbfDoor", (0.11, 0.13, 0.15), 0.5, 0.5),
        GLASS_P,
        simple_material("NRW_BusbfBench", (0.05, 0.055, 0.06), 0.5, 0.7),
        simple_material("NRW_BusbfSign", (0.013, 0.1, 0.4), 0.4),
        simple_material("NRW_BusbfBacking", (0.04, 0.04, 0.045), 0.8)]


class Frame:
    def __init__(self, c, u, v, z):
        self.c, self.u, self.v, self.z = c, np.array((*u, 0.0)), np.array((*v, 0.0)), z

    def p(self, u, v, z):
        return np.array((*self.c, 0.0)) + u * self.u + v * self.v + z * Z3

    def box(self, u0, u1, v0, v1, z0, z1, mat):
        box(mb, self.p((u0 + u1) / 2, (v0 + v1) / 2, (z0 + z1) / 2),
            self.u * (u1 - u0) / 2, self.v * (v1 - v0) / 2, Z3 * (z1 - z0) / 2, mat)

    def post(self, u, v, z0, z1, r=0.085):
        base = self.p(u, v, 0.0)
        for k in range(8):
            a0, a1 = 2 * math.pi * k / 8, 2 * math.pi * (k + 1) / 8
            q0 = base + r * np.array((math.cos(a0), math.sin(a0), 0.0))
            q1 = base + r * np.array((math.cos(a1), math.sin(a1), 0.0))
            mb.face([q0 + Z3 * z0, q1 + Z3 * z0, q1 + Z3 * z1, q0 + Z3 * z1], POST)

    def canopy(self, hl, hw, thick, rib_step=0.25):
        """flat roof slab with a trapezoid-sheet underside (ribs across the short side)"""
        self.box(-hl, hl, -hw, hw, self.z - thick, self.z, FASCIA)
        u = -hl + 0.2
        while u < hl - 0.15:
            self.box(u - 0.035, u + 0.035, -hw + 0.12, hw - 0.12, self.z - thick - 0.06, self.z - thick, SOFFIT)
            u += rib_step

    def bench(self, u, v, length, along_u, facing, g):
        """perforated-metal bench on two legs; facing = +1/-1 along the axis across the seat"""
        def b(a0, a1, c0, c1, z0, z1, mat):
            if along_u:
                self.box(a0, a1, c0, c1, z0, z1, mat)
            else:
                self.box(c0, c1, a0, a1, z0, z1, mat)
        a, c = (u, v) if along_u else (v, u)
        b(a - length / 2, a + length / 2, c - 0.2, c + 0.2, g + 0.42, g + 0.46, BENCH)
        back = c - facing * 0.2
        b(a - length / 2, a + length / 2, back - 0.02, back + 0.02, g + 0.5, g + 0.85, BENCH)
        for end in (-1, 1):
            la = a + end * (length / 2 - 0.15)
            b(la - 0.03, la + 0.03, c - 0.18, c + 0.18, g - 0.05, g + 0.42, POST)
            b(la - 0.03, la + 0.03, back - 0.03, back + 0.03, g + 0.42, g + 0.85, POST)


def kiosk_wall(f, fixed, fixed_val, out, a0, segs, g, top):
    """one louvred kiosk wall: segs = [(width, kind)] from a0 along the wall; kinds L louvre, W white, T teal,
    D grey door, E teal door. fixed = 'u' or 'v' (the wall's constant coordinate), out = outward sign"""
    def b(s0, s1, d0, d1, z0, z1, mat):
        w0, w1 = fixed_val + out * d0, fixed_val + out * d1
        lo, hi = min(w0, w1), max(w0, w1)
        if fixed == "u":
            f.box(lo, hi, s0, s1, z0, z1, mat)
        else:
            f.box(s0, s1, lo, hi, z0, z1, mat)

    def louvre(s0, s1, z0, z1):
        z = z0 + 0.08
        while z < z1 - 0.06:
            b(s0 + 0.03, s1 - 0.03, 0.0, 0.05, z - 0.045, z + 0.045, LOUVRE)
            z += 0.13
        for e in (s0, s1 - 0.03):
            b(e, e + 0.03, 0.0, 0.07, z0, z1, LOUVRE)

    s = a0
    for w, kind in segs:
        s0, s1 = s, s + w
        if kind == "L":
            louvre(s0, s1, g, top)
        elif kind in "WT":
            b(s0, s1, 0.0, 0.05, g - 0.15, top, WHITE if kind == "W" else TEAL)
        else:
            b(s0, s1, 0.0, 0.05, g - 0.15, g + 2.2, DOOR if kind == "D" else TEAL)
            b(s0 - 0.03, s1 + 0.03, 0.0, 0.07, g + 2.2, g + 2.28, LOUVRE)
            louvre(s0, s1, g + 2.28, top)
            if kind == "D":  # accessible-toilet sign
                b(s0 + w / 2 - 0.14, s0 + w / 2 + 0.14, 0.05, 0.06, g + 1.45, g + 1.73, SIGN)
        s = s1


def cabin(k, side, KL, KW, g):
    """teal-framed glass waiting cabin against the kiosk end at u = side * KL, on the front (+v) half"""
    u_in, u_out = side * KL, side * (KL + 2.6)
    ulo, uhi = min(u_in, u_out), max(u_in, u_out)
    v0, v1, top = 0.4, KW, g + 2.5
    for u in (u_out,):
        for v in (v0, v1):
            k.box(u - 0.05, u + 0.05, v - 0.05, v + 0.05, g - 0.1, top, TEAL)
    for v in (v0, v1):
        k.box(ulo, uhi, v - 0.05, v + 0.05, top - 0.14, top, TEAL)
        k.box(ulo, uhi, v - 0.05, v + 0.05, g - 0.1, g + 0.1, TEAL)
        k.box(ulo + 0.05, uhi - 0.05, v - 0.012, v + 0.012, g + 0.1, top - 0.14, GLASS)
        mid = side * (KL + 1.3)
        k.box(mid - 0.035, mid + 0.035, v - 0.05, v + 0.05, g + 0.1, top - 0.14, TEAL)
    k.box(u_out - 0.05, u_out + 0.05, v0, v1, top - 0.14, top, TEAL)
    k.box(ulo, uhi, v0 - 0.05, v1 + 0.05, top, top + 0.06, TEAL)
    k.bench(side * (KL + 1.3), v0 + 0.35, 1.8, True, 1, g)


drop = set(int(b) for b in BUSBF_BLOCKS)
frames = {b: roof_frame(b) for b in drop}
canopy_b = max(drop, key=lambda b: frames[b][3] * frames[b][4])
cc, cu, cv, CL, CW, cz = frames[canopy_b]
# the kiosk is the building whose roof sits inside the big canopy; the rest are the island shelters
kiosk_b = min((b for b in drop if b != canopy_b), key=lambda b: np.linalg.norm(frames[b][0] - cc))
shelters = [b for b in drop if b not in (canopy_b, kiosk_b)]

# ---- main canopy over the kiosk ----
g = float(ground(cc[None])[0])
cf = Frame(cc, cu, cv, cz)
cf.canopy(CL, CW, 0.32)
under = cz - 0.38
for su in (-1, 1):
    for sv in (-1, 1):
        cf.post(su * (CL - 1.0), sv * (CW - 0.8), g - 0.2, under)

# ---- kiosk: 5 x 5 m, louvred walls up to the canopy soffit ----
kc = frames[kiosk_b][0]
k = Frame(kc, cu, cv, cz)
KL = KW = 2.5
kg = float(ground(kc[None])[0])
top = cz - 0.32
k.box(-KL + 0.05, KL - 0.05, -KW + 0.05, KW - 0.05, kg - 0.2, top, BACKING)
kiosk_wall(k, "v", KW - 0.05, 1, -KL, [(1.4, "L"), (0.6, "W"), (1.0, "E"), (2.0, "L")], kg, top)
kiosk_wall(k, "v", -KW + 0.05, -1, -KL, [(1.6, "L"), (1.0, "D"), (2.4, "L")], kg, top)
for side in (-1, 1):
    kiosk_wall(k, "u", side * (KL - 0.05), side, -KW,
               [(0.4, "L"), (0.25, "T"), (0.4, "L"), (1.0, "D"), (0.35, "L"), (0.5, "W"), (2.1, "L")], kg, top)
    cabin(k, side, KL, KW, kg)
    k.bench(side * (KL + 1.5), -1.3, 1.8, True, -1, kg)

# ---- small shelters on the island ----
for b in shelters:
    c, u, v, hl, hw, z = frames[b]
    f = Frame(c, u, v, z)
    sg = float(ground(c[None])[0])
    f.canopy(hl, hw, 0.22)
    for su in (-1, 1):
        for sv in (-1, 1):
            f.post(su * (hl - 0.3), sv * (hw - 0.35), sg - 0.2, z - 0.28)
        # glass wind shield across each end
        eu = su * (hl - 0.3)
        f.box(eu - 0.012, eu + 0.012, -hw + 0.43, hw - 0.43, sg + 0.15, sg + 2.2, GLASS)
        for zz in (sg + 0.1, sg + 2.2):
            f.box(eu - 0.04, eu + 0.04, -hw + 0.35, hw - 0.35, zz - 0.04, zz + 0.04, TEAL)
    f.bench(0.0, -0.2, 2.2, True, 1, sg)

busbf = mb.build("NRW_Busbf", "UVMap", mats)
print("BUSBF canopy", canopy_b, "kiosk", kiosk_b, "shelters", shelters, "faces", len(mb.f))
