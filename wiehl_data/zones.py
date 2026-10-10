"""Architecture zones of the region (the user's analysis, September 2026) -> per-building facade parameters.
exec'd by export_region.py; uses its globals np, json, os, P and tm (lat/lon -> local metres).
Zones had no polygons, so a building belongs to:
  a town core   within CORE_M of one of the core's streets (OSM names) and within `r` of its centre,
  else a town   by distance to its centre (first match in TOWNS order),
  else R1       farms and hamlets look like Brüchermühle."""


def lin(h):
    c = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# styles: plaster, slate, timber, brick, modern (the analysis' "commercial" share of houses counts as modern)
# plaster: wall colours with weights; frames: white, light grey, anthracite, grey (weights)
# wt: window types single upright / pair with a mullion / square / wide low (WTYPES), bay: window spacing range (m)
# — from the user's Street View shots (2026-09-24): far fewer and smaller windows than a grid, lots of plain wall
# shutters/roller: share of houses; shop: share of 2+ storey houses with a shop ground floor to the street
# plinth, steps, overhang: ranges; canopy: share of house doors with a porch roof
# balc: share of houses, share of flats (3+ storeys), side
ZONES = {
    "W1": dict(styles=(.35, .25, .15, .10, .15), plaster=(("F1EEE7", .35), ("D8D2C7", .25), ("E4E0D8", .2), ("C5C1B9", .2)),
               frames=(.5, .25, .25, 0), wt=(.5, .3, .2, 0), bay=(3.4, 4.2), shutters=.15, roller=.35, shop=.45,
               plinth=(.4, 1.0), steps=(1, 5), canopy=.25, balc=(.1, .35), side="street", overhang=(.3, .6)),
    "W2": dict(styles=(.45, .20, .03, .10, .22), plaster=(("F1EFE9", .35), ("E1DED7", .3), ("D3D0C9", .2), ("C5C5C0", .15)),
               frames=(.5, .25, .25, 0), wt=(.3, .2, .2, .3), bay=(4.0, 5.2), shutters=.08, roller=.6, shop=0,
               plinth=(.8, 1.8), steps=(2, 8), canopy=.35, balc=(.35, .65), side="garden", overhang=(.4, .8)),
    "A1": dict(styles=(.42, .28, .12, .06, .12), plaster=(("F0EEE7", .4), ("DDD8CE", .3), ("CFCAC0", .2), ("E8E5DE", .1)),
               frames=(.7, 0, .3, 0), wt=(.45, .25, .2, .1), bay=(4.2, 5.5), shutters=.2, roller=.25, shop=.02,
               plinth=(.5, 1.2), steps=(1, 6), canopy=.3, balc=(.15, .25), side="garden", overhang=(.4, .9)),
    "B1": dict(styles=(.40, .27, .10, .08, .15), plaster=(("F0EEE8", .4), ("DDD9D1", .3), ("CEC9C1", .2), ("E6E2DB", .1)),
               frames=(.7, 0, .3, 0), wt=(.4, .25, .2, .15), bay=(3.8, 5.0), shutters=.12, roller=.45, shop=.2,
               plinth=(.4, 1.4), steps=(1, 7), canopy=.3, balc=(.25, .5), side="valley", overhang=(.3, .8)),
    "D1": dict(styles=(.30, .20, .05, .20, .25), plaster=(("F0EEE8", .4), ("D8D5CD", .3), ("B7B0A5", .2), ("D5D1C8", .1)),
               frames=(.5, .25, .25, 0), wt=(.5, .3, .2, 0), bay=(3.5, 4.5), shutters=.08, roller=.45, shop=.5,
               plinth=(.4, 1.0), steps=(0, 5), canopy=.25, balc=(.15, .35), side="street", overhang=(.3, .7)),
    # the hillside housing of Dieringhausen and Gummersbach (first analysis, zone D3)
    "D3": dict(styles=(.55, .15, .03, .12, .15), plaster=(("F0EEE8", .35), ("E0DDD6", .3), ("D1CDC5", .25), ("C8C4BC", .1)),
               frames=(.6, .3, .1, 0), wt=(.3, .2, .2, .3), bay=(4.0, 5.2), shutters=.05, roller=.6, shop=0,
               plinth=(.7, 1.5), steps=(2, 6), canopy=.3, balc=(.3, .6), side="valley", overhang=(.3, .7)),
    "G1": dict(styles=(.25, .08, .02, .25, .40), plaster=(("E6E2DB", .4), ("D2CDC5", .3), ("B9B3AA", .2), ("DAD6CE", .1)),
               frames=(.3, 0, .5, .2), wt=(.4, .4, .2, 0), bay=(3.2, 4.0), shutters=.02, roller=.3, shop=.8,
               plinth=(.2, .8), steps=(0, 4), canopy=.5, balc=(.05, .15), side="street", overhang=(.1, .6)),
    "R1": dict(styles=(.43, .30, .12, .04, .11), plaster=(("F0EEE7", .4), ("DDD8CE", .3), ("CFCAC0", .2), ("E7E3DB", .1)),
               frames=(.7, 0, .3, 0), wt=(.45, .25, .2, .1), bay=(4.2, 5.6), shutters=.2, roller=.2, shop=.02,
               plinth=(.5, 1.3), steps=(1, 6), canopy=.3, balc=(.15, .2), side="garden", overhang=(.4, 1.0)),
}
WTYPES = np.array([(1.0, 1.35), (1.6, 1.35), (1.05, 1.05), (1.9, 1.15)])
# Bergisch slate comes in parts too: all over, gable only, upper floors, ground floor only, weather (west) side only
SLATE_MODES = (.35, .2, .2, .1, .15)
ZONE_IDS = list(ZONES)
MODERN = (("D9D9D4", .4), ("B8B5AE", .3), ("8A8A84", .2), ("303030", .1))
TIMBER_FILL = "F2F0E8"
CORE_M = 40.0
# town cores: (zone, centre lat/lon, radius m, street names, whole-disk radius m)
CORES = [("W1", (50.9470, 7.5496), 900, ("Hauptstraße", "Bahnhofstraße", "Mühlenstraße", "Wiesenstraße", "Weiherplatz", "Weiherstraße"), 0),
         ("D1", (50.9850, 7.5370), 1300, ("Dieringhauser Straße", "Neudieringhauser Straße", "Vollmerhauser Straße", "Bahnhofstraße"), 0),
         ("G1", (51.0240, 7.5656), 1500, ("Kaiserstraße", "Hindenburgstraße", "Steinmüllerallee", "Moltkestraße", "La-Roche-sur-Yon-Straße"), 450)]
TOWNS = [("A1", (50.9870, 7.5760), 700), ("B1", (50.9570, 7.4950), 1000), ("R1", (50.9265, 7.6380), 900),
         ("W2", (50.9470, 7.5496), 1800), ("D3", (50.9850, 7.5370), 1400), ("D3", (51.0240, 7.5656), 2000)]
CAR_ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street",
             "pedestrian", "primary_link", "secondary_link", "tertiary_link"}

_roads = json.load(open(os.path.join(os.path.dirname(P), "osm_roads.json"), encoding="utf-8"))["elements"]


def _dense(geom, step=4.0):
    pts = np.array([tm(p["lat"], p["lon"]) for p in geom])
    out = [pts[:1]]
    for a, b in zip(pts[:-1], pts[1:]):
        k = max(1, int(np.hypot(*(b - a)) / step))
        out.append(a + (b - a) * (np.arange(1, k + 1) / k)[:, None])
    return np.vstack(out)


ROAD_PTS, ROAD_NAMES = [], []
for e in _roads:
    t = e.get("tags", {})
    if t.get("highway") in CAR_ROADS and len(e.get("geometry", [])) >= 2:
        d = _dense(e["geometry"])
        ROAD_PTS.append(d)
        ROAD_NAMES += [t.get("name", "")] * len(d)
ROAD_PTS = np.vstack(ROAD_PTS)
ROAD_NAMES = np.array(ROAD_NAMES)
del _roads


class Grid:
    """nearest of many 2-D points, bucketed in square cells"""
    def __init__(self, pts, cell=30.0):
        self.p, self.c = pts, cell
        keys = np.floor(pts / cell).astype(np.int64)
        k = keys[:, 0] * 1000003 + keys[:, 1]
        order = np.argsort(k)
        self.order, ks = order, k[order]
        u, start = np.unique(ks, return_index=True)
        self.start = dict(zip(u.tolist(), start.tolist()))
        self.end = dict(zip(u.tolist(), np.r_[start[1:], len(ks)].tolist()))

    def nearest(self, q, reach=2):
        """index of the nearest point (or -1) within `reach` cells, for each query point"""
        out = np.full(len(q), -1, np.int64)
        best = np.full(len(q), np.inf)
        qk = np.floor(q / self.c).astype(np.int64)
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                kk = (qk[:, 0] + dx) * 1000003 + (qk[:, 1] + dy)
                for i in np.nonzero([k_ in self.start for k_ in kk.tolist()])[0]:
                    idx = self.order[self.start[int(kk[i])]:self.end[int(kk[i])]]
                    d = np.hypot(*(self.p[idx] - q[i]).T)
                    j = int(np.argmin(d))
                    if d[j] < best[i]:
                        best[i], out[i] = d[j], idx[j]
        return out, best


ROADS = Grid(ROAD_PTS)


def assign_zones(cen):
    """zone index per building centre (local metres)"""
    zone = np.full(len(cen), ZONE_IDS.index("R1"))
    done = np.zeros(len(cen), bool)
    for zid, (la, lo), r, streets, disk in CORES:
        c = np.array(tm(la, lo))
        dc = np.hypot(*(cen - c).T)
        cand = np.nonzero(~done & (dc < r))[0]
        sel = np.isin(ROAD_NAMES, streets)
        g = Grid(ROAD_PTS[sel])
        _, d = g.nearest(cen[cand], reach=2)
        hit = cand[(d < CORE_M) | (dc[cand] < disk)]
        zone[hit], done[hit] = ZONE_IDS.index(zid), True
    for zid, (la, lo), r in TOWNS:
        hit = ~done & (np.hypot(*(cen - np.array(tm(la, lo))).T) < r)
        zone[hit], done[hit] = ZONE_IDS.index(zid), True
    return zone


def pick(rng, options, n):
    """weighted choice of hex colours -> (n, 3) linear"""
    cols = np.array([lin(h) for h, _ in options])
    w = np.array([w for _, w in options], float)
    return cols[rng.choice(len(cols), n, p=w / w.sum())]
