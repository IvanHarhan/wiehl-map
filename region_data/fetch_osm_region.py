"""OSM for the region bbox (from prepared/meta.json): roads, everything build_extras/props/details read
(same keys as wiehl_data/fetch_osm_extra.py), and all house numbers (to find buildings by address here, locally).
Writes prepared/osm_roads.json, osm_extra.json, osm_addr.json."""
import json, math, os, time, urllib.parse, urllib.request

D = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(D, "prepared")
meta = json.load(open(os.path.join(P, "meta.json")))
UX0, UY0, UX1, UY1 = meta["utm_bbox"]
SERVERS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
           "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]


def utm_inv(x, y):
    a, f, k0, lon0 = 6378137.0, 1 / 298.257222101, 0.9996, math.radians(9)
    e2 = f * (2 - f); ep2 = e2 / (1 - e2); e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    mu = y / k0 / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    p1 = mu + (3*e1/2 - 27*e1**3/32) * math.sin(2*mu) + (21*e1**2/16 - 55*e1**4/32) * math.sin(4*mu) + (151*e1**3/96) * math.sin(6*mu)
    n1 = a / math.sqrt(1 - e2 * math.sin(p1)**2); t1 = math.tan(p1)**2; c1 = ep2 * math.cos(p1)**2
    r1 = a * (1 - e2) / (1 - e2 * math.sin(p1)**2) ** 1.5; d = (x - 500000) / (n1 * k0)
    lat = p1 - (n1 * math.tan(p1) / r1) * (d*d/2 - (5 + 3*t1 + 10*c1 - 4*c1*c1 - 9*ep2) * d**4/24)
    lon = lon0 + (d - (1 + 2*t1 + c1) * d**3/6) / math.cos(p1)
    return math.degrees(lat), math.degrees(lon)


corners = [utm_inv(x, y) for x in (UX0, UX1) for y in (UY0, UY1)]
BBOX = "%.5f,%.5f,%.5f,%.5f" % (min(c[0] for c in corners), min(c[1] for c in corners),
                                max(c[0] for c in corners), max(c[1] for c in corners))
print("bbox", BBOX, flush=True)


def overpass(q, name):
    for attempt in range(9):
        url = SERVERS[attempt % len(SERVERS)]
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "blender-wiehl-scene/1.0", "Accept": "application/json"})
            raw = urllib.request.urlopen(req, timeout=400).read()
            els = json.loads(raw)["elements"]
            open(os.path.join(P, name), "wb").write(raw)
            print(name, len(els), "elements from", url, flush=True)
            return
        except Exception as e:
            print("retry", name, url, type(e).__name__, e, flush=True)
            time.sleep(15)
    raise IOError("overpass failed for " + name)


overpass(f'[out:json][timeout:300];(way["highway"]({BBOX}););out tags geom;', "osm_roads.json")
overpass(f"""[out:json][timeout:300];
(
  nwr["amenity"]({BBOX}); nwr["shop"]({BBOX}); nwr["leisure"]({BBOX}); nwr["tourism"]({BBOX});
  nwr["office"]({BBOX}); nwr["craft"]({BBOX}); nwr["healthcare"]({BBOX}); nwr["sport"]({BBOX});
  node["highway"]({BBOX}); nwr["barrier"]({BBOX}); nwr["railway"]({BBOX}); nwr["public_transport"]({BBOX});
  nwr["landuse"]({BBOX}); nwr["natural"]({BBOX}); nwr["water"]({BBOX}); way["waterway"]({BBOX});
  node["entrance"]({BBOX}); way["building"]["name"]({BBOX}); nwr["man_made"]({BBOX}); way["area:highway"]({BBOX});
);
out tags geom;""", "osm_extra.json")
overpass(f'[out:json][timeout:300];(nwr["addr:housenumber"]({BBOX}););out tags center;', "osm_addr.json")
print("OSM_DONE")
