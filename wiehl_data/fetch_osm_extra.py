"""Everything except roads from OSM for the Wiehl bbox: POIs, parking, lamps, crossings, barriers, rail."""
import json, urllib.parse, urllib.request
from collections import Counter

BBOX = "50.9445,7.5415,50.9525,7.5596"
Q = f"""[out:json][timeout:120];
(
  nwr["amenity"]({BBOX}); nwr["shop"]({BBOX}); nwr["leisure"]({BBOX}); nwr["tourism"]({BBOX});
  nwr["office"]({BBOX}); nwr["craft"]({BBOX}); nwr["healthcare"]({BBOX}); nwr["sport"]({BBOX});
  node["highway"]({BBOX}); nwr["barrier"]({BBOX}); nwr["railway"]({BBOX}); nwr["public_transport"]({BBOX});
  nwr["landuse"]({BBOX}); nwr["natural"]({BBOX}); node["entrance"]({BBOX}); way["building"]["name"]({BBOX});
  nwr["man_made"]({BBOX}); way["area:highway"]({BBOX});
);
out tags geom;"""
req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                             data=urllib.parse.urlencode({"data": Q}).encode(),
                             headers={"User-Agent": "blender-wiehl-scene/1.0", "Accept": "application/json"})
raw = urllib.request.urlopen(req, timeout=180).read()
open(r"A:\blender_work\wiehl_data\osm_extra.json", "wb").write(raw)
els = json.loads(raw)["elements"]
print("elements", len(els))
keys = ("amenity", "shop", "leisure", "highway", "barrier", "railway", "public_transport", "landuse", "natural", "sport", "man_made", "entrance")
for k in keys:
    c = Counter(f'{e["type"][0]}:{e["tags"][k]}' for e in els if k in e.get("tags", {}))
    print(k, c.most_common(25))
print("named:")
for e in els:
    t = e.get("tags", {})
    if "name" in t and any(k in t for k in ("amenity", "shop", "leisure", "sport", "building", "tourism")):
        print(" ", e["type"], t.get("name"), "|", {k: t[k] for k in ("amenity", "shop", "leisure", "sport", "building", "tourism") if k in t})
