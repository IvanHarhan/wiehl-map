import json, urllib.parse, urllib.request
from collections import Counter

Q = '[out:json][timeout:60];(way["highway"](50.9445,7.5415,50.9525,7.5596););out tags geom;'
req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                             data=urllib.parse.urlencode({"data": Q}).encode(),
                             headers={"User-Agent": "blender-wiehl-scene/1.0", "Accept": "application/json"})
raw = urllib.request.urlopen(req, timeout=120).read()
open(r"A:\blender_work\wiehl_data\osm_roads.json", "wb").write(raw)
els = json.loads(raw)["elements"]
print("ways", len(els))
print(Counter(e["tags"].get("highway") for e in els).most_common())
print("sidewalk", Counter(e["tags"].get("sidewalk") for e in els).most_common())
print("footway", Counter(e["tags"].get("footway") for e in els).most_common())
