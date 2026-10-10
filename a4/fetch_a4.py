"""A4 motorway (both carriageways, links too) between Köln and Wiehl from Overpass -> a4/a4_ways.json"""
import json, time, urllib.parse, urllib.request
SERVERS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
Q = '[out:json][timeout:300];(way["highway"~"^motorway"]["ref"~"(^|;)A ?4($|;)"](50.85,6.95,51.02,7.62););out tags geom;'
for attempt in range(8):
    try:
        req = urllib.request.Request(SERVERS[attempt % 2], data=urllib.parse.urlencode({"data": Q}).encode(),
                                     headers={"User-Agent": "blender-wiehl-scene/1.0"})
        raw = urllib.request.urlopen(req, timeout=400).read()
        els = json.loads(raw)["elements"]
        open(r"A:\blender_work\a4\a4_ways.json", "wb").write(raw)
        lons = [p["lon"] for e in els for p in e["geometry"]]
        print(len(els), "ways, lon", min(lons), max(lons))
        break
    except Exception as e:
        print("retry", e); time.sleep(15)
