"""Roads from the A4 corridor's west end into Köln-Deutz: the B 55a (Stadtautobahn), the A4/A3 at the Kreuz Köln-Ost
and their links, for the Köln corridor. -> a4/koeln_route.json"""
import json, time, urllib.parse, urllib.request
SERVERS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
           "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
Q = '[out:json][timeout:300];(way["highway"~"^(motorway|trunk|primary)(_link)?$"](50.925,6.94,50.965,7.085););out tags geom;'
for attempt in range(9):
    try:
        req = urllib.request.Request(SERVERS[attempt % 3], data=urllib.parse.urlencode({"data": Q}).encode(),
                                     headers={"User-Agent": "blender-wiehl-scene/1.0"})
        raw = urllib.request.urlopen(req, timeout=400).read()
        els = json.loads(raw)["elements"]
        open(r"A:\blender_work\a4\koeln_route.json", "wb").write(raw)
        refs = {}
        for e in els:
            r = e["tags"].get("ref", "-") + " " + e["tags"]["highway"]
            refs[r] = refs.get(r, 0) + 1
        print(len(els), "ways"); print(sorted(refs.items(), key=lambda x: -x[1])[:25])
        break
    except Exception as ex:
        print("retry", ex); time.sleep(15)
