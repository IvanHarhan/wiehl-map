"""Download NRW open-data 1 km tiles (DGM1, DOM1, DOP10, LoD2) for a lat/lon box into region_data/tiles.
Resumable: files already on disk with the right size are skipped. Usage:
python fetch_tiles.py MIN_LAT MIN_LON MAX_LAT MAX_LON"""
import json, math, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://www.opengeodata.nrw.de/produkte/geobasis"
PRODUCTS = {"dgm1": f"{BASE}/hm/dgm1_tiff/dgm1_tiff/", "dom1": f"{BASE}/hm/dom1_tiff/dom1_tiff/",
            "dop10": f"{BASE}/lusat/akt/dop/dop_jp2_f10/", "lod2": f"{BASE}/3dg/lod2_gml/lod2_gml/"}
PATTERN = {"dgm1": r"dgm1_32_(\d{3})_(\d{4})_1_nw_\d{4}\.tif", "dom1": r"dom1_32_(\d{3})_(\d{4})_1_nw_\d{4}\.tif",
           "dop10": r"dop10rgbi_32_(\d{3})_(\d{4})_1_nw_\d{4}\.jp2", "lod2": r"LoD2_32_(\d{3})_(\d{4})_1_NW\.gml"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiles")
UA = {"User-Agent": "blender-wiehl-scene/1.0"}

E_A, E_F, K0, LON0 = 6378137.0, 1 / 298.257222101, 0.9996, math.radians(9)
E2 = E_F * (2 - E_F); EP2 = E2 / (1 - E2)


def utm(lat, lon):
    phi, lam = math.radians(lat), math.radians(lon)
    N = E_A / math.sqrt(1 - E2 * math.sin(phi) ** 2)
    T = math.tan(phi) ** 2; C = EP2 * math.cos(phi) ** 2; A = math.cos(phi) * (lam - LON0)
    M = E_A * ((1 - E2/4 - 3*E2**2/64 - 5*E2**3/256) * phi - (3*E2/8 + 3*E2**2/32 + 45*E2**3/1024) * math.sin(2*phi)
               + (15*E2**2/256 + 45*E2**3/1024) * math.sin(4*phi) - (35*E2**3/3072) * math.sin(6*phi))
    x = 500000 + K0 * N * (A + (1 - T + C) * A**3 / 6 + (5 - 18*T + T*T + 72*C - 58*EP2) * A**5 / 120)
    y = K0 * (M + N * math.tan(phi) * (A*A/2 + (5 - T + 9*C + 4*C*C) * A**4 / 24 + (61 - 58*T + T*T + 600*C - 330*EP2) * A**6 / 720))
    return x, y


def get(url, timeout=300):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def main():
    if sys.argv[1] == "--tiles":  # a JSON list of [E, N] km tiles, e.g. territory_tiles.json
        wanted = {tuple(t) for t in json.load(open(sys.argv[2]))}
    else:
        la0, lo0, la1, lo1 = map(float, sys.argv[1:5])
        cs = [utm(la, lo) for la in (la0, la1) for lo in (lo0, lo1)]
        wanted = {(e, n) for e in range(int(min(c[0] for c in cs) // 1000), int(max(c[0] for c in cs) // 1000) + 1)
                  for n in range(int(min(c[1] for c in cs) // 1000), int(max(c[1] for c in cs) // 1000) + 1)}
    ex = range(min(t[0] for t in wanted), max(t[0] for t in wanted) + 1)
    ny = range(min(t[1] for t in wanted), max(t[1] for t in wanted) + 1)
    print(f"{len(wanted)} tiles per product within E {ex.start}..{ex.stop - 1}  N {ny.start}..{ny.stop - 1}", flush=True)
    jobs = []
    for prod, url in PRODUCTS.items():
        for attempt in range(5):
            try:
                listing = get(url).read().decode("utf-8", "replace")
                break
            except Exception:
                time.sleep(10)
        else:
            raise IOError(f"no listing for {prod}")
        names = {}
        for m in re.finditer(PATTERN[prod], listing):
            e, n = int(m.group(1)), int(m.group(2))
            if (e, n) in wanted:
                names[(e, n)] = max(names.get((e, n), ""), m.group(0))  # newest year wins
        os.makedirs(os.path.join(OUT, prod), exist_ok=True)
        missing = sorted(t for t in wanted if t not in names)
        print(prod, len(names), "tiles found", "missing:" if missing else "", missing or "", flush=True)
        jobs += [(prod, url + f, os.path.join(OUT, prod, f)) for f in sorted(names.values())]

    def fetch_once(url, path):
        r = get(url)
        size = int(r.headers.get("Content-Length", -1))
        if os.path.exists(path) and os.path.getsize(path) == size:
            return 0
        with open(path + ".part", "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        if size >= 0 and os.path.getsize(path + ".part") != size:
            raise IOError(f"short read {url}")
        os.replace(path + ".part", path)
        return size

    def fetch(job):
        prod, url, path = job
        # the server drops connections now and then: retry with a growing pause
        for attempt in range(8):
            try:
                return fetch_once(url, path)
            except Exception as e:
                print("retry", attempt + 1, os.path.basename(path), type(e).__name__, flush=True)
                time.sleep(5 * (attempt + 1))
        raise IOError(f"gave up on {url}")

    total = 0
    with ThreadPoolExecutor(4) as pool:
        for k, got in enumerate(pool.map(fetch, jobs), 1):
            total += got
            if k % 10 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} files, {total / 1e9:.2f} GB downloaded", flush=True)


main()
