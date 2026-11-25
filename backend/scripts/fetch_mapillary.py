"""Pull geotagged street-level images of Indian roads from Mapillary and send them to /detect.

Setup: create a free account at mapillary.com -> Developers -> register an app -> copy the
client access token.

    export MAPILLARY_TOKEN='MLY|xxxx'
    python scripts/fetch_mapillary.py --per-city 40

Images are CC-BY-SA; cite Mapillary in the report.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import requests

GRAPH_URL = "https://graph.mapillary.com/images"

# Small boxes (~2 km) on busy roads; (min_lon, min_lat, max_lon, max_lat)
CITIES = {
    "bengaluru": (77.595, 12.965, 77.615, 12.985),
    "mumbai": (72.837, 19.109, 72.857, 19.129),
    "pune": (73.837, 18.520, 73.857, 18.540),
    "hyderabad": (78.438, 17.427, 78.458, 17.447),
    "chennai": (80.220, 13.030, 80.240, 13.050),
    "kolkata": (88.350, 22.560, 88.370, 22.580),
}


def fetch_city(token: str, bbox: tuple, limit: int) -> list[dict]:
    params = {
        "access_token": token,
        "bbox": ",".join(str(v) for v in bbox),
        "fields": "id,computed_geometry,geometry,thumb_1024_url",
        "limit": limit,
    }
    resp = requests.get(GRAPH_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--per-city", type=int, default=40)
    parser.add_argument("--cities", nargs="*", default=list(CITIES))
    parser.add_argument("--out", default="data/mapillary")
    args = parser.parse_args()

    token = os.getenv("MAPILLARY_TOKEN")
    if not token:
        sys.exit("Set MAPILLARY_TOKEN first (see docstring).")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sent = with_damage = 0

    for city in args.cities:
        items = fetch_city(token, CITIES[city], args.per_city)
        print(f"{city}: {len(items)} images")
        for item in items:
            geom = item.get("computed_geometry") or item.get("geometry")
            url = item.get("thumb_1024_url")
            if not geom or not url:
                continue
            lon, lat = geom["coordinates"]
            path = out_dir / f"{city}_{item['id']}.jpg"
            if not path.exists():
                path.write_bytes(requests.get(url, timeout=30).content)
            with path.open("rb") as f:
                r = requests.post(
                    f"{args.api}/detect",
                    files={"file": (path.name, f, "image/jpeg")},
                    data={"lat": lat, "lon": lon, "source": "mapillary"},
                    timeout=60,
                )
            if r.ok:
                sent += 1
                n = len(r.json()["detections"])
                with_damage += n > 0
                print(f"  {path.name}: {n} detections")
            else:
                print(f"  {path.name}: HTTP {r.status_code} {r.text[:120]}")
            time.sleep(0.1)

    print(f"Done. Sent {sent} images, {with_damage} had damage detected.")


if __name__ == "__main__":
    main()
