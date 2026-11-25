"""Backup demo data: send RDD2022 India test images to /detect with SIMULATED locations.

RDD2022 images have no GPS, so each image is placed at a random point near a main road
in an Indian city. Locations are stored with simulated = true and must be labelled as
simulated in the dashboard and report.

    python scripts/seed_simulated.py path/to/rdd2022/India/test/images --n 80
"""
import argparse
import random
import sys
from pathlib import Path

import requests

# (lat, lon) anchor points on main roads
ANCHORS = {
    "bengaluru": (12.9716, 77.5946),
    "mumbai": (19.0760, 72.8777),
    "pune": (18.5204, 73.8567),
    "hyderabad": (17.3850, 78.4867),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
    "jaipur": (26.9124, 75.7873),
    "ahmedabad": (23.0225, 72.5714),
}
JITTER_DEG = 0.02  # roughly 2 km


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images = sorted(
        p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not images:
        sys.exit(f"No images found in {args.image_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(images)
    cities = list(ANCHORS)
    ok = 0

    for i, path in enumerate(images[: args.n]):
        lat0, lon0 = ANCHORS[cities[i % len(cities)]]
        lat = lat0 + rng.uniform(-JITTER_DEG, JITTER_DEG)
        lon = lon0 + rng.uniform(-JITTER_DEG, JITTER_DEG)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        with path.open("rb") as f:
            r = requests.post(
                f"{args.api}/detect",
                files={"file": (path.name, f, mime)},
                data={"lat": lat, "lon": lon, "source": "simulated"},
                timeout=60,
            )
        if r.ok:
            ok += 1
            print(f"{path.name}: {len(r.json()['detections'])} detections")
        else:
            print(f"{path.name}: HTTP {r.status_code} {r.text[:120]}")

    print(f"Done. {ok}/{min(args.n, len(images))} images processed.")


if __name__ == "__main__":
    main()
