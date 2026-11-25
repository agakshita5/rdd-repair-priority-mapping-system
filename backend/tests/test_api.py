from tests.conftest import make_image, upload

LAT, LON = 19.0760, 72.8777
M_PER_DEG_LAT = 111_195


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_mode": "mock"}


def test_detect_uses_exif_gps(client):
    r = upload(client, make_image(LAT, LON))
    assert r.status_code == 200
    body = r.json()
    assert body["gps_source"] == "exif"
    assert abs(body["lat"] - LAT) < 1e-5
    assert len(body["detections"]) >= 1
    assert body["location"]["priority"] in ("High", "Medium", "Low")
    assert body["location"]["status"] == "Reported"
    assert client.get(body["annotated_url"]).status_code == 200


def test_detect_without_gps_needs_location(client):
    r = upload(client, make_image())
    assert r.status_code == 422
    assert "location_required" in r.json()["detail"]


def test_detect_with_manual_pin(client):
    r = upload(client, make_image(), lat=LAT, lon=LON)
    assert r.status_code == 200
    assert r.json()["gps_source"] == "manual"


def test_exif_takes_precedence_over_form(client):
    r = upload(client, make_image(LAT, LON), lat=10.0, lon=10.0)
    assert r.json()["gps_source"] == "exif"


def test_invalid_coordinates_rejected(client):
    assert upload(client, make_image(), lat=100, lon=LON).status_code == 422


def test_non_image_rejected(client):
    r = client.post("/detect", files={"file": ("x.txt", b"hello", "text/plain")}, data={"lat": LAT, "lon": LON})
    assert r.status_code == 415


def test_unsupported_format_rejected(client):
    r = upload(client, make_image(fmt="GIF"), lat=LAT, lon=LON)
    assert r.status_code == 415


def test_invalid_source_rejected(client):
    assert upload(client, make_image(), lat=LAT, lon=LON, source="other").status_code == 422


def test_nearby_uploads_merge_into_one_location(client):
    a = upload(client, make_image(seed=1), lat=LAT, lon=LON).json()
    b = upload(client, make_image(seed=2), lat=LAT + 5 / M_PER_DEG_LAT, lon=LON).json()
    assert a["location"]["id"] == b["location"]["id"]
    assert b["location"]["n_images"] == 2
    assert b["location"]["total_score"] > a["location"]["total_score"]


def test_distant_uploads_create_separate_locations(client):
    a = upload(client, make_image(seed=1), lat=LAT, lon=LON).json()
    b = upload(client, make_image(seed=2), lat=LAT + 50 / M_PER_DEG_LAT, lon=LON).json()
    assert a["location"]["id"] != b["location"]["id"]


def test_locations_ranked_by_score(client):
    for i in range(4):
        upload(client, make_image(seed=i), lat=LAT + i * 0.01, lon=LON)
    rows = client.get("/locations").json()
    scores = [r["total_score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))


def test_locations_filters(client):
    for i in range(4):
        upload(client, make_image(seed=i), lat=LAT + i * 0.01, lon=LON)
    for row in client.get("/locations", params={"priority": "Low"}).json():
        assert row["priority"] == "Low"
    for row in client.get("/locations", params={"damage_type": "D40"}).json():
        assert "D40" in row["damage_types"]
    assert client.get("/locations", params={"priority": "Urgent"}).status_code == 422


def test_location_detail(client):
    loc_id = upload(client, make_image(), lat=LAT, lon=LON).json()["location"]["id"]
    body = client.get(f"/locations/{loc_id}").json()
    assert len(body["detections"]) == body["n_detections"]
    assert len(body["images"]) == 1
    assert client.get("/locations/9999").status_code == 404


def test_status_update(client):
    loc_id = upload(client, make_image(), lat=LAT, lon=LON).json()["location"]["id"]
    r = client.patch(f"/locations/{loc_id}/status", json={"status": "Scheduled"})
    assert r.status_code == 200 and r.json()["status"] == "Scheduled"
    assert client.patch(f"/locations/{loc_id}/status", json={"status": "Done"}).status_code == 422
    assert client.patch("/locations/9999/status", json={"status": "Repaired"}).status_code == 404


def test_repaired_location_not_reused(client):
    loc_id = upload(client, make_image(seed=1), lat=LAT, lon=LON).json()["location"]["id"]
    client.patch(f"/locations/{loc_id}/status", json={"status": "Repaired"})
    new = upload(client, make_image(seed=2), lat=LAT, lon=LON).json()
    assert new["location"]["id"] != loc_id


def test_simulated_flag(client):
    r = upload(client, make_image(), lat=LAT, lon=LON, source="simulated")
    assert r.json()["location"]["simulated"] is True
