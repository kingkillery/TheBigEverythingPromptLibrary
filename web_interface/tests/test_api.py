"""API endpoint tests for the Prompt Library backend."""

from fastapi.testclient import TestClient
import pytest

# Import the FastAPI app
from backend.app import app  # noqa: E402, import after adjusting path

client = TestClient(app)


def test_ping_endpoint():
    """Health endpoint should return status ok and version info."""
    response = client.get("/api/ping")
    assert response.status_code == 200, "Ping endpoint did not return 200"
    data = response.json()
    assert data.get("status") == "ok", "Ping status not ok"
    assert "version" in data, "Version missing in ping response"


def test_get_search_config():
    """Ensure we can retrieve current search configuration."""
    response = client.get("/api/search/config")
    assert response.status_code == 200, "Search config GET failed"
    data = response.json()
    assert "stop_words" in data and "weights" in data, "Config structure invalid"


def test_update_search_config():
    """POST new search configuration and verify it is applied."""
    payload = {
        "stop_words": ["foo", "bar"],
        "weights": {"title": 0.5, "quality": 0.1}
    }
    response = client.post("/api/search/config", json=payload)
    assert response.status_code == 200, "Search config POST failed"
    data = response.json()
    assert data.get("status") == "updated", "Config update status incorrect"

    cfg = data.get("config", {})
    # Check stop words contain the new additions
    assert set(["foo", "bar"]).issubset(set(cfg.get("stop_words", []))), "Stop words not updated"
    # Check weight update applied
    assert abs(cfg.get("weights", {}).get("title", 0) - 0.5) < 1e-6, "Title weight not updated"


def test_search_details():
    """Detailed search should return score breakdown for each item."""
    params = {"query": "coding", "limit": 1}
    response = client.get("/api/search/details", params=params)
    assert response.status_code == 200, "Search details endpoint failed"

    data = response.json()
    assert "details" in data, "Details key missing in response"
    if data["details"]:
        detail = data["details"][0]
        assert "score" in detail and "breakdown" in detail, "Score breakdown missing"


def test_trending_feed():
    """Trending feed endpoint should return a list of articles."""
    resp = client.get("/api/trending-feed")
    if resp.status_code == 503:
        pytest.skip("Trending feed dependency not installed")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
