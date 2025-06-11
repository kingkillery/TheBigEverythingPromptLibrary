"""Tests for the chat process endpoint"""

from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_chat_process_basic():
    resp = client.post("/api/chat/process", json={"prompt": "How to write better code?", "max_results": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert "matches" in data and isinstance(data["matches"], list)
    assert "optimized_prompt" in data
    assert "tweaked_match" in data


def test_chat_process_no_prompt():
    resp = client.post("/api/chat/process", json={"prompt": ""})
    assert resp.status_code == 400
