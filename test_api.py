"""Basic API tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
sys.path.insert(0, str(APP_DIR))

from app import app  # noqa: E402


def test_home_endpoint():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.get_json()["message"]


def test_chat_endpoint():
    client = app.test_client()
    response = client.post("/chat", json={"message": "Where is my order?"})
    assert response.status_code == 200
    data = response.get_json()
    assert "intent" in data
    assert "response" in data
