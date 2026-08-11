from fastapi.testclient import TestClient

from taskview_ai.config import get_settings
from taskview_ai.main import app


def test_health_and_plan(monkeypatch):
    monkeypatch.setenv("TASKVIEW_AI_FAKE_MODE", "true")
    get_settings.cache_clear()
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["provider"] == "fake"

    response = client.post(
        "/v1/agent/plan",
        json={
            "purpose": "VOC를 지역과 이슈별로 묶어 다음 스프린트 우선순위를 정하고 싶다",
            "audience": "product",
            "ttl_days": 7,
        },
    )
    assert response.status_code == 200
    plan = response.json()
    assert plan["selected_sources"] == ["voc"]
    assert "ticket_id" not in plan["preview_columns"]
    assert plan["needs_owner_approval"] is True

    get_settings.cache_clear()

