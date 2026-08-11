from fastapi.testclient import TestClient

from taskview_ai.agent import _is_catalog_safe
from taskview_ai.config import get_settings
from taskview_ai.main import app
from taskview_ai.schemas import PurposeSpec, TransformPlanItem, ViewPlan


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


def test_catalog_guard_rejects_hallucinated_plan():
    plan = ViewPlan(
        purpose_spec=PurposeSpec(
            objective="VOC 우선순위를 결정한다",
            decision_to_support="스프린트 우선순위 결정",
            audience="product",
            requested_fields=["invented_field"],
        ),
        selected_sources=["voc"],
        transformations=[
            TransformPlanItem(
                source="voc",
                input_fields=["invented_field"],
                output_field="invented_summary",
                transformation="aggregate",
                rationale="모델이 생성한 존재하지 않는 필드",
            )
        ],
        preview_columns=["invented_preview"],
    )

    assert _is_catalog_safe(plan) is False
