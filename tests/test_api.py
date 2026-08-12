import asyncio

from agents import Runner
from agents.exceptions import ModelBehaviorError
from fastapi.testclient import TestClient

from taskview_ai.agent import _is_catalog_safe, _safe_plan, build_view_plan
from taskview_ai.config import Settings, get_settings
from taskview_ai.main import app
from taskview_ai.schemas import PlanRequest, PurposeSpec, TransformPlanItem, ViewPlan


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


def test_model_behavior_error_uses_safe_plan(monkeypatch):
    async def invalid_structured_output(*_args, **_kwargs):
        raise ModelBehaviorError("model returned fenced JSON with the wrong schema")

    monkeypatch.setattr(Runner, "run", invalid_structured_output)
    request = PlanRequest(
        purpose="VOC를 지역과 이슈별로 묶어 다음 스프린트 우선순위를 정하고 싶다",
        audience="product",
        ttl_days=7,
    )

    plan = asyncio.run(
        build_view_plan(request, Settings(taskview_ai_fake_mode=False))
    )

    assert _is_catalog_safe(plan) is True
    assert "구조화 출력 검증에 실패" in plan.assumptions[0]
    assert "message" not in plan.preview_columns


def test_age_band_is_added_only_for_age_cohort_purposes():
    age_request = PlanRequest(
        purpose="연령대별 고객지원 문의 유형을 비교해 콘텐츠 순서를 정하고 싶다",
        audience="support",
        ttl_days=3,
    )
    regular_request = PlanRequest(
        purpose="고객지원 문의 유형을 비교해 콘텐츠 순서를 정하고 싶다",
        audience="support",
        ttl_days=3,
    )

    age_plan = _safe_plan(
        age_request,
        selected_source="voc",
        decision_to_support="도움말 콘텐츠 순서를 정한다",
    )
    regular_plan = _safe_plan(
        regular_request,
        selected_source="voc",
        decision_to_support="도움말 콘텐츠 순서를 정한다",
    )

    age_transform = next(
        item for item in age_plan.transformations if "age" in item.input_fields
    )
    assert age_transform.transformation == "age_band"
    assert age_transform.output_field == "age_band"
    assert "age_band" in age_plan.preview_columns
    assert "age" not in age_plan.preview_columns
    assert all("age" not in item.input_fields for item in regular_plan.transformations)
