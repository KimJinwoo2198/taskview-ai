import asyncio

from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from agents.exceptions import ModelBehaviorError
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .config import Settings
from .schemas import PlanRequest, PurposeSpec, TransformPlanItem, ViewPlan
from .tools import CATALOG

INSTRUCTIONS = """
당신은 TaskView의 목적 분석 Agent다. 목적을 짧은 의사결정 문장과 승인된 데이터 소스 하나로 분류한다.

반드시 지킬 규칙:
1. selected_source는 product, operations, voc 중 정확히 하나다.
2. 제품 사용량·기능 지표 목적은 product, 운영 티켓·처리시간 목적은 operations, 고객 의견·불만 목적은 voc를 고른다.
3. decision_to_support는 데이터 소스 이름을 언급하지 않고 반드시 '~를 정한다' 또는 '~를 결정한다'로 끝낸다.
4. 다른 키, 설명, 마크다운을 추가하지 않는다.
5. 정확한 JSON 형식만 반환한다: {"decision_to_support":"제품 개선 우선순위를 정한다","selected_source":"voc"}
6. 실제 SQL, 개인정보 필드, 승인 여부는 만들지 않는다. 안전한 필드와 변환은 결정론적 정책 계층이 적용한다.
""".strip()


class IntentAnalysis(BaseModel):
    decision_to_support: str = Field(min_length=5, max_length=120)
    selected_source: str = Field(pattern="^(product|operations|voc)$")


def _safe_plan(
    request: PlanRequest,
    *,
    selected_source: str,
    decision_to_support: str,
) -> ViewPlan:
    if selected_source == "product":
        requested_fields = ["event_date", "feature", "usage_count", "account_id", "user_id"]
        transformations = [
            TransformPlanItem(
                source="product",
                input_fields=["event_date"],
                output_field="week",
                transformation="aggregate",
                rationale="개별 활동 시각 대신 주간 단위만 유지",
            ),
            TransformPlanItem(
                source="product",
                input_fields=["feature"],
                output_field="feature",
                transformation="select",
                rationale="기능별 사용량 비교에 필요한 차원",
            ),
            TransformPlanItem(
                source="product",
                input_fields=["usage_count"],
                output_field="usage_count",
                transformation="aggregate",
                rationale="개별 사용 기록 대신 집계 사용량만 제공",
            ),
            TransformPlanItem(
                source="product",
                input_fields=["account_id"],
                output_field="account_segment",
                transformation="mask",
                rationale="계정 식별자는 복원할 수 없는 구간 값으로 대체",
            ),
            TransformPlanItem(
                source="product",
                input_fields=["user_id"],
                output_field="user_id",
                transformation="drop",
                rationale="개인 식별자는 목적 달성에 필요하지 않음",
            ),
        ]
        preview_columns = ["week", "feature", "usage_count", "case_count"]
    elif selected_source == "operations":
        requested_fields = ["created_at", "region", "status", "resolution_hours", "ticket_id"]
        transformations = [
            TransformPlanItem(
                source="operations",
                input_fields=["created_at"],
                output_field="week",
                transformation="aggregate",
                rationale="운영 추세 비교를 위해 주간 단위만 유지",
            ),
            TransformPlanItem(
                source="operations",
                input_fields=["region"],
                output_field="region",
                transformation="select",
                rationale="지역별 운영 차이를 비교하는 최소 차원",
            ),
            TransformPlanItem(
                source="operations",
                input_fields=["status"],
                output_field="status",
                transformation="select",
                rationale="처리 상태별 병목을 구분하는 최소 필드",
            ),
            TransformPlanItem(
                source="operations",
                input_fields=["resolution_hours"],
                output_field="avg_resolution_hours",
                transformation="aggregate",
                rationale="개별 티켓 대신 평균 처리시간만 제공",
            ),
            TransformPlanItem(
                source="operations",
                input_fields=["ticket_id"],
                output_field="ticket_id",
                transformation="drop",
                rationale="직접 식별자는 운영 비교에 필요하지 않음",
            ),
        ]
        preview_columns = ["week", "region", "status", "avg_resolution_hours", "case_count"]
    else:
        selected_source = "voc"
        requested_fields = ["created_at", "address", "message", "ticket_id"]
        transformations = [
            TransformPlanItem(
                source="voc",
                input_fields=["created_at"],
                output_field="week",
                transformation="aggregate",
                rationale="주간 추세 비교에 필요한 시간 단위만 유지",
            ),
            TransformPlanItem(
                source="voc",
                input_fields=["address"],
                output_field="region",
                transformation="region_group",
                rationale="정확한 주소를 노출하지 않고 지역 수준으로 축약",
            ),
            TransformPlanItem(
                source="voc",
                input_fields=["message"],
                output_field="issue_type",
                transformation="classify",
                rationale="VOC 원문 대신 업무에 필요한 이슈 유형만 제공",
            ),
            TransformPlanItem(
                source="voc",
                input_fields=["ticket_id"],
                output_field="ticket_id",
                transformation="drop",
                rationale="제품 우선순위 판단에 직접 식별자는 불필요",
            ),
        ]
        preview_columns = ["week", "region", "issue_type", "case_count"]

    return ViewPlan(
        purpose_spec=PurposeSpec(
            objective=request.purpose,
            decision_to_support=decision_to_support,
            audience=request.audience,
            requested_fields=requested_fields,
        ),
        selected_sources=[selected_source],
        transformations=transformations,
        preview_columns=preview_columns,
        assumptions=[
            f"View는 {request.ttl_days}일 뒤 만료된다",
            "집계 그룹은 20건 이상이어야 한다",
        ],
    )


def _fake_plan(request: PlanRequest) -> ViewPlan:
    return _safe_plan(
        request,
        selected_source="voc",
        decision_to_support="다음 스프린트의 개선 우선순위를 정한다",
    )


def _is_catalog_safe(plan: ViewPlan) -> bool:
    if not plan.selected_sources:
        return False
    produced_columns = {
        item.output_field for item in plan.transformations if item.transformation != "drop"
    } | {"case_count"}
    if not set(plan.preview_columns).issubset(produced_columns):
        return False
    for item in plan.transformations:
        if item.source not in plan.selected_sources:
            return False
        catalog_fields = set(CATALOG[item.source]["fields"])
        if not set(item.input_fields).issubset(catalog_fields):
            return False
    return True


async def build_view_plan(request: PlanRequest, settings: Settings) -> ViewPlan:
    if settings.taskview_ai_fake_mode:
        return _fake_plan(request)

    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=settings.ollama_base_url, api_key="ollama-local")
    model = OpenAIChatCompletionsModel(model=settings.ollama_model, openai_client=client)
    agent = Agent(
        name="TaskView Intent Analyzer",
        instructions=INSTRUCTIONS,
        model=model,
        model_settings=ModelSettings(
            temperature=0,
            max_tokens=120,
            parallel_tool_calls=False,
            extra_body={"reasoning_effort": "none"},
        ),
        output_type=IntentAnalysis,
    )
    try:
        async with asyncio.timeout(15):
            result = await Runner.run(
                agent,
                input=(
                    f"목적: {request.purpose}\n"
                    f"대상 사용자: {request.audience}\n"
                    f"정확히 두 키의 JSON만 반환하세요."
                ),
                max_turns=1,
            )
    except (ModelBehaviorError, TimeoutError):
        safe_plan = _fake_plan(request)
        safe_plan.assumptions.insert(
            0,
            "로컬 모델의 구조화 출력 검증에 실패해 보수적인 VOC 최소 계획을 적용했다",
        )
        return safe_plan
    intent = result.final_output
    plan = _safe_plan(
        request,
        selected_source=intent.selected_source,
        decision_to_support=intent.decision_to_support,
    )
    if _is_catalog_safe(plan):
        return plan

    safe_plan = _fake_plan(request)
    safe_plan.assumptions.insert(
        0,
        "로컬 모델 출력이 데이터 카탈로그 검증에 실패해 보수적인 최소 계획을 적용했다",
    )
    return safe_plan
