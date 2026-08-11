from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from openai import AsyncOpenAI

from .config import Settings
from .schemas import PlanRequest, PurposeSpec, TransformPlanItem, ViewPlan
from .tools import get_privacy_transform, search_data_catalog

INSTRUCTIONS = """
당신은 TaskView의 단일 오케스트레이터 Agent다. 사용자의 업무 목적을 실행 가능한 View 계획으로 바꾼다.

반드시 지킬 규칙:
1. search_data_catalog로 존재하는 소스와 필드를 확인한다.
2. 직접 식별자는 drop, 계정 식별자는 mask, 주소는 region_group, 나이는 age_band로 제안한다.
3. VOC 원문 message는 제품 분석 목적일 때 그대로 노출하지 말고 issue_type으로 classify한다.
4. 실제 SQL을 만들거나 실행하지 않는다. 접근 허용, 승인, materialization은 BE의 책임이다.
5. 최소 필드만 선택하고 모든 변환에 짧고 구체적인 rationale을 쓴다.
6. 결과는 요청된 ViewPlan 스키마를 정확히 따른다.
""".strip()


def _fake_plan(request: PlanRequest) -> ViewPlan:
    return ViewPlan(
        purpose_spec=PurposeSpec(
            objective=request.purpose,
            decision_to_support="다음 스프린트의 개선 우선순위를 정한다",
            audience=request.audience,
            requested_fields=["created_at", "address", "message", "ticket_id"],
        ),
        selected_sources=["voc"],
        transformations=[
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
        ],
        preview_columns=["week", "region", "issue_type", "case_count"],
        assumptions=[f"View는 {request.ttl_days}일 뒤 만료된다", "집계 그룹은 20건 이상이어야 한다"],
    )


async def build_view_plan(request: PlanRequest, settings: Settings) -> ViewPlan:
    if settings.taskview_ai_fake_mode:
        return _fake_plan(request)

    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=settings.ollama_base_url, api_key="ollama-local")
    model = OpenAIChatCompletionsModel(model=settings.ollama_model, openai_client=client)
    agent = Agent(
        name="TaskView Orchestrator",
        instructions=INSTRUCTIONS,
        model=model,
        tools=[search_data_catalog, get_privacy_transform],
        output_type=ViewPlan,
    )
    result = await Runner.run(
        agent,
        input=(
            f"목적: {request.purpose}\n"
            f"대상 사용자: {request.audience}\n"
            f"요청 TTL: {request.ttl_days}일"
        ),
        max_turns=6,
    )
    return result.final_output

