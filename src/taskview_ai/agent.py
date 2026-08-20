import asyncio
from collections import OrderedDict

from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from agents.exceptions import ModelBehaviorError
from openai import AsyncOpenAI

from .config import Settings
from .schemas import BusinessIntent, PlanRequest, PurposeSpec, TransformPlanItem, ViewPlan
from .tools import CATALOG

INSTRUCTIONS = """
당신은 여러 부서가 함께 쓰는 Needex의 업무 분석 도우미다. 사용자의 말을 쉬운 한국어로 정리한다.

반드시 지킬 규칙:
1. 사용자가 말하지 않은 사실, 숫자, 지역, 대상을 만들지 않는다.
2. summary, subject, comparison_dimensions, desired_outcome, region_label은 업무 담당자가 바로 이해할 수 있는 자연스러운 한국어로 쓴다.
3. selected_source는 product, operations, voc 중 하나다. 통신·FCC·소비자 불만은 product, 311·도시 운영·처리시간은 operations, 차량·안전·사고·NHTSA는 voc를 고른다.
4. desired_outcome은 실제 의사결정을 표현하고 '~를 정한다' 또는 '~를 결정한다'로 끝낸다.
5. 목적이 너무 모호하면 needs_clarification을 true로 하고, clarifying_question에 한 번에 답할 수 있는 질문 하나만 쓴다.
6. Semantic, schema, policy engine, utility, inference 같은 내부 기술 용어를 사용하지 않는다.
7. SQL, 개인정보 필드, 승인 여부를 만들지 않는다. 안전한 데이터 선택과 변환은 별도의 결정론적 정책이 담당한다.
""".strip()

_INTENT_CACHE: OrderedDict[tuple[str, str], BusinessIntent] = OrderedDict()
_INTENT_CACHE_MAX_SIZE = 128
_FALLBACK_KEYS: set[tuple[str, str]] = set()


def _fallback_intent(request: PlanRequest) -> BusinessIntent:
    purpose = request.purpose.strip()
    normalized = purpose.casefold()
    if any(keyword in normalized for keyword in ("311", "민원", "도시 운영", "처리시간")):
        source = "operations"
        subject = "도시 민원 처리"
        dimensions = ["지역", "담당 기관", "민원 유형"]
        region = "뉴욕시 전체" if "311" in normalized or "뉴욕" in normalized else "요청 지역"
        outcome = "민원 대응 우선순위를 정한다"
    elif any(
        keyword in normalized for keyword in ("nhtsa", "차량", "자동차", "안전", "사고", "화재")
    ):
        source = "voc"
        subject = "차량 안전 신고"
        dimensions = ["제조사", "연식", "문제 부위"]
        region = "미국 전체"
        outcome = "안전 조사 우선순위를 정한다"
    elif any(keyword in normalized for keyword in ("voc", "상담", "고객 문의", "고객지원")):
        source = "voc"
        subject = "고객 문의"
        dimensions = ["지역", "문의 유형", "기간"]
        region = "요청 지역"
        outcome = "고객지원 개선 우선순위를 정한다"
    else:
        source = "product"
        subject = "소비자 불만"
        dimensions = ["지역", "불만 유형", "접수 방법"]
        region = "미국 전체"
        outcome = "업무 개선 우선순위를 정한다"

    vague = len(purpose) < 16 or purpose in {"분석해줘", "데이터가 필요해", "현황을 보고 싶어"}
    return BusinessIntent(
        summary=purpose,
        subject=subject,
        comparison_dimensions=dimensions,
        desired_outcome=outcome,
        region_label=region,
        department=request.audience,
        selected_source=source,
        confidence=0.45 if vague else 0.72,
        needs_clarification=vague,
        clarifying_question=(
            "어떤 결정을 내리기 위해 어떤 대상이나 기간을 비교하고 싶으신가요?" if vague else None
        ),
    )


def _cache_key(request: PlanRequest) -> tuple[str, str]:
    return (" ".join(request.purpose.casefold().split()), request.audience)


def _remember_intent(key: tuple[str, str], intent: BusinessIntent) -> BusinessIntent:
    _INTENT_CACHE[key] = intent
    _INTENT_CACHE.move_to_end(key)
    while len(_INTENT_CACHE) > _INTENT_CACHE_MAX_SIZE:
        _INTENT_CACHE.popitem(last=False)
    return intent


def _enforce_source_route(request: PlanRequest, intent: BusinessIntent) -> BusinessIntent:
    """Keep model wording, but make source routing deterministic and auditable."""
    normalized = request.purpose.casefold()
    if any(keyword in normalized for keyword in ("fcc", "통신위원회")):
        source = "product"
    elif any(keyword in normalized for keyword in ("311", "도시 운영", "처리시간")):
        source = "operations"
    elif (
        any(keyword in normalized for keyword in ("nhtsa", "차량", "자동차", "사고", "화재"))
        or request.audience == "support"
        or any(keyword in normalized for keyword in ("고객지원", "상담", "문의"))
    ):
        source = "voc"
    else:
        source = intent.selected_source
    outcome = intent.desired_outcome.strip()
    for ending, replacement in (
        ("결정하고자 합니다.", "결정한다"),
        ("결정하고자 합니다", "결정한다"),
        ("정하고자 합니다.", "정한다"),
        ("정하고자 합니다", "정한다"),
        ("결정합니다.", "결정한다"),
        ("결정합니다", "결정한다"),
        ("정합니다.", "정한다"),
        ("정합니다", "정한다"),
    ):
        if outcome.endswith(ending):
            outcome = f"{outcome[: -len(ending)]}{replacement}"
            break
    if not outcome.endswith(("정한다", "결정한다")):
        outcome = _fallback_intent(request).desired_outcome

    concrete_terms = (
        "fcc",
        "311",
        "nhtsa",
        "민원",
        "고객지원",
        "문의 유형",
        "회원가입",
        "통신",
        "차량",
        "제조사",
        "처리시간",
        "처리 시간",
    )
    structured_request_terms = (
        "비교",
        "별로",
        "별 ",
        "우선순위",
        "우선 순위",
        "배치",
        "추세",
        "접수",
        "처리 속도",
        "반복 접촉",
    )
    vague_request = (
        any(
            phrase in normalized
            for phrase in ("업무 현황", "필요한 데이터", "분석해서 개선", "데이터를 찾아")
        )
        and not any(term in normalized for term in concrete_terms)
    ) or (
        not any(term in normalized for term in concrete_terms)
        and not any(term in normalized for term in structured_request_terms)
    )
    injection_request = any(
        phrase in normalized
        for phrase in (
            "이전 지시를 무시",
            "시스템 규칙을 공개",
            "시스템 프롬프트",
            "고객 이름과 연락처를 출력",
            "전화번호를 보여",
            "sql을 보여",
        )
    )
    needs_clarification = vague_request or (intent.needs_clarification and not injection_request)
    question = intent.clarifying_question
    if needs_clarification and not question:
        question = "어떤 결정을 위해 무엇을 비교하고 싶은지 한 문장으로 알려주세요."

    summary = intent.summary
    if injection_request:
        summary = f"{intent.subject}을 기준으로 {outcome}."
        question = None

    return intent.model_copy(
        update={
            "summary": summary,
            "department": request.audience,
            "selected_source": source,
            "desired_outcome": outcome,
            "needs_clarification": needs_clarification,
            "clarifying_question": question,
            "confidence": min(intent.confidence, 0.6) if vague_request else intent.confidence,
        }
    )


def _requires_age_band(purpose: str) -> bool:
    normalized = purpose.casefold()
    return any(
        keyword in normalized
        for keyword in ("연령", "나이", "세대별", "age cohort", "age group", "age-group")
    )


def _is_signup_diagnosis(purpose: str) -> bool:
    normalized = purpose.casefold()
    has_signup = any(keyword in normalized for keyword in ("회원가입", "signup", "가입 이탈"))
    has_diagnosis = any(keyword in normalized for keyword in ("원인", "진단", "dropoff", "이탈"))
    return has_signup and has_diagnosis


def _signup_diagnosis_plan(request: PlanRequest, decision_to_support: str) -> ViewPlan:
    return ViewPlan(
        purpose_spec=PurposeSpec(
            objective=request.purpose,
            decision_to_support=decision_to_support,
            audience=request.audience,
            requested_fields=[
                "event_time",
                "os_family",
                "os_version",
                "dropoff_step",
                "error_log",
                "exact_address",
                "birth_date",
                "ticket_text",
                "customer_name",
                "phone",
                "email",
            ],
        ),
        selected_sources=["product", "operations", "voc"],
        transformations=[
            TransformPlanItem(
                source="product",
                input_fields=["event_time"],
                output_field="week",
                transformation="aggregate",
                rationale="개별 이벤트 시각 대신 주 단위만 유지",
            ),
            TransformPlanItem(
                source="product",
                input_fields=["os_family"],
                output_field="os_family",
                transformation="select",
                rationale="OS 계열별 이탈 차이를 비교하는 최소 차원",
            ),
            TransformPlanItem(
                source="product",
                input_fields=["os_version"],
                output_field="os_version",
                transformation="select",
                rationale="세부 빌드 대신 OS 버전 계열만 유지",
            ),
            TransformPlanItem(
                source="product",
                input_fields=["dropoff_step"],
                output_field="signup_step",
                transformation="select",
                rationale="회원가입 단계별 이탈 위치를 비교",
            ),
            TransformPlanItem(
                source="product",
                input_fields=["error_log"],
                output_field="error_category",
                transformation="classify",
                rationale="오류 원문 대신 검증된 오류 범주만 제공",
            ),
            TransformPlanItem(
                source="operations",
                input_fields=["exact_address"],
                output_field="region_group",
                transformation="region_group",
                rationale="정확한 주소를 권역으로 일반화",
            ),
            TransformPlanItem(
                source="operations",
                input_fields=["birth_date"],
                output_field="age_band",
                transformation="age_band",
                rationale="생년월일 대신 연령대만 제공",
            ),
            TransformPlanItem(
                source="voc",
                input_fields=["ticket_text"],
                output_field="complaint_theme",
                transformation="classify",
                rationale="상담 원문 대신 불만 주제만 추출",
            ),
            TransformPlanItem(
                source="voc",
                input_fields=["customer_name"],
                output_field="customer_name",
                transformation="drop",
                rationale="직접 식별자는 목적에 필요하지 않음",
            ),
            TransformPlanItem(
                source="voc",
                input_fields=["phone", "email"],
                output_field="contact",
                transformation="drop",
                rationale="연락처는 Task View에서 제외",
            ),
        ],
        preview_columns=[
            "week",
            "region_group",
            "age_band",
            "os_family",
            "os_version",
            "signup_step",
            "error_category",
            "complaint_theme",
            "case_count",
        ],
        assumptions=[
            f"View는 {request.ttl_days}일 뒤 만료된다",
            "세 소스의 집계 그룹은 20건 이상이어야 한다",
            "원문과 직접 식별자는 어떤 출력에서도 반환하지 않는다",
        ],
    )


def _safe_plan(
    request: PlanRequest,
    *,
    selected_source: str,
    decision_to_support: str,
) -> ViewPlan:
    if _is_signup_diagnosis(request.purpose):
        return _signup_diagnosis_plan(request, "회원가입 이탈의 상위 원인을 정한다")
    if _requires_age_band(request.purpose):
        return ViewPlan(
            purpose_spec=PurposeSpec(
                objective=request.purpose,
                decision_to_support=decision_to_support,
                audience=request.audience,
                requested_fields=["created_at", "address", "age", "message", "ticket_id"],
            ),
            selected_sources=["voc"],
            transformations=[
                TransformPlanItem(
                    source="voc",
                    input_fields=["created_at"],
                    output_field="week",
                    transformation="aggregate",
                    rationale="주간 단위로 집계",
                ),
                TransformPlanItem(
                    source="voc",
                    input_fields=["address"],
                    output_field="region",
                    transformation="region_group",
                    rationale="지역 수준으로 일반화",
                ),
                TransformPlanItem(
                    source="voc",
                    input_fields=["age"],
                    output_field="age_band",
                    transformation="age_band",
                    rationale="정확한 나이 대신 연령대만 제공",
                ),
                TransformPlanItem(
                    source="voc",
                    input_fields=["message"],
                    output_field="issue_type",
                    transformation="classify",
                    rationale="원문 대신 이슈 유형만 제공",
                ),
                TransformPlanItem(
                    source="voc",
                    input_fields=["ticket_id"],
                    output_field="ticket_id",
                    transformation="drop",
                    rationale="직접 식별자를 제외",
                ),
            ],
            preview_columns=["week", "region", "age_band", "issue_type", "case_count"],
            assumptions=["공개 데이터에는 연령 필드가 없어 연결된 조직 데이터가 필요하다"],
        )
    return _public_plan(
        request,
        selected_source=selected_source,
        decision_to_support=decision_to_support,
    )


def _public_plan(
    request: PlanRequest, *, selected_source: str, decision_to_support: str
) -> ViewPlan:
    if selected_source == "product":
        return ViewPlan(
            purpose_spec=PurposeSpec(
                objective=request.purpose,
                decision_to_support=decision_to_support,
                audience=request.audience,
                requested_fields=[
                    "ticket_created",
                    "state",
                    "issue_type",
                    "issue",
                    "method",
                    "caller_id_number",
                ],
            ),
            selected_sources=["product"],
            transformations=[
                TransformPlanItem(
                    source="product",
                    input_fields=["ticket_created"],
                    output_field="week",
                    transformation="aggregate",
                    rationale="접수 시각을 주 단위로 집계",
                ),
                TransformPlanItem(
                    source="product",
                    input_fields=["state"],
                    output_field="region",
                    transformation="region_group",
                    rationale="상세 위치 없이 주 수준만 유지",
                ),
                TransformPlanItem(
                    source="product",
                    input_fields=["issue_type"],
                    output_field="issue_type",
                    transformation="select",
                    rationale="불만 유형별 개선 우선순위 비교",
                ),
                TransformPlanItem(
                    source="product",
                    input_fields=["method"],
                    output_field="channel",
                    transformation="select",
                    rationale="접수 채널별 차이 비교",
                ),
                TransformPlanItem(
                    source="product",
                    input_fields=["caller_id_number"],
                    output_field="caller_id_number",
                    transformation="drop",
                    rationale="전화번호는 수집·출력하지 않음",
                ),
            ],
            preview_columns=["week", "region", "issue_type", "channel", "case_count"],
            assumptions=[
                f"View는 {request.ttl_days}일 뒤 만료된다",
                "FCC 공식 공개 데이터에서 20건 이상 그룹만 제공한다",
            ],
        )
    if selected_source == "operations":
        return ViewPlan(
            purpose_spec=PurposeSpec(
                objective=request.purpose,
                decision_to_support=decision_to_support,
                audience=request.audience,
                requested_fields=[
                    "created_date",
                    "borough",
                    "agency",
                    "complaint_type",
                    "status",
                    "resolution_hours",
                    "incident_address",
                    "latitude",
                    "longitude",
                ],
            ),
            selected_sources=["operations"],
            transformations=[
                TransformPlanItem(
                    source="operations",
                    input_fields=["created_date"],
                    output_field="week",
                    transformation="aggregate",
                    rationale="민원 접수를 주 단위로 집계",
                ),
                TransformPlanItem(
                    source="operations",
                    input_fields=["borough"],
                    output_field="region",
                    transformation="region_group",
                    rationale="borough 수준으로만 일반화",
                ),
                TransformPlanItem(
                    source="operations",
                    input_fields=["agency"],
                    output_field="agency",
                    transformation="select",
                    rationale="담당 기관별 병목 비교",
                ),
                TransformPlanItem(
                    source="operations",
                    input_fields=["complaint_type"],
                    output_field="complaint_type",
                    transformation="select",
                    rationale="민원 유형별 운영 수요 비교",
                ),
                TransformPlanItem(
                    source="operations",
                    input_fields=["resolution_hours"],
                    output_field="avg_resolution_hours",
                    transformation="aggregate",
                    rationale="개별 민원이 아닌 평균 처리시간 제공",
                ),
                TransformPlanItem(
                    source="operations",
                    input_fields=["incident_address", "latitude", "longitude"],
                    output_field="precise_location",
                    transformation="drop",
                    rationale="정확한 위치는 수집·출력하지 않음",
                ),
            ],
            preview_columns=[
                "week",
                "region",
                "agency",
                "complaint_type",
                "avg_resolution_hours",
                "case_count",
            ],
            assumptions=[
                f"View는 {request.ttl_days}일 뒤 만료된다",
                "NYC 공식 공개 데이터에서 20건 이상 그룹만 제공한다",
            ],
        )
    return ViewPlan(
        purpose_spec=PurposeSpec(
            objective=request.purpose,
            decision_to_support=decision_to_support,
            audience=request.audience,
            requested_fields=[
                "date_complaint_filed",
                "manufacturer",
                "model_year",
                "component",
                "crash",
                "fire",
                "vin",
                "summary",
            ],
        ),
        selected_sources=["voc"],
        transformations=[
            TransformPlanItem(
                source="voc",
                input_fields=["date_complaint_filed"],
                output_field="week",
                transformation="aggregate",
                rationale="접수일을 주 단위로 집계",
            ),
            TransformPlanItem(
                source="voc",
                input_fields=["manufacturer"],
                output_field="manufacturer",
                transformation="select",
                rationale="제조사별 안전 신호 비교",
            ),
            TransformPlanItem(
                source="voc",
                input_fields=["model_year"],
                output_field="model_year",
                transformation="select",
                rationale="연식별 위험 신호 비교",
            ),
            TransformPlanItem(
                source="voc",
                input_fields=["component"],
                output_field="component",
                transformation="select",
                rationale="부품군별 안전 이슈 비교",
            ),
            TransformPlanItem(
                source="voc",
                input_fields=["crash"],
                output_field="crash_count",
                transformation="aggregate",
                rationale="사고 보고 건수를 집계",
            ),
            TransformPlanItem(
                source="voc",
                input_fields=["fire"],
                output_field="fire_count",
                transformation="aggregate",
                rationale="화재 보고 건수를 집계",
            ),
            TransformPlanItem(
                source="voc",
                input_fields=["vin", "summary"],
                output_field="raw_complaint",
                transformation="drop",
                rationale="VIN과 원문은 수집·출력하지 않음",
            ),
        ],
        preview_columns=[
            "manufacturer",
            "model_year",
            "component",
            "crash_count",
            "fire_count",
            "case_count",
        ],
        assumptions=[
            f"View는 {request.ttl_days}일 뒤 만료된다",
            "NHTSA 공식 공개 데이터에서 20건 이상 그룹만 제공한다",
        ],
    )
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

        if _requires_age_band(request.purpose):
            requested_fields.insert(2, "age")
            transformations.insert(
                2,
                TransformPlanItem(
                    source="voc",
                    input_fields=["age"],
                    output_field="age_band",
                    transformation="age_band",
                    rationale="정확한 나이 대신 목적에 필요한 연령대 구간만 제공",
                ),
            )
            preview_columns.insert(2, "age_band")

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


async def build_business_intent(request: PlanRequest, settings: Settings) -> BusinessIntent:
    key = _cache_key(request)
    cached = _INTENT_CACHE.get(key)
    if cached is not None:
        _INTENT_CACHE.move_to_end(key)
        return cached

    if settings.taskview_ai_fake_mode:
        return _fallback_intent(request)

    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=settings.ollama_base_url, api_key="ollama-local")
    model = OpenAIChatCompletionsModel(model=settings.ollama_model, openai_client=client)
    agent = Agent(
        name="Needex Business Intent Analyzer",
        instructions=INSTRUCTIONS,
        model=model,
        model_settings=ModelSettings(
            temperature=0,
            max_tokens=360,
            parallel_tool_calls=False,
            extra_body={"reasoning_effort": "none"},
        ),
        output_type=BusinessIntent,
    )
    try:
        async with asyncio.timeout(20):
            result = await Runner.run(
                agent,
                input=(
                    f"목적: {request.purpose}\n"
                    f"요청 부서: {request.audience}\n"
                    "요청 내용을 업무 담당자가 확인할 수 있는 쉬운 한국어 구조로 정리하세요."
                ),
                max_turns=1,
            )
    except (ModelBehaviorError, TimeoutError):
        _FALLBACK_KEYS.add(key)
        return _remember_intent(key, _fallback_intent(request))
    intent = result.final_output
    if not isinstance(intent, BusinessIntent):
        return _remember_intent(key, _fallback_intent(request))
    return _remember_intent(key, _enforce_source_route(request, intent))


async def build_view_plan(request: PlanRequest, settings: Settings) -> ViewPlan:
    if settings.taskview_ai_fake_mode:
        return _fake_plan(request)
    intent = await build_business_intent(request, settings)
    plan = _safe_plan(
        request,
        selected_source=intent.selected_source,
        decision_to_support=intent.desired_outcome,
    )
    if _is_catalog_safe(plan):
        if _cache_key(request) in _FALLBACK_KEYS:
            plan.assumptions.insert(
                0,
                "로컬 모델의 구조화 출력 검증에 실패해 안전한 기본 계획을 적용했다",
            )
        return plan

    safe_plan = _fake_plan(request)
    safe_plan.assumptions.insert(
        0,
        "로컬 모델 출력이 데이터 카탈로그 검증에 실패해 보수적인 최소 계획을 적용했다",
    )
    return safe_plan
