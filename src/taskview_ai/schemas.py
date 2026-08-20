from typing import Literal

from pydantic import BaseModel, Field

Audience = Literal["product", "operations", "support", "executive"]


class PlanRequest(BaseModel):
    purpose: str = Field(min_length=10, max_length=1000)
    audience: Audience = "product"
    ttl_days: int = Field(default=7, ge=1, le=30)


class BusinessIntent(BaseModel):
    summary: str = Field(min_length=10, max_length=180)
    subject: str = Field(min_length=2, max_length=80)
    comparison_dimensions: list[str] = Field(min_length=1, max_length=4)
    desired_outcome: str = Field(min_length=4, max_length=100)
    region_label: str = Field(min_length=2, max_length=40)
    department: Literal["product", "operations", "support", "executive"]
    selected_source: Literal["product", "operations", "voc"]
    confidence: float = Field(ge=0, le=1)
    needs_clarification: bool = False
    clarifying_question: str | None = Field(default=None, max_length=120)


class PurposeSpec(BaseModel):
    objective: str
    decision_to_support: str
    audience: Audience
    requested_fields: list[str]


class TransformPlanItem(BaseModel):
    source: Literal["product", "operations", "voc"]
    input_fields: list[str]
    output_field: str
    transformation: Literal[
        "select", "drop", "mask", "age_band", "region_group", "aggregate", "classify"
    ]
    rationale: str


class ViewPlan(BaseModel):
    purpose_spec: PurposeSpec
    selected_sources: list[Literal["product", "operations", "voc"]]
    transformations: list[TransformPlanItem]
    preview_columns: list[str]
    assumptions: list[str] = []
    needs_owner_approval: bool = True


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    provider: Literal["ollama", "fake"]
    model: str
