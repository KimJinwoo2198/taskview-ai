from typing import Literal

from pydantic import BaseModel, Field

Audience = Literal["product", "operations", "support", "executive"]


class PlanRequest(BaseModel):
    purpose: str = Field(min_length=10, max_length=1000)
    audience: Audience = "product"
    ttl_days: int = Field(default=7, ge=1, le=30)


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

