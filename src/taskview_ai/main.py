from fastapi import FastAPI, HTTPException

from .agent import build_view_plan
from .config import get_settings
from .schemas import HealthResponse, PlanRequest, ViewPlan

app = FastAPI(title="TaskView AI", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        provider="fake" if settings.taskview_ai_fake_mode else "ollama",
        model=settings.ollama_model,
    )


@app.post("/v1/agent/plan", response_model=ViewPlan)
async def create_plan(request: PlanRequest) -> ViewPlan:
    try:
        return await build_view_plan(request, get_settings())
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="로컬 LLM에 연결하지 못했습니다. Ollama 실행 및 모델 설치 상태를 확인하세요.",
        ) from exc

