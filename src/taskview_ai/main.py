import secrets

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .agent import build_business_intent, build_view_plan
from .config import get_settings
from .schemas import BusinessIntent, HealthResponse, PlanRequest, ViewPlan

app = FastAPI(title="Needex AI", version="0.1.0")


@app.middleware("http")
async def protect_agent_api(request: Request, call_next):
    settings = get_settings()
    if request.url.path.startswith("/v1/agent/") and settings.taskview_ai_shared_secret:
        expected = f"Bearer {settings.taskview_ai_shared_secret}"
        provided = request.headers.get("authorization", "")
        if not secrets.compare_digest(provided, expected):
            return JSONResponse({"detail": "유효한 AI 서비스 인증이 필요합니다."}, status_code=401)
    return await call_next(request)


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


@app.post("/v1/agent/interpret", response_model=BusinessIntent)
async def interpret_purpose(request: PlanRequest) -> BusinessIntent:
    try:
        return await build_business_intent(request, get_settings())
    except Exception as exc:
        raise HTTPException(status_code=503, detail="업무 목적을 분석하지 못했습니다.") from exc
