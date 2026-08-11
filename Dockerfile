FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --frozen --no-dev || uv sync --no-dev
COPY src ./src
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src PORT=8100
EXPOSE 8100
CMD ["sh", "-c", "uvicorn taskview_ai.main:app --host 0.0.0.0 --port ${PORT}"]

