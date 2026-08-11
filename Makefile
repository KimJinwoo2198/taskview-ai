.PHONY: install dev test lint model eval

install:
	uv sync

dev:
	uv run uvicorn taskview_ai.main:app --reload --host 0.0.0.0 --port $${PORT:-8100}

test:
	uv run pytest -q

lint:
	uv run ruff check .

model:
	ollama pull $${OLLAMA_MODEL:-qwen3.5:9b}

eval:
	uv run python scripts/evaluate.py
