import asyncio
import json
from pathlib import Path

from taskview_ai.agent import build_view_plan
from taskview_ai.config import Settings
from taskview_ai.schemas import PlanRequest


async def main() -> None:
    cases = json.loads((Path(__file__).parents[1] / "evals" / "cases.json").read_text())
    settings = Settings(taskview_ai_fake_mode=False)
    failures: list[str] = []

    for case in cases:
        plan = await build_view_plan(
            PlanRequest(
                purpose=case["purpose"],
                audience=case["audience"],
                ttl_days=case["ttl_days"],
            ),
            settings,
        )
        applied = {
            field: item.transformation
            for item in plan.transformations
            for field in item.input_fields
        }
        for field, expected in case["required_transforms"].items():
            if applied.get(field) != expected:
                failures.append(
                    f"{case['name']}: {field} expected {expected}, got {applied.get(field)}"
                )
        leaked = set(plan.preview_columns) & set(case["forbidden_preview_columns"])
        if leaked:
            failures.append(f"{case['name']}: forbidden preview columns {sorted(leaked)}")
        print(f"PASS {case['name']}" if not failures else f"CHECK {case['name']}")

    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(main())

