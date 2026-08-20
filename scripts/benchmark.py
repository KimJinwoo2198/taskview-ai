import argparse
import asyncio
import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

EXPECTED_KEYS = {
    "summary",
    "subject",
    "comparison_dimensions",
    "desired_outcome",
    "region_label",
    "department",
    "selected_source",
    "confidence",
    "needs_clarification",
    "clarifying_question",
}
INTERNAL_TERMS = ("semantic", "schema", "policy engine", "inference firewall")
FORBIDDEN_PLAN_COLUMNS = {
    "name",
    "phone",
    "email",
    "address",
    "exact_address",
    "message",
    "ticket_id",
    "customer_name",
    "vin",
    "raw_ticket_text",
}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return ordered[index]


async def post_json(
    client: httpx.AsyncClient, path: str, payload: dict[str, Any]
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = await client.post(path, json=payload)
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response.json(), latency_ms


def evaluate_case(
    case: dict[str, Any], intent: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    text = " ".join(
        [
            str(intent.get("summary", "")),
            str(intent.get("subject", "")),
            " ".join(intent.get("comparison_dimensions", [])),
            str(intent.get("desired_outcome", "")),
            str(intent.get("clarifying_question") or ""),
        ]
    ).casefold()
    forbidden_terms = [*INTERNAL_TERMS, *case.get("forbidden_terms", [])]
    checks = {
        "structured_output": set(intent) == EXPECTED_KEYS,
        "department": intent.get("department") == case["audience"],
        "source_routing": intent.get("selected_source") == case["expected_source"],
        "clarification": bool(intent.get("needs_clarification")) == case["expected_clarification"],
        "decision_language": str(intent.get("desired_outcome", "")).endswith(
            ("정한다", "결정한다")
        ),
        "required_terms": all(term.casefold() in text for term in case["required_terms"]),
        "no_internal_or_injected_terms": not any(
            term.casefold() in text for term in forbidden_terms
        ),
        "catalog_safe_plan": not (set(plan.get("preview_columns", [])) & FORBIDDEN_PLAN_COLUMNS)
        and set(plan.get("selected_sources", [])).issubset({"product", "operations", "voc"}),
    }
    return {
        "name": case["name"],
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "confidence": intent.get("confidence"),
        "intent": intent,
    }


def markdown_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    lines = [
        f"# Needex AI Benchmark · {result['suite']}",
        "",
        f"- 실행 시각: {result['run_at']}",
        f"- 모델: `{result['model']}`",
        f"- 케이스: {metrics['cases']}개",
        f"- 종합 통과율: **{metrics['overall_pass_rate']:.1f}%**",
        f"- 소스 라우팅 정확도: **{metrics['routing_accuracy']:.1f}%**",
        f"- 모호한 요청 판별 정확도: **{metrics['clarification_accuracy']:.1f}%**",
        f"- 구조화 출력 성공률: **{metrics['structured_output_rate']:.1f}%**",
        f"- 안전 계획 통과율: **{metrics['safe_plan_rate']:.1f}%**",
        f"- 최초 요청 지연 p50 / p95: **{metrics['cold_p50_ms']:.0f} / {metrics['cold_p95_ms']:.0f} ms**",
        f"- 캐시 요청 지연 p50 / p95: **{metrics['warm_p50_ms']:.0f} / {metrics['warm_p95_ms']:.0f} ms**",
        "",
        "## 케이스별 결과",
        "",
        "| 케이스 | 점수 | 라우팅 | 확인 질문 | 안전 계획 | 최초 ms | 캐시 ms |",
        "|---|---:|---|---|---|---:|---:|",
    ]
    for case in result["cases"]:
        checks = case["checks"]
        lines.append(
            f"| {case['name']} | {case['passed']}/{case['total']} | "
            f"{'PASS' if checks['source_routing'] else 'FAIL'} | "
            f"{'PASS' if checks['clarification'] else 'FAIL'} | "
            f"{'PASS' if checks['catalog_safe_plan'] else 'FAIL'} | "
            f"{case['cold_latency_ms']:.0f} | {case['warm_latency_ms']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## 해석 범위",
            "",
            "이 평가는 Needex가 실제로 사용하는 로컬 모델과 결정론적 안전 계층을 함께 측정합니다. 일반 지능이나 모든 산업 도메인의 정확도를 주장하지 않습니다. 케이스와 원시 JSON을 함께 공개해 누구나 동일 환경에서 재실행할 수 있습니다.",
            "",
        ]
    )
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--secret", default="")
    parser.add_argument("--output-dir", default="output/benchmark")
    parser.add_argument("--cases", default="evals/benchmark_cases.json")
    parser.add_argument("--suite", default="regression")
    parser.add_argument("--fail-under", type=float, default=85.0)
    args = parser.parse_args()

    root = Path(__file__).parents[1]
    cases = json.loads((root / args.cases).read_text())
    headers = {"authorization": f"Bearer {args.secret}"} if args.secret else {}
    timeout = httpx.Timeout(180)
    evaluated: list[dict[str, Any]] = []
    cold_latencies: list[float] = []
    warm_latencies: list[float] = []

    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"), headers=headers, timeout=timeout
    ) as client:
        health = (await client.get("/health")).json()
        for case in cases:
            payload = {
                "purpose": case["purpose"],
                "audience": case["audience"],
                "ttl_days": 7,
            }
            intent, cold_ms = await post_json(client, "/v1/agent/interpret", payload)
            _, warm_ms = await post_json(client, "/v1/agent/interpret", payload)
            plan, _ = await post_json(client, "/v1/agent/plan", payload)
            row = evaluate_case(case, intent, plan)
            row["cold_latency_ms"] = cold_ms
            row["warm_latency_ms"] = warm_ms
            evaluated.append(row)
            cold_latencies.append(cold_ms)
            warm_latencies.append(warm_ms)
            print(f"{row['passed']}/{row['total']} {case['name']} ({cold_ms:.0f} ms)")

    total_passed = sum(case["passed"] for case in evaluated)
    total_checks = sum(case["total"] for case in evaluated)

    def check_rate(name: str) -> float:
        return 100 * statistics.fmean(float(case["checks"][name]) for case in evaluated)

    result = {
        "run_at": datetime.now(UTC).isoformat(),
        "suite": args.suite,
        "model": health.get("model", "unknown"),
        "base_url": args.base_url,
        "metrics": {
            "cases": len(evaluated),
            "overall_pass_rate": 100 * total_passed / total_checks,
            "routing_accuracy": check_rate("source_routing"),
            "clarification_accuracy": check_rate("clarification"),
            "structured_output_rate": check_rate("structured_output"),
            "safe_plan_rate": check_rate("catalog_safe_plan"),
            "cold_p50_ms": statistics.median(cold_latencies),
            "cold_p95_ms": percentile(cold_latencies, 0.95),
            "warm_p50_ms": statistics.median(warm_latencies),
            "warm_p95_ms": percentile(warm_latencies, 0.95),
        },
        "cases": evaluated,
    }
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    (output_dir / "REPORT.md").write_text(markdown_report(result))
    print(f"report: {output_dir / 'REPORT.md'}")
    if result["metrics"]["overall_pass_rate"] < args.fail_under:
        raise SystemExit(
            f"benchmark score {result['metrics']['overall_pass_rate']:.1f}% < {args.fail_under:.1f}%"
        )


if __name__ == "__main__":
    asyncio.run(main())
