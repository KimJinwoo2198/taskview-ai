# Needex AI 벤치마크 방법론

Needex의 AI 품질은 모델의 자유 대화 능력이 아니라 실제 제품 계약을 기준으로 평가합니다.

## 측정 항목

1. 요청 부서 유지
2. 승인된 데이터 소스 라우팅 정확도
3. 모호한 요청의 확인 질문 생성
4. 고정 JSON schema 준수
5. 쉬운 의사결정 문장 생성
6. 내부 기술 용어와 prompt injection 내용 미노출
7. 직접 식별자가 없는 catalog-safe 계획
8. 최초 요청과 메모리 캐시 요청 지연시간

## 실행

```bash
make benchmark
make benchmark-holdout

# 공유 비밀을 설정한 독립 AI 서버
uv run python scripts/benchmark.py \
  --base-url https://ai.example.com \
  --secret "$TASKVIEW_AI_SHARED_SECRET"
```

결과는 `output/benchmark/REPORT.md`와 원시 `benchmark.json`에 생성됩니다. 케이스는 `evals/benchmark_cases.json`에 있어 심사위원이나 개발자가 그대로 검토하고 재실행할 수 있습니다.

- `benchmark`: 개발 중 반복 실행하는 회귀 품질 게이트
- `benchmark-holdout`: 규칙 조정에 사용하지 않은 별도 표현의 독립 확인 세트

## 주장하지 않는 것

이 벤치마크는 Needex가 지원하는 목적 분석과 안전한 데이터 계획 범위만 측정합니다. 모든 산업·언어에 대한 일반 지능, 원본 데이터의 진실성, 인과 추론 정확도를 의미하지 않습니다. 최종 공개 범위는 AI가 아니라 결정론적 정책과 사람 승인이 집행합니다.
