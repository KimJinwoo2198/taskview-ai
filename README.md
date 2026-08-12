# TaskView AI

TaskView의 목적 해석과 안전한 변환 계획을 만드는 단일 오케스트레이터 Agent입니다. 데이터 접근 허용 여부, 승인, 실제 View 생성은 BE가 결정합니다.

## 기본 구성

- 런타임: Python 3.12 + FastAPI
- Agent: OpenAI Agents SDK
- 로컬 추론: Ollama OpenAI 호환 API
- 기본 모델: `qwen3.5:9b` (16GB Apple Silicon 기준). 메모리가 빠듯하면 `qwen3:8b`로 변경하세요.
- Agent는 로컬 응답시간과 구조화 출력 안정성을 위해 Qwen의 장문 reasoning을 끄고 목적 문장과 승인된 데이터 소스만 선택합니다. 개인정보 변환 계획과 최종 정책 판정은 코드로 검증 가능한 결정론적 계층이 수행합니다.

## 실행

```bash
ollama serve
make model
make install
make dev
```

모델 없이 API 계약만 확인하려면 다음처럼 결정론적 fake 모드를 사용합니다.

```bash
TASKVIEW_AI_FAKE_MODE=true make dev
```

```bash
curl -X POST http://localhost:8100/v1/agent/plan \
  -H 'content-type: application/json' \
  -d '{"purpose":"VOC를 지역과 이슈 유형별로 묶어 다음 스프린트 우선순위를 정하고 싶다","audience":"product","ttl_days":7}'
```

## 책임 경계

- Agent: 목적 구조화와 승인된 데이터 소스 선택
- 결정론적 AI 안전 계층: 카탈로그 필드 제한, 직접 식별자 제거, 최소 변환 계획 생성
- BE: 정책 판정, 승인 상태, materialization, TTL, 감사 로그
- 금지: Agent가 임의 SQL 실행, 정책 우회, 직접 개인정보 승인

API 문서는 실행 후 `/docs`, 상태 확인은 `/health`에서 확인합니다.

실제 모델의 개인정보 변환 및 preview 누출 여부는 다음 회귀 평가로 확인합니다.

```bash
make eval
```
