# Needex AI

Needex의 목적 해석과 안전한 변환 계획을 만드는 단일 오케스트레이터 Agent입니다. 데이터 접근 허용 여부, 승인, 실제 View 생성은 BE가 결정합니다.

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
  -d '{"purpose":"NYC 311 민원에서 기관과 유형별 처리 지연을 찾아 운영 우선순위를 정하고 싶다","audience":"operations","ttl_days":7}'
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
## Ollama 설치와 전체 스택 실행

macOS에서는 다음 순서로 실제 모델을 준비합니다.

```bash
brew install ollama          # 이미 설치했다면 생략
ollama serve                 # 별도 터미널에서 계속 실행
ollama pull qwen3.5:9b       # 또는 이 저장소에서 make model
```

AI만 호스트에서 실행할 때:

```bash
make install
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1 \
OLLAMA_MODEL=qwen3.5:9b \
TASKVIEW_AI_FAKE_MODE=false \
make dev
```

전체 스택은 Ollama를 호스트에 실행한 채 `taskview-be`에서 `docker compose up -d --build`로 시작합니다. Compose AI는 `http://host.docker.internal:11434/v1`로 호스트 Ollama에 연결해 macOS의 로컬 가속을 사용합니다.

## 실패와 fallback 경계

`TASKVIEW_AI_FAKE_MODE=true`는 계약·테스트용 명시적 모드이며 운영에서 사용하지 않습니다. 실제 모델의 구조화 출력이 schema 검증에 실패하거나 15초를 초과하면, Agent는 허용된 catalog만 사용하는 보수적 최소 계획을 반환하고 `assumptions`에 fallback 이유를 넣습니다. 모델 결과가 catalog 안전성 검사를 통과하지 못해도 같은 방식으로 축소합니다. Ollama 연결 자체가 실패하면 API는 `503`을 반환합니다.

fallback 계획도 승인이나 데이터 접근 권한을 부여하지 않습니다. 최종 정책, owner 승인, Evidence, TTL, materialization은 계속 BE 책임입니다. FCC·NYC 311·NHTSA 계획은 BE가 동기화한 공식 공개 데이터 안전 스냅샷에서 materialize하며, 조직이 연결한 임의 warehouse 실행은 별도 범위입니다.

## 운영 설정

- `OLLAMA_BASE_URL`: AI 서버에서 접근 가능한 OpenAI-compatible Ollama `/v1` URL
- `OLLAMA_MODEL`: 배포 전에 pull하고 용량/지연을 검증한 모델(현재 기준 `qwen3.5:9b`)
- `TASKVIEW_AI_FAKE_MODE=false`: 운영 필수
- 외부 공개 없이 BE만 AI를 호출하도록 네트워크를 제한합니다.

## 검증

```bash
make test          # 단위/API 회귀 테스트
make lint          # Ruff
make eval          # 실제 Ollama 개인정보·구조화 출력 평가
git diff --check
```

`make eval`은 fake mode가 아니라 실제 Ollama를 사용하므로 `ollama serve`와 모델 설치가 선행되어야 합니다.

심사·품질 검증용 평가 방법과 실제 결과는 [AI 벤치마크 방법론](docs/AI_BENCHMARK.md), [벤치마크 결과](docs/BENCHMARK_RESULTS.md)를 참고하세요.

## 독립 Docker 배포

이 저장소만 AI 머신에 복제하면 Ollama, Qwen 모델, Needex AI API를 함께 실행할 수 있습니다. BE/FE 저장소는 필요하지 않습니다.

```bash
cp .env.deploy.example .env.deploy
# TASKVIEW_AI_SHARED_SECRET을 `openssl rand -hex 32` 결과로 변경
./scripts/deploy.sh
```

NVIDIA GPU를 사용할 때:

```bash
TASKVIEW_USE_GPU=true ./scripts/deploy.sh
```

첫 실행은 `qwen3.5:9b` 모델을 persistent volume에 내려받아 시간이 걸릴 수 있습니다. 이후 재배포에서는 같은 volume을 재사용합니다. 외부에는 AI 포트만 노출되고 Ollama 포트는 공개하지 않습니다. 운영에서는 방화벽·HTTPS reverse proxy를 적용하고, `.env.deploy`의 공유 비밀을 BE의 `TASKVIEW_AI_SHARED_SECRET`과 동일하게 설정하세요.
