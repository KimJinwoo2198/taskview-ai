#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

if [ ! -f .env.deploy ]; then
  echo ".env.deploy 파일이 없습니다. .env.deploy.example을 복사하고 공유 비밀을 설정하세요." >&2
  exit 1
fi

network_name=$(sed -n 's/^NEEDEX_NETWORK=//p' .env.deploy | tail -1)
network_name=${network_name:-needex_internal}
docker network inspect "$network_name" >/dev/null 2>&1 || docker network create "$network_name" >/dev/null

gpu_args=""
if [ "${TASKVIEW_USE_GPU:-false}" = "true" ]; then
  gpu_args="-f compose.gpu.yaml"
fi

# shellcheck disable=SC2086
docker compose --env-file .env.deploy -f compose.deploy.yaml $gpu_args up -d --build
docker compose --env-file .env.deploy -f compose.deploy.yaml ps
