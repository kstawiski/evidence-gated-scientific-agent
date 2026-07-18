#!/usr/bin/env bash
set -Eeuo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
env_file=${EVIDENCE_BENCH_ENV_FILE:-"$project_dir/.env"}
command_name=${1:-start}

usage() {
  cat <<'EOF'
Usage: ./scripts/local_run.sh [start|stop|restart|status|logs|preflight|update]

Controls the local release deployment created by scripts/local_setup.sh.
Persistent workspaces, package environments, and browser state are preserved by
stop and restart.
EOF
}

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

require_runtime() {
  command -v docker >/dev/null 2>&1 || fail "Docker is not installed. See docs/LOCAL_SETUP.md."
  docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required."
  docker info >/dev/null 2>&1 || fail "Docker is not running. Start Docker and retry."
  [ -f "$env_file" ] || fail "Configuration not found at $env_file. Run ./scripts/local_setup.sh first."
}

compose() {
  docker compose \
    --project-directory "$project_dir" \
    --env-file "$env_file" \
    -f "$project_dir/compose.yaml" \
    -f "$project_dir/compose.local.yaml" \
    "$@"
}

published_port() {
  awk -F= '
    $1 == "WEB_PUBLISHED_PORT" {
      value = $2
      gsub(/^[[:space:]\047\"]+|[[:space:]\047\"]+$/, "", value)
      print value
      found = 1
      exit
    }
    END { if (!found) print "8080" }
  ' "$env_file"
}

wait_for_health() {
  local port attempt
  port=$(published_port)
  attempt=0
  while [ "$attempt" -lt 90 ]; do
    if curl -fsS "http://127.0.0.1:${port}/healthz" >/dev/null 2>&1; then
      printf 'Evidence Bench is ready at http://127.0.0.1:%s\n' "$port"
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 2
  done
  compose ps >&2 || true
  fail "Evidence Bench did not become healthy within 180 seconds."
}

run_preflight() {
  local port
  port=$(published_port)
  curl -fsS "http://127.0.0.1:${port}/healthz" >/dev/null \
    || fail "Evidence Bench health check failed."
  compose exec -T evidence-bench scientific-agent preflight --mcp ""
}

case "$command_name" in
  start)
    require_runtime
    compose pull
    compose up -d --no-build
    wait_for_health
    run_preflight
    ;;
  stop)
    require_runtime
    compose down
    ;;
  restart)
    require_runtime
    compose down
    compose up -d --no-build
    wait_for_health
    run_preflight
    ;;
  status)
    require_runtime
    compose ps
    ;;
  logs)
    require_runtime
    compose logs -f --tail=200 evidence-bench
    ;;
  preflight)
    require_runtime
    run_preflight
    ;;
  update)
    require_runtime
    compose pull
    compose up -d --no-build
    wait_for_health
    run_preflight
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
