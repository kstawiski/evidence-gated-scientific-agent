#!/usr/bin/env bash
set -euo pipefail

if command -v python3 >/dev/null 2>&1; then
  python_command=python3
elif command -v python >/dev/null 2>&1; then
  python_command=python
else
  echo 'Python 3 is required to read project metadata' >&2
  exit 1
fi
project_name=${COMPOSE_PROJECT_NAME:-evidence-bench-smoke-$$}
if [ "$project_name" = evidence-bench ]; then
  echo 'Refusing to run the destructive smoke cleanup against the production Compose project' >&2
  exit 1
fi
novnc_port=${BROWSER_NOVNC_PORT:-$("$python_command" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')}
version=$("$python_command" -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
browser_image=${EVIDENCE_BENCH_BROWSER_IMAGE:-evidence-bench-browser:${version}}
export PACKAGE_WORKER_TOKEN=${PACKAGE_WORKER_TOKEN:-container-smoke-package-token-0001}
smoke_root=$(mktemp -d "${TMPDIR:-/tmp}/evidence-bench-smoke.XXXXXX")
mkdir -p "$smoke_root/browser" "$smoke_root/data" "$smoke_root/environments"
chmod 0777 "$smoke_root/browser" "$smoke_root/data" "$smoke_root/environments"
export BROWSER_BIND_ADDRESS=127.0.0.1
export BROWSER_NOVNC_PORT=$novnc_port
export EVIDENCE_BENCH_BROWSER_PATH=$smoke_root/browser
export EVIDENCE_BENCH_DATA_PATH=$smoke_root/data
export EVIDENCE_BENCH_ENVIRONMENTS_PATH=$smoke_root/environments
private_probe="${project_name}-private-api-probe"
app_probe="${project_name}-app-network-probe"

compose() {
  docker compose --project-name "$project_name" "$@"
}

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    compose logs --tail 200 browser browser-egress browser-cdp-gateway \
      browser-ui-gateway environment-worker environment-gateway || true
  fi
  docker rm --force "$private_probe" "$app_probe" 2>/dev/null || true
  compose down --volumes || true
  if [ -d "$smoke_root" ]; then
    docker run --rm --network none \
      --volume "$smoke_root:/smoke" \
      --entrypoint /bin/chmod "$browser_image" \
      -R a+rwX /smoke >/dev/null 2>&1 || true
  fi
  rm -rf "$smoke_root" || true
  trap - EXIT
  exit "$status"
}
trap cleanup EXIT

wait_healthy() {
  service=$1
  container=$(compose ps --quiet "$service")
  test -n "$container"
  status=starting
  for _attempt in $(seq 1 90); do
    status=$(docker inspect --format '{{.State.Health.Status}}' "$container")
    if [ "$status" = healthy ]; then
      return 0
    fi
    if [ "$status" = unhealthy ]; then
      return 1
    fi
    sleep 2
  done
  return 1
}

compose up --detach --no-build browser-cdp-gateway browser-ui-gateway environment-gateway
wait_healthy browser-cdp-gateway
wait_healthy environment-gateway
curl --fail --silent "http://127.0.0.1:${novnc_port}/vnc.html" >/dev/null

docker run --detach --network "${project_name}_public-egress" --name "$private_probe" \
  --entrypoint /bin/sleep "$browser_image" 300
docker run --detach --network "${project_name}_browser-client" --name "$app_probe" \
  --entrypoint /bin/sleep "$browser_image" 300

compose exec -T browser curl --fail --silent --show-error \
  --proxy http://browser-egress:3128 https://example.com/ >/dev/null
if compose exec -T browser curl --fail --connect-timeout 3 \
  https://example.com/ >/dev/null 2>&1; then
  echo 'browser unexpectedly has direct Internet egress' >&2
  exit 1
fi
if compose exec -T browser getent hosts "$private_probe" >/dev/null 2>&1; then
  echo 'browser unexpectedly resolves an application-network service' >&2
  exit 1
fi
if compose exec -T browser-egress getent hosts "$app_probe" >/dev/null 2>&1; then
  echo 'egress proxy unexpectedly resolves an application-network service' >&2
  exit 1
fi

code=$(compose exec -T browser curl --silent --output /dev/null \
  --write-out '%{http_code}' --proxy http://browser-egress:3128 \
  "http://${private_probe}:8080/")
test "$code" = 403
for target in 10.0.0.1 100.64.0.1 169.254.169.254; do
  code=$(compose exec -T browser curl --silent --output /dev/null \
    --write-out '%{http_code}' --proxy http://browser-egress:3128 "http://$target/")
  test "$code" = 403
done

compose exec -T environment-worker python - "$private_probe" "$app_probe" <<'PY'
import socket
import sys
import urllib.error
import urllib.request

private_probe, app_probe = sys.argv[1:]
direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    direct.open("https://example.com/", timeout=3)
except (OSError, urllib.error.URLError):
    pass
else:
    raise SystemExit("environment worker unexpectedly has direct Internet egress")

for hostname in (private_probe, app_probe):
    try:
        socket.getaddrinfo(hostname, 80)
    except socket.gaierror:
        continue
    raise SystemExit(f"environment worker unexpectedly resolves {hostname}")

proxy = urllib.request.build_opener(
    urllib.request.ProxyHandler(
        {
            "http": "http://browser-egress:3128",
            "https": "http://browser-egress:3128",
        }
    )
)
with proxy.open("https://example.com/", timeout=15) as response:
    if response.status != 200:
        raise SystemExit(f"package proxy returned HTTP {response.status}")

for target in (
    f"http://{private_probe}:8080/",
    "http://10.0.0.1/",
    "http://100.64.0.1/",
    "http://169.254.169.254/",
):
    try:
        proxy.open(target, timeout=5)
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            continue
        raise
    raise SystemExit(f"package proxy unexpectedly allowed {target}")
PY

compose exec -T environment-worker python - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

token = os.environ["SCIENTIFIC_AGENT_PACKAGE_WORKER_TOKEN"]
workspace_id = str(uuid.uuid4())


def post(path, payload, authorization):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:8091{path}",
        data=body,
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=360) as response:
        return response.status, json.load(response)


install = {
    "request_id": str(uuid.uuid4()),
    "workspace_id": workspace_id,
    "language": "python",
    "repository": "pypi",
    "packages": ["six==1.17.0"],
    "timeout_seconds": 300,
}
try:
    post("/install", install, "Bearer invalid-container-smoke-token")
except urllib.error.HTTPError as exc:
    if exc.code != 401:
        raise
else:
    raise SystemExit("package worker accepted an invalid token")

status, result = post("/install", install, f"Bearer {token}")
if status != 200 or result.get("status") != "succeeded":
    raise SystemExit(f"package installation failed: {result}")
if {item.get("name").lower() for item in result.get("installed", [])} != {"six"}:
    raise SystemExit(f"unexpected package inventory: {result.get('installed')}")

package_root = (
    Path("/environments") / workspace_id / "python" / "packages"
).resolve()
sys.path.insert(0, str(package_root))
import six  # noqa: E402

if six.__version__ != "1.17.0":
    raise SystemExit(f"unexpected six version: {six.__version__}")
if not Path(six.__file__).resolve().is_relative_to(package_root):
    raise SystemExit("installed package did not load from the workspace environment")

status, cleanup = post(
    "/cleanup",
    {"workspace_id": workspace_id},
    f"Bearer {token}",
)
if status != 200 or cleanup.get("status") != "deleted":
    raise SystemExit(f"package cleanup failed: {cleanup}")
if (Path("/environments") / workspace_id).exists():
    raise SystemExit("package cleanup left the workspace environment behind")
print("real environment-worker install/import/cleanup smoke passed")
PY

docker run --rm --network "${project_name}_browser-client" \
  --entrypoint /usr/bin/curl "$browser_image" \
  --fail --silent http://browser-cdp-gateway:9222/json/version >/dev/null
