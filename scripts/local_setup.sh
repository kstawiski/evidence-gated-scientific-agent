#!/usr/bin/env bash
set -Eeuo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
evidence_bench_version=$(awk -F'"' '/^version = "/ {print $2; exit}' "$project_dir/pyproject.toml")
[ -n "$evidence_bench_version" ] || {
  printf 'Error: could not read the Evidence Bench version\n' >&2
  exit 1
}
env_file="$project_dir/.env"
profile=auto
profile_requested=false
qwen_source_model=""
gemma_source_model=""
qwen_model_overridden=false
gemma_model_overridden=false
context_tokens=32768
ram_override=""
vram_override=""
non_interactive=false
recommend_only=false
dry_run=false
skip_ollama_install=false
skip_model_pull=false
skip_start=false
platform=""
ram_gb=0
vram_gb=0
docker_desktop=false
ollama_api_url=http://127.0.0.1:11434
ollama_host_gateway=""
temporary_dir=""

usage() {
  cat <<'EOF'
Usage: ./scripts/local_setup.sh [options]

Install and configure a local Qwen + Gemma Evidence Bench deployment on macOS,
Linux, or WSL2.

Options:
  --profile NAME       auto, compact, balanced, performance, or workstation
  --qwen-model NAME    override the selected Ollama Qwen source model
  --gemma-model NAME   override the selected Ollama Gemma source model
  --context TOKENS     model context window (8192-131072; default: 32768)
  --env-file PATH      deployment environment file (default: ./.env)
  --ram-gb N           override detected system RAM (for testing/manual choice)
  --vram-gb N          override detected dedicated GPU VRAM
  --non-interactive    accept the recommendation without prompting
  --recommend-only     print the hardware recommendation and make no changes
  --dry-run            print the selected setup without installing or starting
  --skip-ollama-install require an existing Ollama installation
  --skip-model-pull    do not download or create model aliases
  --skip-start         configure models and .env without starting Compose
  -h, --help           show this help
EOF
}

info() {
  printf '\n==> %s\n' "$*"
}

warn() {
  printf 'Warning: %s\n' "$*" >&2
}

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

is_positive_integer() {
  case ${1:-} in
    ''|*[!0-9]*) return 1 ;;
    *) [ "$1" -gt 0 ] ;;
  esac
}

is_nonnegative_integer() {
  case ${1:-} in
    ''|*[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

validate_model_name() {
  case $1 in
    ''|*[!A-Za-z0-9._:/-]*) return 1 ;;
    *) return 0 ;;
  esac
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile)
      [ "$#" -ge 2 ] || fail "--profile requires a value"
      profile=$2
      profile_requested=true
      shift 2
      ;;
    --qwen-model)
      [ "$#" -ge 2 ] || fail "--qwen-model requires a value"
      qwen_source_model=$2
      qwen_model_overridden=true
      shift 2
      ;;
    --gemma-model)
      [ "$#" -ge 2 ] || fail "--gemma-model requires a value"
      gemma_source_model=$2
      gemma_model_overridden=true
      shift 2
      ;;
    --context)
      [ "$#" -ge 2 ] || fail "--context requires a value"
      context_tokens=$2
      shift 2
      ;;
    --env-file)
      [ "$#" -ge 2 ] || fail "--env-file requires a value"
      env_file=$2
      shift 2
      ;;
    --ram-gb)
      [ "$#" -ge 2 ] || fail "--ram-gb requires a value"
      ram_override=$2
      shift 2
      ;;
    --vram-gb)
      [ "$#" -ge 2 ] || fail "--vram-gb requires a value"
      vram_override=$2
      shift 2
      ;;
    --non-interactive)
      non_interactive=true
      shift
      ;;
    --recommend-only)
      recommend_only=true
      non_interactive=true
      shift
      ;;
    --dry-run)
      dry_run=true
      non_interactive=true
      shift
      ;;
    --skip-ollama-install)
      skip_ollama_install=true
      shift
      ;;
    --skip-model-pull)
      skip_model_pull=true
      shift
      ;;
    --skip-start)
      skip_start=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      fail "unknown option: $1"
      ;;
  esac
done

case "$profile" in
  auto|compact|balanced|performance|workstation) ;;
  *) fail "unsupported profile: $profile" ;;
esac
is_positive_integer "$context_tokens" || fail "--context must be a positive integer"
if [ "$context_tokens" -lt 8192 ] || [ "$context_tokens" -gt 131072 ]; then
  fail "--context must be between 8192 and 131072"
fi
[ -z "$ram_override" ] || is_positive_integer "$ram_override" \
  || fail "--ram-gb must be a positive integer"
[ -z "$vram_override" ] || is_nonnegative_integer "$vram_override" \
  || fail "--vram-gb must be a non-negative integer"
[ -z "$qwen_source_model" ] || validate_model_name "$qwen_source_model" \
  || fail "invalid Qwen model name"
[ -z "$gemma_source_model" ] || validate_model_name "$gemma_source_model" \
  || fail "invalid Gemma model name"

detect_platform() {
  case $(uname -s) in
    Darwin)
      platform=macos
      ;;
    Linux)
      if grep -qi microsoft /proc/sys/kernel/osrelease /proc/version 2>/dev/null; then
        platform=wsl2
      else
        platform=linux
      fi
      ;;
    *)
      fail "supported platforms are macOS, Linux, and WSL2"
      ;;
  esac
}

detect_ram() {
  if [ -n "$ram_override" ]; then
    ram_gb=$ram_override
    return
  fi
  if [ "$platform" = macos ]; then
    ram_gb=$(sysctl -n hw.memsize | awk '{printf "%.0f", $1 / 1073741824}')
  else
    ram_gb=$(awk '/^MemTotal:/ {printf "%.0f", $2 / 1048576}' /proc/meminfo)
  fi
  [ "$ram_gb" -gt 0 ] || fail "could not detect system RAM; pass --ram-gb"
}

detect_vram() {
  local nvidia_command sysfs_bytes
  if [ -n "$vram_override" ]; then
    vram_gb=$vram_override
    return
  fi
  nvidia_command=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia_command=nvidia-smi
  elif command -v nvidia-smi.exe >/dev/null 2>&1; then
    nvidia_command=nvidia-smi.exe
  fi
  if [ -n "$nvidia_command" ]; then
    vram_gb=$(
      "$nvidia_command" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | tr -d '\r' \
        | awk '{sum += $1} END {printf "%.0f", sum / 1024}'
    )
  elif [ "$platform" != macos ]; then
    sysfs_bytes=$(
      for memory_file in /sys/class/drm/card[0-9]*/device/mem_info_vram_total; do
        [ -r "$memory_file" ] && cat "$memory_file"
      done | awk '{sum += $1} END {printf "%.0f", sum}'
    )
    if [ -n "$sysfs_bytes" ] && [ "$sysfs_bytes" -gt 0 ] 2>/dev/null; then
      vram_gb=$(awk -v bytes="$sysfs_bytes" 'BEGIN {printf "%.0f", bytes / 1073741824}')
    fi
  fi
  vram_gb=${vram_gb:-0}
}

recommend_profile() {
  local recommendation=compact
  if [ "$ram_gb" -ge 40 ] \
    || { [ "$vram_gb" -ge 12 ] && [ "$ram_gb" -ge 24 ]; }; then
    recommendation=balanced
  fi
  if [ "$ram_gb" -ge 64 ] \
    || { [ "$vram_gb" -ge 24 ] && [ "$ram_gb" -ge 32 ]; }; then
    recommendation=performance
  fi
  if [ "$ram_gb" -ge 96 ] \
    || { [ "$vram_gb" -ge 32 ] && [ "$ram_gb" -ge 48 ]; }; then
    recommendation=workstation
  fi
  if [ "$profile" = auto ]; then profile=$recommendation; fi
}

assign_profile_models() {
  case "$profile" in
    compact)
      selected_qwen=qwen3:4b
      selected_gemma=gemma3:4b
      model_download_gb=6
      profile_note="Evaluation and lighter analyses; expect more gate rejections."
      ;;
    balanced)
      selected_qwen=qwen3:14b
      selected_gemma=gemma4:12b
      model_download_gb=17
      profile_note="Recommended laptop/workstation balance."
      ;;
    performance)
      selected_qwen=qwen3.6:27b
      selected_gemma=gemma4:26b
      model_download_gb=35
      profile_note="Strong local reasoning and multimodal review."
      ;;
    workstation)
      selected_qwen=qwen3.6:35b
      selected_gemma=gemma4:31b
      model_download_gb=44
      profile_note="Highest-quality supported local pair."
      ;;
  esac
  [ -z "$qwen_source_model" ] && qwen_source_model=$selected_qwen
  [ -z "$gemma_source_model" ] && gemma_source_model=$selected_gemma
  if [ "$qwen_model_overridden" = true ] || [ "$gemma_model_overridden" = true ]; then
    model_download_gb=custom
  fi
  return 0
}

choose_profile_interactively() {
  local choice
  [ "$non_interactive" = false ] || return 0
  [ "$profile_requested" = false ] || return 0
  [ -t 0 ] || return 0
  printf '\nModel profiles:\n'
  printf '  1) %s (recommended for this machine)\n' "$profile"
  printf '  2) compact      qwen3:4b + gemma3:4b       ~6 GB\n'
  printf '  3) balanced     qwen3:14b + gemma4:12b     ~17 GB\n'
  printf '  4) performance  qwen3.6:27b + gemma4:26b  ~35 GB\n'
  printf '  5) workstation  qwen3.6:35b + gemma4:31b  ~44 GB\n'
  printf 'Choose [1-5, default 1]: '
  read -r choice
  case ${choice:-1} in
    1) ;;
    2) profile=compact ;;
    3) profile=balanced ;;
    4) profile=performance ;;
    5) profile=workstation ;;
    *) fail "invalid profile choice" ;;
  esac
}

model_alias() {
  local role=$1 source=$2 normalized
  normalized=$(printf '%s' "$source" | tr ':./_' '----')
  printf 'evidence-bench-%s-%s-%sk:latest' "$role" "$normalized" "$((context_tokens / 1024))"
}

print_recommendation() {
  printf 'PLATFORM=%s\n' "$platform"
  printf 'RAM_GB=%s\n' "$ram_gb"
  printf 'VRAM_GB=%s\n' "$vram_gb"
  printf 'PROFILE=%s\n' "$profile"
  printf 'QWEN_SOURCE_MODEL=%s\n' "$qwen_source_model"
  printf 'GEMMA_SOURCE_MODEL=%s\n' "$gemma_source_model"
  printf 'CONTEXT_TOKENS=%s\n' "$context_tokens"
  printf 'MODEL_DOWNLOAD_GB=%s\n' "$model_download_gb"
}

detect_platform
detect_ram
detect_vram
recommend_profile
choose_profile_interactively
assign_profile_models
qwen_alias=$(model_alias qwen "$qwen_source_model")
gemma_alias=$(model_alias gemma "$gemma_source_model")

if [ "$recommend_only" = true ]; then
  print_recommendation
  exit 0
fi

info "Detected $platform with ${ram_gb} GB RAM and ${vram_gb} GB dedicated VRAM"
printf 'Selected profile: %s — %s\n' "$profile" "$profile_note"
printf 'Qwen executor:    %s\n' "$qwen_source_model"
printf 'Gemma critic:     %s\n' "$gemma_source_model"
printf 'Context:          %s tokens\n' "$context_tokens"
printf 'Model download:   approximately %s GB\n' "$model_download_gb"
if [ "$ram_gb" -lt 24 ]; then
  warn "Less than 24 GB RAM may cause swapping or startup failure. Use the compact profile and close memory-heavy applications."
fi

if [ "$dry_run" = true ]; then
  print_recommendation
  printf 'QWEN_ALIAS=%s\n' "$qwen_alias"
  printf 'GEMMA_ALIAS=%s\n' "$gemma_alias"
  printf 'ENV_FILE=%s\n' "$env_file"
  exit 0
fi

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v docker >/dev/null 2>&1 || fail "Docker is not installed. Follow docs/LOCAL_SETUP.md, start Docker, and retry."
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required"
docker info >/dev/null 2>&1 || fail "Docker is not running. Start Docker and retry."
if docker info --format '{{.OperatingSystem}}' 2>/dev/null | grep -qi 'docker desktop'; then
  docker_desktop=true
fi

temporary_dir=$(mktemp -d)
cleanup() {
  [ -z "$temporary_dir" ] || rm -rf -- "$temporary_dir"
}
trap cleanup EXIT INT TERM

wait_for_ollama() {
  local attempt
  attempt=0
  while [ "$attempt" -lt 60 ]; do
    if curl -fsS "$ollama_api_url/api/tags" >/dev/null 2>&1; then return 0; fi
    attempt=$((attempt + 1))
    sleep 2
  done
  fail "Ollama did not become reachable at $ollama_api_url"
}

install_ollama_unix() {
  local installer="$temporary_dir/ollama-install.sh"
  curl -fsSL https://ollama.com/install.sh -o "$installer"
  sh "$installer"
}

install_ollama_windows() {
  command -v powershell.exe >/dev/null 2>&1 \
    || fail "PowerShell is unavailable in WSL2. Install Ollama for Windows from https://ollama.com/download/windows."
  powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command \
    'Invoke-Expression (Invoke-RestMethod https://ollama.com/install.ps1)' </dev/null
}

windows_ollama_path() {
  local windows_path
  # PowerShell, not Bash, expands the environment reference in this argument.
  # shellcheck disable=SC2016
  windows_path=$(powershell.exe -NoProfile -NonInteractive -Command \
    '[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"' \
    | tr -d '\r')
  if command -v wslpath >/dev/null 2>&1; then
    wslpath -u "$windows_path"
  else
    printf '%s' "$windows_path"
  fi
}

configure_linux_ollama() {
  local docker_gateway dropin
  docker_gateway=$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null)
  [ -n "$docker_gateway" ] || fail "could not determine the Docker bridge gateway"
  ollama_host_gateway=$docker_gateway
  ollama_api_url="http://${docker_gateway}:11434"
  if command -v systemctl >/dev/null 2>&1 \
    && systemctl list-unit-files ollama.service >/dev/null 2>&1; then
    dropin=/etc/systemd/system/ollama.service.d/evidence-bench.conf
    info "Binding Ollama to the Docker bridge only ($docker_gateway)"
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    printf '[Service]\nEnvironment="OLLAMA_HOST=%s:11434"\nEnvironment="OLLAMA_MAX_LOADED_MODELS=1"\n' "$docker_gateway" \
      | sudo tee "$dropin" >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
  else
    info "Starting Ollama on the Docker bridge ($docker_gateway)"
    mkdir -p "$project_dir/.evidence-bench"
    OLLAMA_HOST="${docker_gateway}:11434" OLLAMA_MAX_LOADED_MODELS=1 \
      nohup ollama serve >"$project_dir/.evidence-bench/ollama.log" 2>&1 &
  fi
}

if [ "$platform" = wsl2 ] && [ "$docker_desktop" = true ]; then
  windows_ollama=$(windows_ollama_path)
  if [ ! -x "$windows_ollama" ]; then
    [ "$skip_ollama_install" = false ] \
      || fail "Ollama for Windows is required but was not found"
    info "Installing Ollama for Windows through its official installer"
    install_ollama_windows
    windows_ollama=$(windows_ollama_path)
  fi
  [ -x "$windows_ollama" ] || fail "Ollama for Windows was installed but its CLI was not found"
  ollama_command=$windows_ollama
  ollama_api_url=http://127.0.0.1:11434
elif [ "$platform" = macos ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    [ "$skip_ollama_install" = false ] || fail "Ollama is required but was not found"
    info "Installing Ollama through its official installer"
    install_ollama_unix
  fi
  ollama_command=$(command -v ollama)
  open -a Ollama --args hidden >/dev/null 2>&1 || true
  ollama_api_url=http://127.0.0.1:11434
elif [ "$platform" = linux ] && [ "$docker_desktop" = true ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    [ "$skip_ollama_install" = false ] || fail "Ollama is required but was not found"
    info "Installing Ollama through its official installer"
    install_ollama_unix
  fi
  ollama_command=$(command -v ollama)
  ollama_api_url=http://127.0.0.1:11434
else
  if ! command -v ollama >/dev/null 2>&1; then
    [ "$skip_ollama_install" = false ] || fail "Ollama is required but was not found"
    info "Installing Ollama through its official installer"
    install_ollama_unix
  fi
  ollama_command=$(command -v ollama)
  configure_linux_ollama
fi

wait_for_ollama

create_alias() {
  local source=$1 alias=$2 response
  response=$(curl -fsS "$ollama_api_url/api/create" \
    -H 'Content-Type: application/json' \
    --data-binary "{\"model\":\"$alias\",\"from\":\"$source\",\"parameters\":{\"num_ctx\":$context_tokens},\"stream\":false}")
  printf '%s' "$response" | grep -Eq '"status"[[:space:]]*:[[:space:]]*"success"' \
    || fail "Ollama could not create $alias"
}

verify_alias() {
  local alias=$1 require_vision=$2 details
  details=$(curl -fsS "$ollama_api_url/api/show" \
    -H 'Content-Type: application/json' \
    --data-binary "{\"model\":\"$alias\"}")
  if [ "$require_vision" = true ]; then
    printf '%s' "$details" | grep -Eq '"vision"' \
      || fail "Gemma model $alias does not advertise vision capability"
  fi
}

if [ "$skip_model_pull" = false ]; then
  info "Downloading Qwen ($qwen_source_model)"
  "$ollama_command" pull "$qwen_source_model"
  info "Downloading Gemma ($gemma_source_model)"
  "$ollama_command" pull "$gemma_source_model"
  info "Creating Evidence Bench model aliases with a $context_tokens-token context"
  create_alias "$qwen_source_model" "$qwen_alias"
  create_alias "$gemma_source_model" "$gemma_alias"
  verify_alias "$qwen_alias" false
  verify_alias "$gemma_alias" true
fi

get_env_value() {
  local key=$1
  [ -f "$env_file" ] || return 0
  awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); gsub(/^[[:space:]\047\"]+|[[:space:]\047\"]+$/, ""); print; exit}' "$env_file"
}

set_env_value() {
  local key=$1 value=$2 temporary
  temporary=$(mktemp "${env_file}.XXXXXX")
  awk -v key="$key" -v value="$value" '
    BEGIN { found = 0 }
    $0 ~ "^" key "=" {
      if (!found) print key "=" value
      found = 1
      next
    }
    { print }
    END { if (!found) print key "=" value }
  ' "$env_file" >"$temporary"
  chmod 600 "$temporary"
  mv "$temporary" "$env_file"
}

random_token() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
  else
    od -An -N32 -tx1 /dev/urandom | tr -d ' \n'
  fi
}

ensure_secret() {
  local key=$1 current
  current=$(get_env_value "$key")
  case "$current" in
    ''|replace-with-*) set_env_value "$key" "$(random_token)" ;;
  esac
}

info "Writing owner-only local deployment configuration"
mkdir -p "$(dirname -- "$env_file")"
if [ ! -f "$env_file" ]; then
  cp "$project_dir/.env.example" "$env_file"
fi
chmod 600 "$env_file"
set_env_value EVIDENCE_BENCH_VERSION "$evidence_bench_version"
set_env_value EVIDENCE_BENCH_DEPLOYMENT_ID local
set_env_value SCIENTIFIC_AGENT_PUBLIC_URL http://127.0.0.1:8080
set_env_value WEB_BIND_ADDRESS 127.0.0.1
set_env_value BROWSER_BIND_ADDRESS 127.0.0.1
set_env_value QWEN_BASE_URL http://host.docker.internal:11434/v1
set_env_value QWEN_MODEL "$qwen_alias"
set_env_value QWEN_API_KEY ollama
set_env_value QWEN_ENABLE_THINKING inherit
set_env_value QWEN_NATIVE_JSON_SCHEMA true
set_env_value GEMMA_BASE_URL http://host.docker.internal:11434/v1
set_env_value GEMMA_MODEL "$gemma_alias"
set_env_value GEMMA_API_KEY ollama
set_env_value GEMMA_ENABLE_THINKING inherit
set_env_value GEMMA_NATIVE_JSON_SCHEMA true
[ -z "$ollama_host_gateway" ] || set_env_value OLLAMA_HOST_GATEWAY "$ollama_host_gateway"
set_env_value EVIDENCE_BENCH_LOCAL_PROFILE "$profile"
set_env_value EVIDENCE_BENCH_LOCAL_QWEN_SOURCE "$qwen_source_model"
set_env_value EVIDENCE_BENCH_LOCAL_GEMMA_SOURCE "$gemma_source_model"
set_env_value EVIDENCE_BENCH_LOCAL_CONTEXT "$context_tokens"
ensure_secret WEB_PASSWORD
ensure_secret A2A_TOKEN
ensure_secret SANDBOX_WORKER_TOKEN
ensure_secret PACKAGE_WORKER_TOKEN

if [ "$skip_start" = false ]; then
  info "Starting the released Evidence Bench stack"
  EVIDENCE_BENCH_ENV_FILE="$env_file" "$project_dir/scripts/local_run.sh" start
else
  printf '\nConfiguration complete. Start later with:\n  EVIDENCE_BENCH_ENV_FILE=%q ./scripts/local_run.sh start\n' "$env_file"
fi

printf '\nLocal setup complete. Credentials are stored only in %s (mode 0600).\n' "$env_file"
