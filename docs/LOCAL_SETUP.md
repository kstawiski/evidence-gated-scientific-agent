# Local setup: macOS, Linux, and WSL2

Evidence Bench can install and run its complete local model stack with one
interactive command. The setup script detects system RAM and dedicated GPU VRAM,
recommends a Qwen executor and vision-capable Gemma critic, lets you change the
profile, and performs the remaining setup.

```bash
git clone https://github.com/kstawiski/evidence-gated-scientific-agent.git
cd evidence-gated-scientific-agent
./scripts/local_setup.sh
```

The script installs Ollama from its official installer when needed, downloads
both models, creates 32K-context Evidence Bench aliases, generates independent
credentials in an owner-only `.env`, pulls the released application images, and
runs a model endpoint preflight. It does not send prompts or files to a hosted
model service.

## Before you begin

Install and start Docker before running the script:

- **macOS:** Docker Desktop on macOS 14 Sonoma or newer. Apple silicon uses
  unified memory; Intel Macs run Ollama on CPU.
- **Linux:** Docker Engine with the Docker Compose v2 plugin. Keep a supported
  NVIDIA/AMD driver installed if you want GPU acceleration.
- **Windows:** WSL2 plus Docker Desktop with integration enabled for the chosen
  Linux distribution. Run the setup command from the WSL2 shell, preferably
  from the Linux filesystem rather than `/mnt/c`.

Docker Desktop includes Compose. On native Linux, confirm both components with:

```bash
docker info
docker compose version
```

The application and its confined workers need at least 16 GB system RAM; 24 GB
or more is strongly recommended. Model downloads require 6–44 GB in addition to
Docker images and workspace storage. Small local models are useful for product
evaluation and lighter analyses, but they do not make consequential scientific
or medical work self-validating.

## Hardware-based model choice

Automatic selection uses total system RAM, summed NVIDIA VRAM when available,
and Linux AMD VRAM reported by the kernel. Apple silicon is selected by unified
system memory. The thresholds leave room for Docker, the operating system, model
context, and runtime overhead; they are recommendations rather than guarantees.

<!-- markdownlint-disable MD013 -->

| Profile | Automatic threshold | Qwen executor | Gemma vision critic | Download |
| --- | --- | --- | --- | ---: |
| Compact | below the higher thresholds | `qwen3:4b` | `gemma3:4b` | ~6 GB |
| Balanced | 40 GB RAM, or 12 GB VRAM + 24 GB RAM | `qwen3:14b` | `gemma4:12b` | ~17 GB |
| Performance | 64 GB RAM, or 24 GB VRAM + 32 GB RAM | `qwen3.6:27b` | `gemma4:26b` | ~35 GB |
| Workstation | 96 GB RAM, or 32 GB VRAM + 48 GB RAM | `qwen3.6:35b` | `gemma4:31b` | ~44 GB |

<!-- markdownlint-enable MD013 -->

The current model names and sizes come from the official
[Qwen 3](https://ollama.com/library/qwen3),
[Qwen 3.6](https://ollama.com/library/qwen3.6),
[Gemma 3](https://ollama.com/library/gemma3), and
[Gemma 4](https://ollama.com/library/gemma4) Ollama libraries. Gemma remains a
different model family and is verified for vision capability during setup.

Inspect the recommendation without changing the machine:

```bash
./scripts/local_setup.sh --recommend-only
```

Choose a profile directly or override either model:

```bash
./scripts/local_setup.sh --profile balanced

./scripts/local_setup.sh \
  --profile balanced \
  --qwen-model qwen3:8b \
  --gemma-model gemma3:12b
```

Custom model names must be present in the Ollama registry. A custom Gemma model
must advertise vision capability or setup stops. Use `--context 16384` to reduce
memory pressure or a value up to 131072 when the hardware can support it.

For unattended installation, accept the hardware recommendation with:

```bash
./scripts/local_setup.sh --non-interactive
```

## What setup changes

The installer makes only the following durable changes:

1. Installs Ollama when it is absent. macOS and Linux use Ollama's official
   `install.sh`. WSL2 with Docker Desktop uses the official Windows installer so
   Windows GPU acceleration remains available.
2. Pulls the selected Qwen and Gemma weights into Ollama's model storage.
3. Creates model aliases whose names preserve the source family, size, and
   configured context, for example `evidence-bench-qwen-qwen3-14b-32k:latest`.
4. Creates or updates `.env` with the local endpoints and generated credentials.
   Existing non-placeholder credentials are preserved and the file is mode 0600.
5. Pulls the released AMD64/ARM64 application, package-worker, and browser images
   through `compose.local.yaml` and starts them without a source rebuild.

On native Linux and native-Docker WSL2, Ollama is bound to the Docker bridge
address instead of `0.0.0.0`; this lets the application reach it without
publishing the unauthenticated model API on the LAN. The systemd drop-in is
`/etc/systemd/system/ollama.service.d/evidence-bench.conf`. Docker Desktop
provides its own `host.docker.internal` bridge on macOS and Windows. On native
Docker, setup records that bridge address as `OLLAMA_HOST_GATEWAY` so the
application container resolves the same private endpoint.

Ollama provides the `/v1/chat/completions`, `/v1/models`, structured-output,
thinking, tools, and vision features needed by Evidence Bench through its
[OpenAI-compatible API](https://docs.ollama.com/api/openai-compatibility).

## Use the local deployment

Open <http://127.0.0.1:8080> after setup finishes. The generated username is
`scientist`; read the password locally without printing the other service tokens:

```bash
grep '^WEB_PASSWORD=' .env
```

The control helper always uses the release Compose overlay and preserves data on
ordinary stop/restart operations:

```bash
./scripts/local_run.sh status
./scripts/local_run.sh logs
./scripts/local_run.sh preflight
./scripts/local_run.sh stop
./scripts/local_run.sh start
./scripts/local_run.sh restart
./scripts/local_run.sh update
```

To prepare models and configuration without starting Docker:

```bash
./scripts/local_setup.sh --skip-start
```

Rerunning setup is safe: Ollama reuses downloaded layers, aliases are refreshed,
and generated credentials are retained. `local_run.sh stop` does not remove
Docker volumes or Ollama models.

## Platform notes

### macOS

The official Ollama application runs outside Docker so Apple silicon can use the
GPU. Docker Desktop for macOS cannot pass the Apple GPU into a Linux container.
If installation succeeds but the command is not yet on `PATH`, start Ollama once
from Applications and accept its CLI-link prompt, then rerun setup.

### Linux

The official installer configures a systemd service when systemd is available.
Setup adds a narrow Docker-bridge binding and limits Ollama to one loaded model
at a time. Without systemd, it starts a background `ollama serve` process and
writes its log to `.evidence-bench/ollama.log`.

Ollama detects supported NVIDIA GPUs automatically. Current AMD acceleration
requires the driver version documented in Ollama's
[hardware support guide](https://docs.ollama.com/gpu). Unsupported GPUs fall back
to CPU and system RAM.

### WSL2

With Docker Desktop, setup installs or reuses native Ollama for Windows and calls
its CLI from WSL2. This is the least fragile route for NVIDIA and AMD acceleration.
Windows must be able to expose `http://localhost:11434` to the WSL2 distribution.

If you deliberately run a native Docker Engine inside WSL2 instead, setup uses
the Linux Ollama installer and Docker-bridge binding. Enable systemd in WSL2 for
boot-persistent Ollama operation, or rerun setup after restarting the WSL2 VM.

## Troubleshooting

### The recommendation is too large

Choose a smaller profile and context:

```bash
./scripts/local_setup.sh --profile compact --context 16384
```

Close other GPU/RAM-heavy applications before a run. Model quality generally
falls with smaller profiles; deterministic Evidence Bench gates may therefore
return an inconclusive result more often.

### Docker cannot reach Ollama

Confirm Ollama first, then the container path:

```bash
curl http://127.0.0.1:11434/api/tags
./scripts/local_run.sh preflight
```

On native Linux, also inspect:

```bash
systemctl status ollama
cat /etc/systemd/system/ollama.service.d/evidence-bench.conf
```

On WSL2 with Docker Desktop, confirm Ollama is running in the Windows taskbar and
that `curl http://127.0.0.1:11434/api/tags` succeeds from WSL2.

### Docker Desktop runs out of memory

Increase Docker Desktop's memory allowance or select the compact profile. The
models run outside Docker, but the browser, Python/R sandbox, and package worker
still need Docker VM memory.

### Start from source instead of release images

The local helper intentionally uses versioned release images. Developers can
build the checked-out source with the base Compose file:

```bash
docker compose up --build -d
```

Read the [deployment guide](../deploy/README.md) and
[threat model](THREAT_MODEL.md) before changing bind addresses or exposing the
workbench beyond the local machine.
