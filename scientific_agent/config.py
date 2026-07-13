"""Configuration with fleet-safe defaults and owner-only secret loading."""

from __future__ import annotations

import os
import shlex
import shutil
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _first_existing(*candidates: str | Path | None) -> Path:
    """Return the first existing runtime path, retaining a useful final error path."""

    usable = [Path(value).expanduser() for value in candidates if value]
    for path in usable:
        if path.exists():
            return path.resolve()
    return usable[0] if usable else Path("/nonexistent")


def _executable(env_name: str, command: str, *fallbacks: str | Path) -> Path:
    configured = os.environ.get(env_name)
    discovered = shutil.which(command)
    return _first_existing(configured, discovered, *fallbacks)


def _python_runtime() -> Path:
    return _first_existing(
        os.environ.get("SCIENTIFIC_AGENT_PYTHON"),
        Path.home() / "micromamba/bin/python3",
        Path.home() / "miniforge3/bin/python3",
        sys.executable,
        shutil.which("python3"),
    )


def _python_prefix() -> Path:
    configured = os.environ.get("SCIENTIFIC_AGENT_PYTHON_PREFIX")
    runtime = _python_runtime()
    inferred = runtime.parent.parent if runtime.parent.name == "bin" else Path(sys.prefix)
    return _first_existing(configured, inferred, sys.prefix)


def _python_packages() -> Path:
    configured = os.environ.get("SCIENTIFIC_AGENT_PYTHON_PACKAGES")
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return _first_existing(
        configured,
        _python_prefix() / "lib" / version / "site-packages",
        Path.home() / ".local/lib" / version / "site-packages",
    )


def _r_library() -> Path:
    configured = os.environ.get("SCIENTIFIC_AGENT_R_LIBRARY")
    candidates = sorted((Path.home() / "R").glob("*-linux-gnu-library/*"), reverse=True)
    return _first_existing(configured, *candidates, Path("/usr/local/lib/R/site-library"))


@dataclass(frozen=True)
class ModelEndpoint:
    base_url: str
    model: str
    api_key: str
    max_tokens: int
    temperature: float
    top_p: float


@dataclass(frozen=True)
class SandboxSettings:
    bwrap: Path = field(
        default_factory=lambda: _executable(
            "SCIENTIFIC_AGENT_BWRAP", "bwrap", "/usr/bin/bwrap"
        )
    )
    prlimit: Path = field(
        default_factory=lambda: _executable(
            "SCIENTIFIC_AGENT_PRLIMIT", "prlimit", "/usr/bin/prlimit"
        )
    )
    python: Path = field(default_factory=_python_runtime)
    python_prefix: Path = field(default_factory=_python_prefix)
    python_packages: Path = field(default_factory=_python_packages)
    rscript: Path = field(
        default_factory=lambda: _executable(
            "SCIENTIFIC_AGENT_RSCRIPT", "Rscript", "/usr/bin/Rscript"
        )
    )
    r_library: Path = field(default_factory=_r_library)
    max_wall_seconds: int = int(
        os.environ.get("SCIENTIFIC_AGENT_MAX_WALL_SECONDS", "300")
    )
    max_memory_bytes: int = int(
        os.environ.get("SCIENTIFIC_AGENT_MAX_MEMORY_BYTES", str(8 * 1024**3))
    )
    max_processes: int = int(
        os.environ.get("SCIENTIFIC_AGENT_MAX_PROCESSES", "32")
    )
    max_file_bytes: int = int(
        os.environ.get("SCIENTIFIC_AGENT_MAX_FILE_BYTES", str(64 * 1024**2))
    )
    max_output_bytes: int = int(
        os.environ.get("SCIENTIFIC_AGENT_MAX_OUTPUT_BYTES", str(256 * 1024**2))
    )
    max_code_bytes: int = int(
        os.environ.get("SCIENTIFIC_AGENT_MAX_CODE_BYTES", str(128 * 1024))
    )
    max_calls_per_attempt: int = int(
        os.environ.get("SCIENTIFIC_AGENT_MAX_CODE_CALLS", "8")
    )


@dataclass(frozen=True)
class Settings:
    sandbox: SandboxSettings = field(default_factory=SandboxSettings)
    qwen: ModelEndpoint = field(
        default_factory=lambda: ModelEndpoint(
            base_url=os.environ.get(
                "QWEN_BASE_URL", "http://127.0.0.1:8000/v1"
            ),
            model=os.environ.get("QWEN_MODEL", "Qwen/Qwen3.6-27B"),
            api_key=os.environ.get("QWEN_API_KEY", ""),
            max_tokens=int(os.environ.get("QWEN_MAX_TOKENS", "16000")),
            temperature=float(os.environ.get("QWEN_TEMPERATURE", "0.6")),
            top_p=float(os.environ.get("QWEN_TOP_P", "0.95")),
        )
    )
    gemma: ModelEndpoint = field(
        default_factory=lambda: ModelEndpoint(
            base_url=os.environ.get(
                "GEMMA_BASE_URL", "http://127.0.0.1:8001/v1"
            ),
            model=os.environ.get("GEMMA_MODEL", "gemma4-12b-it"),
            api_key=os.environ.get("GEMMA_API_KEY", ""),
            max_tokens=int(os.environ.get("GEMMA_MAX_TOKENS", "12000")),
            temperature=float(os.environ.get("GEMMA_TEMPERATURE", "1.0")),
            top_p=float(os.environ.get("GEMMA_TOP_P", "0.95")),
        )
    )
    workspace: Path = field(
        default_factory=lambda: Path(os.environ.get("SCIENTIFIC_AGENT_WORKSPACE", "."))
        .resolve()
    )
    runs_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SCIENTIFIC_AGENT_RUNS_DIR", str(PROJECT_ROOT / "runs"))
        ).resolve()
    )
    max_repair_rounds: int = int(os.environ.get("MAX_REPAIR_ROUNDS", "1"))
    mcp_servers: tuple[str, ...] = ("context7", "brave-search")
    chrome_browser_url: str = os.environ.get(
        "CHROME_DEVTOOLS_BROWSER_URL", "http://127.0.0.1:9222"
    )


_SECRET_NAMES = {"CONTEXT7_API_KEY", "BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY"}


def load_mcp_secrets(path: Path | None = None) -> dict[str, str]:
    """Read only allow-listed assignments from an owner-only environment file.

    The file is parsed as data; it is never sourced or executed.
    """

    secret_file = path or Path(
        os.environ.get(
            "SCIENTIFIC_AGENT_MCP_ENV_FILE",
            str(Path.home() / ".config" / "mcp-services.env"),
        )
    )
    if not secret_file.exists():
        return {}
    info = secret_file.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise PermissionError(f"secret file must be a regular non-symlink: {secret_file}")
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise PermissionError(f"secret file must be current-user-owned and mode 600: {secret_file}")

    values: dict[str, str] = {}
    for raw in secret_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name not in _SECRET_NAMES:
            continue
        parsed = shlex.split(value, posix=True)
        if len(parsed) != 1:
            raise ValueError(f"invalid secret assignment for {name}")
        values[name] = parsed[0]

    if "BRAVE_API_KEY" not in values and values.get("BRAVE_SEARCH_API_KEY"):
        values["BRAVE_API_KEY"] = values["BRAVE_SEARCH_API_KEY"]
    return values
