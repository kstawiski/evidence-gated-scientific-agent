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
DEFAULT_MCP_SERVERS = ("context7", "brave-search", "chrome-devtools")


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _optional_bool_env(name: str, default: str = "inherit") -> bool | None:
    value = os.environ.get(name, default).strip().casefold()
    if value in {"", "auto", "default", "inherit"}:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true, false, or inherit")


def _optional_positive_int_env(name: str, default: str) -> int | None:
    value = int(os.environ.get(name, default))
    if value < 0:
        raise ValueError(f"{name} must be zero or a positive integer")
    return value or None


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
    inferred = (
        runtime.parent.parent if runtime.parent.name == "bin" else Path(sys.prefix)
    )
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
    return _first_existing(
        configured, *candidates, Path("/usr/local/lib/R/site-library")
    )


@dataclass(frozen=True)
class ModelEndpoint:
    base_url: str
    model: str
    api_key: str
    max_tokens: int | None
    temperature: float
    top_p: float
    enable_thinking: bool | None = None
    native_json_schema: bool = True
    request_timeout_seconds: int | None = None


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
    worker_url: str = field(
        default_factory=lambda: os.environ.get(
            "SCIENTIFIC_AGENT_SANDBOX_WORKER_URL", ""
        ).rstrip("/")
    )
    worker_token: str = field(
        default_factory=lambda: os.environ.get(
            "SCIENTIFIC_AGENT_SANDBOX_WORKER_TOKEN", ""
        )
    )
    max_wall_seconds: int = int(
        os.environ.get("SCIENTIFIC_AGENT_MAX_WALL_SECONDS", "300")
    )
    max_memory_bytes: int = int(
        os.environ.get("SCIENTIFIC_AGENT_MAX_MEMORY_BYTES", str(8 * 1024**3))
    )
    max_processes: int = int(os.environ.get("SCIENTIFIC_AGENT_MAX_PROCESSES", "32"))
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
        os.environ.get("SCIENTIFIC_AGENT_MAX_CODE_CALLS", "12")
    )


@dataclass(frozen=True)
class EnvironmentSettings:
    worker_url: str = field(
        default_factory=lambda: os.environ.get(
            "SCIENTIFIC_AGENT_PACKAGE_WORKER_URL", ""
        ).rstrip("/")
    )
    worker_token: str = field(
        default_factory=lambda: os.environ.get(
            "SCIENTIFIC_AGENT_PACKAGE_WORKER_TOKEN", ""
        )
    )
    max_packages_per_call: int = int(
        os.environ.get("SCIENTIFIC_AGENT_MAX_PACKAGES_PER_CALL", "24")
    )
    install_timeout_seconds: int = int(
        os.environ.get("SCIENTIFIC_AGENT_PACKAGE_TIMEOUT_SECONDS", "900")
    )


@dataclass(frozen=True)
class LiteratureSettings:
    """Configuration for fixed-host PubMed and PMC acquisition tools."""

    ncbi_email: str = field(
        default_factory=lambda: os.environ.get(
            "SCIENTIFIC_AGENT_NCBI_EMAIL", ""
        ).strip()
    )
    ncbi_tool: str = field(
        default_factory=lambda: os.environ.get(
            "SCIENTIFIC_AGENT_NCBI_TOOL", "evidence_bench"
        ).strip()
    )
    ncbi_api_key: str = field(
        default_factory=lambda: os.environ.get(
            "SCIENTIFIC_AGENT_NCBI_API_KEY", ""
        ).strip()
    )
    browser_downloads_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SCIENTIFIC_AGENT_BROWSER_DOWNLOADS", "/browser-downloads")
        ).resolve()
    )
    pdftotext: Path = field(
        default_factory=lambda: _executable(
            "SCIENTIFIC_AGENT_PDFTOTEXT", "pdftotext", "/usr/bin/pdftotext"
        )
    )
    pdftoppm: Path = field(
        default_factory=lambda: _executable(
            "SCIENTIFIC_AGENT_PDFTOPPM", "pdftoppm", "/usr/bin/pdftoppm"
        )
    )
    max_pdf_bytes: int = int(
        os.environ.get("SCIENTIFIC_AGENT_MAX_ARTICLE_PDF_BYTES", str(64 * 1024**2))
    )
    max_archive_bytes: int = int(
        os.environ.get("SCIENTIFIC_AGENT_MAX_ARTICLE_ARCHIVE_BYTES", str(128 * 1024**2))
    )


@dataclass(frozen=True)
class Settings:
    sandbox: SandboxSettings = field(default_factory=SandboxSettings)
    environment: EnvironmentSettings = field(default_factory=EnvironmentSettings)
    literature: LiteratureSettings = field(default_factory=LiteratureSettings)
    qwen: ModelEndpoint = field(
        default_factory=lambda: ModelEndpoint(
            base_url=os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8000/v1"),
            model=os.environ.get("QWEN_MODEL", "Qwen/Qwen3.6-27B"),
            api_key=os.environ.get("QWEN_API_KEY", ""),
            max_tokens=_optional_positive_int_env("QWEN_MAX_TOKENS", "0"),
            temperature=float(os.environ.get("QWEN_TEMPERATURE", "0.6")),
            top_p=float(os.environ.get("QWEN_TOP_P", "0.95")),
            enable_thinking=_optional_bool_env("QWEN_ENABLE_THINKING"),
            native_json_schema=_bool_env("QWEN_NATIVE_JSON_SCHEMA", True),
            request_timeout_seconds=_optional_positive_int_env(
                "QWEN_REQUEST_TIMEOUT_SECONDS", "7200"
            ),
        )
    )
    gemma: ModelEndpoint = field(
        default_factory=lambda: ModelEndpoint(
            base_url=os.environ.get("GEMMA_BASE_URL", "http://127.0.0.1:8001/v1"),
            model=os.environ.get("GEMMA_MODEL", "gemma4-12b-it"),
            api_key=os.environ.get("GEMMA_API_KEY", ""),
            max_tokens=_optional_positive_int_env("GEMMA_MAX_TOKENS", "0"),
            temperature=float(os.environ.get("GEMMA_TEMPERATURE", "1.0")),
            top_p=float(os.environ.get("GEMMA_TOP_P", "0.95")),
            enable_thinking=_optional_bool_env("GEMMA_ENABLE_THINKING"),
            native_json_schema=_bool_env("GEMMA_NATIVE_JSON_SCHEMA", True),
            request_timeout_seconds=_optional_positive_int_env(
                "GEMMA_REQUEST_TIMEOUT_SECONDS", "7200"
            ),
        )
    )
    workspace: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SCIENTIFIC_AGENT_WORKSPACE", ".")
        ).resolve()
    )
    runs_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SCIENTIFIC_AGENT_RUNS_DIR", str(PROJECT_ROOT / "runs"))
        ).resolve()
    )
    max_repair_rounds: int = field(
        default_factory=lambda: int(os.environ.get("MAX_REPAIR_ROUNDS", "4"))
    )
    max_research_model_turns: int = field(
        default_factory=lambda: int(
            os.environ.get("SCIENTIFIC_AGENT_MAX_RESEARCH_MODEL_TURNS", "64")
        )
    )
    max_research_tool_calls: int = field(
        default_factory=lambda: int(
            os.environ.get("SCIENTIFIC_AGENT_MAX_RESEARCH_TOOL_CALLS", "48")
        )
    )
    max_repeated_tool_results: int = field(
        default_factory=lambda: int(
            os.environ.get("SCIENTIFIC_AGENT_MAX_REPEATED_TOOL_RESULTS", "2")
        )
    )
    mcp_servers: tuple[str, ...] = DEFAULT_MCP_SERVERS
    chrome_browser_url: str = os.environ.get(
        "CHROME_DEVTOOLS_BROWSER_URL", "http://127.0.0.1:9222"
    )

    def __post_init__(self) -> None:
        if not 0 <= self.max_repair_rounds <= 8:
            raise ValueError("max_repair_rounds must be between 0 and 8")
        if not 1 <= self.max_research_model_turns <= 256:
            raise ValueError("max_research_model_turns must be between 1 and 256")
        if not 1 <= self.max_research_tool_calls <= 256:
            raise ValueError("max_research_tool_calls must be between 1 and 256")
        if not 1 <= self.max_repeated_tool_results <= 8:
            raise ValueError("max_repeated_tool_results must be between 1 and 8")


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
    values: dict[str, str] = {
        name: os.environ[name] for name in _SECRET_NAMES if os.environ.get(name)
    }
    if secret_file.exists():
        info = secret_file.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise PermissionError(
                f"secret file must be a regular non-symlink: {secret_file}"
            )
        mode = stat.S_IMODE(info.st_mode)
        owner_private = info.st_uid == os.getuid() and mode == 0o600
        docker_secret = info.st_uid == 0 and mode in {0o400, 0o440, 0o444}
        if not owner_private and not docker_secret:
            raise PermissionError(
                "secret file must be current-user-owned mode 600 or a "
                f"root-owned read-only container secret: {secret_file}"
            )

        file_values: dict[str, str] = {}
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
            file_values[name] = parsed[0]
        values = {**file_values, **values}

    if "BRAVE_API_KEY" not in values and values.get("BRAVE_SEARCH_API_KEY"):
        values["BRAVE_API_KEY"] = values["BRAVE_SEARCH_API_KEY"]
    return values
