"""Environment-backed settings for the standalone service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class WebSettings:
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SCIENTIFIC_AGENT_DATA_DIR", "./web-data")
        ).resolve()
    )
    auth_enabled: bool = field(default_factory=lambda: _flag("WEB_AUTH_ENABLED", True))
    username: str = field(
        default_factory=lambda: os.environ.get("WEB_USERNAME", "scientist")
    )
    password: str = field(default_factory=lambda: os.environ.get("WEB_PASSWORD", ""))
    a2a_token: str = field(default_factory=lambda: os.environ.get("A2A_TOKEN", ""))
    public_url: str = field(
        default_factory=lambda: os.environ.get(
            "SCIENTIFIC_AGENT_PUBLIC_URL", "http://127.0.0.1:8080"
        ).rstrip("/")
    )
    host: str = field(default_factory=lambda: os.environ.get("WEB_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("WEB_PORT", "8080")))
    max_upload_bytes: int = field(
        default_factory=lambda: int(
            os.environ.get("WEB_MAX_UPLOAD_BYTES", str(256 * 1024**2))
        )
    )
    max_workers: int = field(
        default_factory=lambda: int(os.environ.get("WEB_MAX_WORKERS", "2"))
    )
    a2a_enabled: bool = field(default_factory=lambda: _flag("A2A_ENABLED", True))
    browser_public_url: str = field(
        default_factory=lambda: os.environ.get("BROWSER_PUBLIC_URL", "").strip()
    )
    browser_novnc_port: int = field(
        default_factory=lambda: int(os.environ.get("BROWSER_NOVNC_PORT", "6080"))
    )

    @property
    def database_path(self) -> Path:
        return self.data_dir / "evidence-bench.sqlite3"

    @property
    def workspaces_dir(self) -> Path:
        return self.data_dir / "workspaces"

    @property
    def browser_frame_sources(self) -> tuple[str, ...]:
        if self.browser_public_url:
            parsed = urlsplit(self.browser_public_url)
            return (f"{parsed.scheme}://{parsed.netloc}",)
        port = self.browser_novnc_port
        return (f"http://*:{port}", f"https://*:{port}")

    def validate(self) -> None:
        if self.auth_enabled and (not self.username or not self.password):
            raise RuntimeError(
                "WEB_USERNAME and WEB_PASSWORD are required when WEB_AUTH_ENABLED is true"
            )
        if self.a2a_enabled and not self.a2a_token:
            raise RuntimeError("A2A_TOKEN is required when A2A_ENABLED is true")
        if self.max_workers < 1:
            raise ValueError("WEB_MAX_WORKERS must be positive")
        if self.max_upload_bytes < 1:
            raise ValueError("WEB_MAX_UPLOAD_BYTES must be positive")
        if not 1 <= self.browser_novnc_port <= 65535:
            raise ValueError("BROWSER_NOVNC_PORT must be between 1 and 65535")
        if self.browser_public_url:
            parsed = urlsplit(self.browser_public_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username
                or parsed.password
            ):
                raise ValueError(
                    "BROWSER_PUBLIC_URL must be an HTTP(S) URL without credentials"
                )
