"""Deterministic registration and rendering of scientific report displays."""

from __future__ import annotations

import csv
from functools import lru_cache
import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from .provenance import sha256_file, write_json
from .schemas import (
    ComputationEvidence,
    ReportDisplay,
    RetrievalEvidence,
    ScientificReport,
)


FIGURE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
TABLE_DELIMITERS = {".csv": ",", ".tsv": "\t"}
MAX_FIGURE_BYTES = 20 * 1024 * 1024
MAX_TABLE_BYTES = 20 * 1024 * 1024
MAX_TABLE_ROWS = 50
MAX_TABLE_COLUMNS = 20
MIN_REPORTED_FIGURE_DPI = 300.0
MAX_READER_TABLE_SIGNIFICANT_DIGITS = 4
MAX_OCR_WORDS = 400
MAX_OCR_TEXT_CHARS = 12_000
_NUMBER_CELL = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _successful_artifacts(computation: ComputationEvidence) -> dict[str, str | None]:
    return {
        str(Path(artifact.path).resolve()): artifact.sha256
        for artifact in computation.artifacts
    }


def logical_report_output_key(path: Path) -> str | None:
    """Return a stable /output/figures-or-tables key across repair attempts."""

    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    for index in range(len(parts) - 2, -1, -1):
        if lowered[index] != "output" or lowered[index + 1] not in {
            "figures",
            "tables",
        }:
            continue
        return "/".join([lowered[index + 1], *parts[index + 2 :]])
    return None


def resolve_display_artifact(
    display: ReportDisplay,
    computation: ComputationEvidence,
) -> Path:
    """Resolve a display only when it is an exact successful computation artifact."""

    path = Path(display.artifact_path)
    if not path.is_absolute():
        raise ValueError(f"display {display.display_id} artifact_path must be absolute")
    resolved = path.resolve()
    known = _successful_artifacts(computation)
    if str(resolved) not in known:
        raise ValueError(
            f"display {display.display_id} is not a successful computation artifact"
        )
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"display {display.display_id} artifact is unavailable")
    expected_hash = known[str(resolved)]
    if expected_hash and sha256_file(resolved) != expected_hash:
        raise ValueError(f"display {display.display_id} artifact hash changed")
    return resolved


def inspect_figure(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix not in FIGURE_MEDIA_TYPES:
        raise ValueError("inline figures must be PNG, JPEG, or WebP")
    size = path.stat().st_size
    if size < 1 or size > MAX_FIGURE_BYTES:
        raise ValueError("figure has an invalid file size")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            detected_format = (image.format or "").upper()
            reported_dpi = image.info.get("dpi")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("figure is not a readable raster image") from exc
    if width < 240 or height < 160 or width > 20_000 or height > 20_000:
        raise ValueError(
            f"figure dimensions are outside the supported range: {width}x{height}"
        )
    expected_formats = {
        ".png": {"PNG"},
        ".jpg": {"JPEG"},
        ".jpeg": {"JPEG"},
        ".webp": {"WEBP"},
    }
    if detected_format not in expected_formats[suffix]:
        raise ValueError("figure extension does not match its encoded format")
    dpi: float | None = None
    if (
        isinstance(reported_dpi, (tuple, list))
        and len(reported_dpi) >= 2
        and all(isinstance(value, (int, float)) for value in reported_dpi[:2])
    ):
        dpi = min(float(reported_dpi[0]), float(reported_dpi[1]))
    elif isinstance(reported_dpi, (int, float)):
        dpi = float(reported_dpi)
    return {
        "media_type": FIGURE_MEDIA_TYPES[suffix],
        "width": width,
        "height": height,
        "bytes": size,
        "dpi": dpi,
    }


def _parse_tesseract_tsv(raw: bytes) -> dict[str, Any]:
    """Parse bounded Tesseract TSV into model-safe text and word boxes."""

    try:
        tsv = raw.decode("utf-8")
        reader = csv.DictReader(io.StringIO(tsv), delimiter="\t")
        words: list[dict[str, Any]] = []
        for row in reader:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            try:
                confidence = float(row.get("conf", "-1"))
                box = {
                    key: int(row.get(key, "0"))
                    for key in ("left", "top", "width", "height")
                }
            except (TypeError, ValueError):
                continue
            words.append({"text": text, "confidence": confidence, **box})
            if len(words) >= MAX_OCR_WORDS:
                break
    except (UnicodeError, csv.Error):
        return {"available": False, "reason": "tesseract_invalid_output"}
    text = " ".join(word["text"] for word in words)[:MAX_OCR_TEXT_CHARS]
    return {
        "available": True,
        "engine": "tesseract",
        "text": text,
        "words": words,
        "truncated": len(words) >= MAX_OCR_WORDS or len(text) >= MAX_OCR_TEXT_CHARS,
    }


@lru_cache(maxsize=256)
def _extract_figure_ocr_cached(
    path_text: str, size: int, mtime_ns: int
) -> dict[str, Any]:
    del size, mtime_ns
    path = Path(path_text)
    worker_url = os.environ.get("SCIENTIFIC_AGENT_SANDBOX_WORKER_URL", "").rstrip("/")
    worker_token = os.environ.get("SCIENTIFIC_AGENT_SANDBOX_WORKER_TOKEN", "")
    if worker_url and len(worker_token) >= 24:
        body = json.dumps(
            {"request_id": str(uuid.uuid4()), "figure_path": str(path)}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{worker_url}/extract-figure-ocr",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {worker_token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                payload = json.loads(response.read())
            if payload.get("figure_sha256") != sha256_file(path):
                return {"available": False, "reason": "ocr_worker_hash_mismatch"}
            ocr = payload.get("ocr")
            if isinstance(ocr, dict):
                return ocr
        except (OSError, ValueError, json.JSONDecodeError):
            return {"available": False, "reason": "ocr_worker_failed"}
        return {"available": False, "reason": "ocr_worker_invalid_output"}

    # Development fallback. Release containers always use the authenticated,
    # network-isolated sandbox worker instead of parsing model output in the web
    # process.

    executable = shutil.which("tesseract")
    if executable is None:
        return {"available": False, "reason": "tesseract_unavailable"}
    command = [executable, str(path), "stdout", "-l", "eng", "--psm", "3", "tsv"]
    bwrap = shutil.which("bwrap")
    # The release container has Debian's fixed OCR binary and Bubblewrap. Keep
    # the native image parser away from workspace data and the network; local
    # development environments with differently packaged binaries use the
    # direct command only for their own test fixtures.
    if executable == "/usr/bin/tesseract" and bwrap is not None:
        confined_image = f"/input/figure{path.suffix.lower()}"
        command = [
            bwrap,
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind-try",
            "/lib",
            "/lib",
            "--ro-bind-try",
            "/lib64",
            "/lib64",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/input",
            "--ro-bind",
            str(path),
            confined_image,
            "--chdir",
            "/tmp",
            executable,
            confined_image,
            "stdout",
            "-l",
            "eng",
            "--psm",
            "3",
            "tsv",
        ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=30,
            env={
                "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                "OMP_THREAD_LIMIT": "1",
                "LANG": "C.UTF-8",
            },
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "reason": "tesseract_failed"}
    if completed.returncode != 0:
        return {"available": False, "reason": "tesseract_failed"}
    return _parse_tesseract_tsv(completed.stdout)


def extract_figure_ocr(path: Path) -> dict[str, Any]:
    """Extract bounded visible text and geometry through the confined OCR path."""

    info = path.stat()
    return _extract_figure_ocr_cached(str(path), info.st_size, info.st_mtime_ns)


def read_table_preview(
    path: Path,
    *,
    row_limit: int = MAX_TABLE_ROWS,
    column_limit: int = MAX_TABLE_COLUMNS,
) -> dict[str, Any]:
    suffix = path.suffix.lower()
    delimiter = TABLE_DELIMITERS.get(suffix)
    if delimiter is None:
        raise ValueError("report tables must be CSV or TSV")
    size = path.stat().st_size
    if size < 1 or size > MAX_TABLE_BYTES:
        raise ValueError("table has an invalid file size")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter, strict=True)
            header = next(reader)
            if not header or any(not cell.strip() for cell in header):
                raise ValueError("table requires nonempty headers")
            if len(header) != len(set(header)):
                raise ValueError("table headers must be unique")
            rows: list[list[str]] = []
            total_rows = 0
            for row in reader:
                total_rows += 1
                if len(row) != len(header):
                    raise ValueError("table rows must be rectangular")
                if len(rows) < row_limit:
                    rows.append(row[:column_limit])
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError("table is not strict UTF-8 delimited text") from exc
    return {
        "columns": header[:column_limit],
        "rows": rows,
        "total_rows": total_rows,
        "total_columns": len(header),
        "truncated": total_rows > row_limit or len(header) > column_limit,
        "bytes": size,
        "media_type": "text/csv" if suffix == ".csv" else "text/tab-separated-values",
    }


def excessive_table_precision(
    preview: dict[str, Any],
    *,
    maximum_digits: int = MAX_READER_TABLE_SIGNIFICANT_DIGITS,
    example_limit: int = 5,
) -> list[str]:
    """Return bounded locations of reader-table cells with raw-like precision."""

    examples: list[str] = []
    columns = preview.get("columns", [])
    for row_number, row in enumerate(preview.get("rows", []), start=1):
        for column_number, cell in enumerate(row, start=1):
            value = cell.strip()
            if not _NUMBER_CELL.fullmatch(value) or not any(
                marker in value.lower() for marker in (".", "e")
            ):
                continue
            mantissa = value.lower().split("e", 1)[0].lstrip("+-")
            digits = mantissa.replace(".", "").lstrip("0")
            significant_digits = len(digits) if digits else 1
            if significant_digits <= maximum_digits:
                continue
            column = (
                str(columns[column_number - 1])
                if column_number <= len(columns)
                else str(column_number)
            )
            examples.append(f"row {row_number}, column {column}: {value}")
            if len(examples) >= example_limit:
                return examples
    return examples


def prepare_display_audit(
    report: ScientificReport,
    computation: ComputationEvidence,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Return actual raster paths and bounded table previews for Gemma's audit."""

    images: list[Path] = []
    tables: list[dict[str, Any]] = []
    figure_number = 0
    table_number = 0
    registered_paths: set[str] = set()
    for display in report.displays:
        source = resolve_display_artifact(display, computation)
        registered_paths.add(str(source.resolve()))
        if display.kind == "figure":
            figure_number += 1
            metadata = inspect_figure(source)
            images.append(source)
            tables.append(
                {
                    "display_id": display.display_id,
                    "kind": "figure",
                    "number": figure_number,
                    "registered": True,
                    "artifact_path": str(source),
                    "sha256": sha256_file(source),
                    "ocr": extract_figure_ocr(source),
                    **metadata,
                }
            )
        else:
            table_number += 1
            tables.append(
                {
                    "display_id": display.display_id,
                    "kind": "table",
                    "number": table_number,
                    "registered": True,
                    "artifact_path": str(source),
                    "sha256": sha256_file(source),
                    **read_table_preview(source),
                }
            )

    # A missing ReportDisplay must not hide the underlying artifact from the
    # first visual audit. Audit the latest version of every successful logical
    # figure/table path so a single bounded repair can fix registration and the
    # actual display together.
    latest_outputs: dict[str, tuple[Path, str | None]] = {}
    for artifact in computation.artifacts:
        source = Path(artifact.path)
        key = logical_report_output_key(source)
        if key is not None:
            latest_outputs[key] = (source, artifact.sha256)
    for key, (source, expected_hash) in latest_outputs.items():
        resolved = source.resolve()
        if str(resolved) in registered_paths or not resolved.is_file():
            continue
        actual_hash = sha256_file(resolved)
        if expected_hash and actual_hash != expected_hash:
            continue
        kind = key.split("/", 1)[0]
        display_id = f"unregistered:{key}"
        if kind == "figures":
            figure_number += 1
            metadata = inspect_figure(resolved)
            images.append(resolved)
            tables.append(
                {
                    "display_id": display_id,
                    "kind": "figure",
                    "number": figure_number,
                    "registered": False,
                    "artifact_path": str(resolved),
                    "sha256": actual_hash,
                    "ocr": extract_figure_ocr(resolved),
                    **metadata,
                }
            )
        else:
            table_number += 1
            tables.append(
                {
                    "display_id": display_id,
                    "kind": "table",
                    "number": table_number,
                    "registered": False,
                    "artifact_path": str(resolved),
                    "sha256": actual_hash,
                    **read_table_preview(resolved),
                }
            )
    return images, tables


def materialize_displays(
    run_dir: Path,
    report: ScientificReport,
    computation: ComputationEvidence,
) -> dict[str, Any]:
    """Copy registered displays into a portable, path-confined report bundle."""

    destination_root = run_dir / "displays"
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, mode=0o700)
    entries: list[dict[str, Any]] = []
    counters = {"figure": 0, "table": 0}
    for display in report.displays:
        source = resolve_display_artifact(display, computation)
        counters[display.kind] += 1
        suffix = source.suffix.lower()
        destination = destination_root / f"{display.display_id}{suffix}"
        shutil.copy2(source, destination)
        destination.chmod(0o600)
        entry: dict[str, Any] = {
            **display.model_dump(mode="json"),
            "number": counters[display.kind],
            "path": destination.relative_to(run_dir).as_posix(),
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        }
        if display.kind == "figure":
            entry.update(inspect_figure(destination))
        else:
            entry.update(read_table_preview(destination))
        entries.append(entry)
    manifest = {"version": 1, "displays": entries}
    write_json(run_dir / "display_manifest.json", manifest)
    return manifest


def _escape_table_cell(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _display_markdown(entry: dict[str, Any]) -> list[str]:
    label = "Figure" if entry["kind"] == "figure" else "Table"
    number = entry["number"]
    lines = ["", f"**{label} {number}. {entry['title']}**", ""]
    if entry["kind"] == "figure":
        lines.extend(
            [
                f"![{entry['alt_text']}]({entry['path']})",
                "",
                entry["caption"],
            ]
        )
    else:
        columns = [_escape_table_cell(str(item)) for item in entry["columns"]]
        lines.extend(
            [
                "| " + " | ".join(columns) + " |",
                "| " + " | ".join("---" for _ in columns) + " |",
            ]
        )
        for row in entry["rows"]:
            lines.append(
                "| " + " | ".join(_escape_table_cell(str(item)) for item in row) + " |"
            )
        if entry["truncated"]:
            lines.extend(["", "_Preview truncated; use the complete artifact below._"])
        lines.extend(["", entry["caption"], "", f"[Complete table]({entry['path']})"])
    evidence = ", ".join(entry.get("evidence_refs", []))
    if evidence:
        lines.extend(["", f"Evidence: {evidence}"])
    return lines


def render_report_markdown(
    report: ScientificReport,
    display_manifest: dict[str, Any] | None = None,
    reference_manifest: dict[str, Any] | None = None,
) -> str:
    """Render a portable article with an exact, controller-owned heading order."""

    displays = (display_manifest or {}).get("displays", [])
    by_placement: dict[str, list[dict[str, Any]]] = {
        "methods": [],
        "results": [],
        "discussion": [],
    }
    for display in displays:
        by_placement[display["placement"]].append(display)

    lines = [
        f"# {report.title}",
        "",
        "## Abstract",
        "",
        report.executive_summary,
        "",
        "## Introduction",
        "",
        report.introduction,
        "",
        "## Methods",
        "",
    ]
    lines.extend(f"- {method}" for method in report.methods)
    for entry in by_placement["methods"]:
        lines.extend(_display_markdown(entry))
    lines.extend(["", "## Results", "", report.results])
    for entry in by_placement["results"]:
        lines.extend(_display_markdown(entry))
    lines.extend(["", "## Discussion", "", report.discussion])
    if report.limitations:
        lines.extend(["", "### Limitations", ""])
        lines.extend(f"- {item}" for item in report.limitations)
    for entry in by_placement["discussion"]:
        lines.extend(_display_markdown(entry))
    lines.extend(["", "## Conclusions", "", report.conclusions])
    lines.extend(["", "## Evidence ledger", ""])
    for claim in report.claims:
        refs = ", ".join(claim.evidence_refs) or "none"
        lines.append(
            f"- **{claim.claim_id} [{claim.status.value}]** {claim.text} "
            f"(evidence: {refs})"
        )
    lines.extend(["", "## Sources", ""])
    local_references = {
        item["source_id"]: item
        for item in (reference_manifest or {}).get("references", [])
    }
    for source in report.sources:
        if source.url is not None:
            local = local_references.get(source.source_id, {})
            markdown = local.get("markdown")
            pdf = local.get("pdf")
            title = (
                f"[{source.title}]({markdown['path']})" if markdown else source.title
            )
            links = []
            if pdf:
                links.append(f"[PDF]({pdf['path']})")
            if markdown:
                label = (
                    "Local abstract"
                    if source.full_text_status == "abstract_only"
                    else "Markdown"
                )
                links.append(f"[{label}]({markdown['path']})")
            links.append(f"[Canonical record]({source.url})")
            identifiers = []
            if source.pmid:
                identifiers.append(f"PMID {source.pmid}")
            if source.doi:
                identifiers.append(f"DOI {source.doi}")
            suffix = "; ".join([*identifiers, *links])
            lines.append(f"- **{source.source_id}:** {title} — {suffix}")
        else:
            # Artifact locations are deliberately rendered as non-clickable labels;
            # the portable bundle exposes registered files through its manifest.
            artifact_name = Path(source.artifact_path or "artifact").name
            lines.append(f"- **{source.source_id}:** {source.title} ({artifact_name})")
    if report.unresolved_issues:
        lines.extend(["", "## Unresolved issues", ""])
        lines.extend(f"- {item}" for item in report.unresolved_issues)
    return "\n".join(lines) + "\n"


def materialize_references(
    run_dir: Path,
    report: ScientificReport,
    retrieval: RetrievalEvidence,
) -> dict[str, Any]:
    """Copy controller-verified local literature files into the portable run bundle."""

    known = {str(Path(item).resolve()) for item in retrieval.artifacts}
    entries: list[dict[str, Any]] = []
    for source in report.sources:
        if source.url is None:
            continue
        entry: dict[str, Any] = {
            "source_id": source.source_id,
            "title": source.title,
            "canonical_url": str(source.url),
            "doi": source.doi,
            "pmid": source.pmid,
            "pmcid": source.pmcid,
            "citekey": source.citekey,
            "license": source.license,
            "rights_status": source.rights_status,
            "terms_warning": source.terms_warning,
            "retracted": source.retracted,
            "full_text_status": source.full_text_status,
            "pdf": None,
            "markdown": None,
        }
        for kind, value, suffix in (
            ("pdf", source.local_pdf_path, ".pdf"),
            ("markdown", source.local_markdown_path, ".md"),
        ):
            if value is None:
                continue
            origin = Path(value).resolve()
            if str(origin) not in known or not origin.is_file() or origin.is_symlink():
                raise ValueError(
                    f"source {source.source_id} has an unverified local {kind} artifact"
                )
            if origin.suffix.lower() != suffix:
                raise ValueError(
                    f"source {source.source_id} local {kind} has the wrong file type"
                )
            stem = source.citekey or re.sub(
                r"[^a-z0-9-]+", "-", source.source_id.lower()
            ).strip("-")
            if not stem:
                raise ValueError("source cannot be assigned a portable reference name")
            relative = (
                Path("references")
                / ("pdfs" if kind == "pdf" else "markdown")
                / f"{stem}{suffix}"
            )
            destination = run_dir / relative
            destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            digest = sha256_file(origin)
            if destination.exists() and sha256_file(destination) != digest:
                raise ValueError(f"conflicting local literature artifact: {relative}")
            if not destination.exists():
                shutil.copy2(origin, destination)
                destination.chmod(0o600)
            entry[kind] = {
                "path": relative.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": digest,
            }
        entries.append(entry)
    manifest = {"references": entries}
    write_json(run_dir / "reference_manifest.json", manifest)
    return manifest


def display_media_type(path: Path) -> str:
    return FIGURE_MEDIA_TYPES.get(
        path.suffix.lower(),
        mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    )


def caption_has_number_prefix(caption: str) -> bool:
    return bool(re.match(r"^\s*(?:figure|fig\.?|table)\s+\d+\b", caption, re.I))
