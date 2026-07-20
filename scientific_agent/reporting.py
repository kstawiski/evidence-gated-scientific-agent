"""Deterministic registration and rendering of scientific report displays."""

from __future__ import annotations

import csv
from functools import lru_cache
import io
import json
import math
import mimetypes
import os
import re
import shutil
import statistics
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
MAX_LAYOUT_OVERLAP_EXAMPLES = 8
_NUMBER_CELL = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_WORDLIKE = re.compile(r"[A-Za-z]{2,}")
_LEGEND_CUES = {
    "analysis",
    "ancova",
    "cohort",
    "control",
    "group",
    "groups",
    "mean",
    "observation",
    "observations",
    "placebo",
    "primary",
    "sd",
    "sensitivity",
    "treatment",
}


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


@lru_cache(maxsize=256)
def _excessive_internal_blank_band_cached(
    path_text: str, size: int, mtime_ns: int
) -> dict[str, Any] | None:
    """Locate an extreme blank band separating visible figure content."""

    del size, mtime_ns
    try:
        with Image.open(path_text) as source:
            source.thumbnail((512, 512), Image.Resampling.BILINEAR)
            if "A" in source.getbands():
                foreground = source.convert("RGBA")
                background = Image.new("RGBA", foreground.size, "white")
                background.alpha_composite(foreground)
                image = background.convert("RGB")
            else:
                image = source.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    width, height = image.size
    if width < 32 or height < 32:
        return None
    pixels = image.load()
    minimum_nonwhite = max(2, math.ceil(width * 0.01))
    content_rows = []
    for y in range(height):
        nonwhite = sum(1 for x in range(width) if min(pixels[x, y]) < 245)
        content_rows.append(nonwhite >= minimum_nonwhite)
    try:
        first = content_rows.index(True)
        last = len(content_rows) - 1 - content_rows[::-1].index(True)
    except ValueError:
        return None
    longest_start = longest_end = first
    cursor = first
    while cursor <= last:
        if content_rows[cursor]:
            cursor += 1
            continue
        start = cursor
        while cursor <= last and not content_rows[cursor]:
            cursor += 1
        if cursor - start > longest_end - longest_start:
            longest_start, longest_end = start, cursor
    gap_rows = longest_end - longest_start
    content_above = sum(content_rows[first:longest_start])
    content_below = sum(content_rows[longest_end : last + 1])
    if (
        gap_rows < max(32, math.ceil(height * 0.45))
        or content_above < 2
        or content_below < 2
    ):
        return None
    return {
        "gap_fraction": round(gap_rows / height, 4),
        "gap_start_fraction": round(longest_start / height, 4),
        "gap_end_fraction": round(longest_end / height, 4),
        "sample_width": width,
        "sample_height": height,
    }


def excessive_internal_blank_band(path: Path) -> dict[str, Any] | None:
    """Return geometry for a clearly excessive blank band, if present."""

    info = path.stat()
    return _excessive_internal_blank_band_cached(
        str(path), info.st_size, info.st_mtime_ns
    )


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


def _figure_layout_review_questions(
    path: Path,
    ocr: dict[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Turn bounded OCR/raster geometry into questions for Gemma to clear.

    These signals prioritize visual inspection; they never decide whether pixels
    are acceptable. Gemma remains the sole image-understanding authority.
    """

    words: list[dict[str, Any]] = []
    for raw in ocr.get("words", []) if isinstance(ocr, dict) else []:
        text = str(raw.get("text") or "").strip()[:80]
        if not _WORDLIKE.search(text):
            continue
        try:
            left = max(0, int(raw["left"]))
            top = max(0, int(raw["top"]))
            word_width = max(0, int(raw["width"]))
            word_height = max(0, int(raw["height"]))
            confidence = float(raw.get("confidence", -1))
        except (KeyError, TypeError, ValueError):
            continue
        right = min(width, left + word_width)
        bottom = min(height, top + word_height)
        if right <= left or bottom <= top:
            continue
        words.append(
            {
                "text": text,
                "confidence": round(confidence, 1),
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": right - left,
                "height": bottom - top,
            }
        )

    overlaps: list[dict[str, Any]] = []
    top_band_count = 0
    for index, first in enumerate(words):
        for second in words[index + 1 :]:
            intersection_width = max(
                0,
                min(first["right"], second["right"])
                - max(first["left"], second["left"]),
            )
            intersection_height = max(
                0,
                min(first["bottom"], second["bottom"])
                - max(first["top"], second["top"]),
            )
            if not intersection_width or not intersection_height:
                continue
            smaller_area = min(
                first["width"] * first["height"],
                second["width"] * second["height"],
            )
            overlap_fraction = intersection_width * intersection_height / smaller_area
            if overlap_fraction < 0.20:
                continue
            if (
                first["text"].casefold() == second["text"].casefold()
                and overlap_fraction >= 0.90
            ):
                continue
            in_top_band = max(first["top"], second["top"]) < height * 0.22
            top_band_count += int(in_top_band)
            overlaps.append(
                {
                    "texts": [first["text"], second["text"]],
                    "confidence": [first["confidence"], second["confidence"]],
                    "overlap_fraction_of_smaller_box": round(overlap_fraction, 3),
                    "in_top_22_percent": in_top_band,
                    "union_box_fraction": [
                        round(min(first["left"], second["left"]) / width, 4),
                        round(min(first["top"], second["top"]) / height, 4),
                        round(max(first["right"], second["right"]) / width, 4),
                        round(max(first["bottom"], second["bottom"]) / height, 4),
                    ],
                }
            )
    overlaps.sort(
        key=lambda item: item["overlap_fraction_of_smaller_box"], reverse=True
    )
    annotation_overlap_candidates = figure_annotation_overlap_candidates(
        path,
        ocr,
        width=width,
        height=height,
    )

    cue_words = []
    for word in words:
        normalized = re.sub(r"[^a-z]", "", word["text"].casefold())
        center_x = (word["left"] + word["right"]) / 2
        center_y = (word["top"] + word["bottom"]) / 2
        if (
            normalized in _LEGEND_CUES
            and center_x > width * 0.45
            and height * 0.08 < center_y < height * 0.75
        ):
            cue_words.append(word)

    legend_region: dict[str, Any] | None = None
    if len(cue_words) >= 4:
        ordered_heights = sorted(word["height"] for word in cue_words)
        median_height = ordered_heights[len(ordered_heights) // 2]
        pad_y = max(2 * median_height, int(height * 0.02))
        pad_x = max(2 * median_height, int(width * 0.04))
        left = max(
            0,
            min(word["left"] for word in cue_words) - max(pad_x, int(width * 0.08)),
        )
        top = max(0, min(word["top"] for word in cue_words) - pad_y)
        right = min(width, max(word["right"] for word in cue_words) + pad_x)
        bottom = min(height, max(word["bottom"] for word in cue_words) + pad_y)
        color_start = left + int((right - left) * 0.40)
        chromatic_fraction = 0.0
        try:
            with Image.open(path) as image:
                sample = image.convert("RGB").crop((color_start, top, right, bottom))
            sample.thumbnail((512, 512), Image.Resampling.BILINEAR)
            pixels = sample.load()
            pixel_count = sample.width * sample.height
            chromatic_pixels = sum(
                1
                for y in range(sample.height)
                for x in range(sample.width)
                if max(pixels[x, y]) - min(pixels[x, y]) >= 40
            )
            if pixel_count:
                chromatic_fraction = chromatic_pixels / pixel_count
        except (OSError, ValueError, Image.DecompressionBombError):
            chromatic_fraction = 0.0
        legend_region = {
            "candidate_box_fraction": [
                round(left / width, 4),
                round(top / height, 4),
                round(right / width, 4),
                round(bottom / height, 4),
            ],
            "cue_words": [word["text"] for word in cue_words[:16]],
            "chromatic_pixel_fraction_beyond_key_zone": round(chromatic_fraction, 4),
            "priority": "high" if chromatic_fraction >= 0.005 else "routine",
        }

    return {
        "source": "controller_ocr_and_raster_geometry",
        "pixel_interpretation_authority": "Gemma",
        "top_text_clearance": {
            "required": True,
            "candidate_overlap_count": len(overlaps),
            "candidate_overlap_count_in_top_22_percent": top_band_count,
            "priority": "high" if top_band_count >= 2 else "routine",
            "examples": overlaps[:MAX_LAYOUT_OVERLAP_EXAMPLES],
            "question": (
                "Inspect the top band at native detail: are title, subtitle, "
                "test label, estimate, and interval mutually separated?"
            ),
        },
        "legend_data_clearance": {
            "required": True,
            "candidate": legend_region,
            "question": (
                "Locate every legend and trace its full rectangle: does it cover "
                "any data point, error bar, annotation, or statistical text?"
            ),
        },
        "annotation_data_clearance": {
            "required": True,
            "candidate_count": len(annotation_overlap_candidates),
            "priority": "high" if annotation_overlap_candidates else "routine",
            "examples": annotation_overlap_candidates[:MAX_LAYOUT_OVERLAP_EXAMPLES],
            "question": (
                "Inspect every in-panel annotation at native detail: does any point, "
                "confidence interval, error bar, or other data mark cross its text?"
            ),
        },
    }


def figure_annotation_overlap_candidates(
    path: Path,
    ocr: dict[str, Any],
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Detect high-confidence colored marks merged into horizontal OCR words.

    Tesseract commonly expands a word box across adjacent lines when a plotted
    point or interval crosses annotation text. Rotated axis labels are excluded,
    and a candidate must also contain a material fraction of chromatic pixels.
    The result is deliberately narrow: ordinary OCR uncertainty remains Gemma's
    responsibility, while this reproducible geometry anomaly fails closed.
    """

    horizontal: list[dict[str, Any]] = []
    for raw in ocr.get("words", []) if isinstance(ocr, dict) else []:
        text = str(raw.get("text") or "").strip()[:80]
        if not re.search(r"[A-Za-z0-9]{2,}", text):
            continue
        try:
            left = max(0, int(raw["left"]))
            top = max(0, int(raw["top"]))
            word_width = max(0, int(raw["width"]))
            word_height = max(0, int(raw["height"]))
            confidence = float(raw.get("confidence", -1))
        except (KeyError, TypeError, ValueError):
            continue
        right = min(width, left + word_width)
        bottom = min(height, top + word_height)
        word_width = right - left
        word_height = bottom - top
        if (
            confidence < 50
            or word_width <= 0
            or word_height <= 0
            or word_width / word_height < 0.75
            or word_height > height * 0.16
        ):
            continue
        horizontal.append(
            {
                "text": text,
                "confidence": round(confidence, 1),
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": word_width,
                "height": word_height,
            }
        )
    if len(horizontal) < 4:
        return []
    reference_heights = [
        word["height"] for word in horizontal if word["height"] <= height * 0.08
    ]
    if len(reference_heights) < 3:
        return []
    median_height = float(statistics.median(reference_heights))
    minimum_height = max(median_height * 2.35, height * 0.045)
    candidates: list[dict[str, Any]] = []
    try:
        with Image.open(path) as image:
            raster = image.convert("RGB")
            for word in horizontal:
                if word["confidence"] < 80 or word["height"] < minimum_height:
                    continue
                sample = raster.crop(
                    (word["left"], word["top"], word["right"], word["bottom"])
                )
                pixels = list(sample.get_flattened_data())
                if not pixels:
                    continue
                chromatic_fraction = sum(
                    max(pixel) - min(pixel) >= 40 for pixel in pixels
                ) / len(pixels)
                if chromatic_fraction < 0.02:
                    continue
                candidates.append(
                    {
                        "text": word["text"],
                        "confidence": word["confidence"],
                        "height_vs_median": round(word["height"] / median_height, 2),
                        "chromatic_pixel_fraction": round(chromatic_fraction, 4),
                        "box_fraction": [
                            round(word["left"] / width, 4),
                            round(word["top"] / height, 4),
                            round(word["right"] / width, 4),
                            round(word["bottom"] / height, 4),
                        ],
                    }
                )
    except (OSError, ValueError, Image.DecompressionBombError):
        return []
    candidates.sort(
        key=lambda item: (
            item["height_vs_median"],
            item["chromatic_pixel_fraction"],
        ),
        reverse=True,
    )
    return candidates


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
            ocr = extract_figure_ocr(source)
            images.append(source)
            tables.append(
                {
                    "display_id": display.display_id,
                    "kind": "figure",
                    "number": figure_number,
                    "registered": True,
                    "artifact_path": str(source),
                    "sha256": sha256_file(source),
                    "ocr": ocr,
                    "layout_review_questions": _figure_layout_review_questions(
                        source,
                        ocr,
                        width=metadata["width"],
                        height=metadata["height"],
                    ),
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
            ocr = extract_figure_ocr(resolved)
            images.append(resolved)
            tables.append(
                {
                    "display_id": display_id,
                    "kind": "figure",
                    "number": figure_number,
                    "registered": False,
                    "artifact_path": str(resolved),
                    "sha256": actual_hash,
                    "ocr": ocr,
                    "layout_review_questions": _figure_layout_review_questions(
                        resolved,
                        ocr,
                        width=metadata["width"],
                        height=metadata["height"],
                    ),
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
    *,
    validated: bool = True,
    quality_status: str | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Copy registered displays into a portable, path-confined report bundle."""

    destination_root = run_dir / "displays"
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, mode=0o700)
    entries: list[dict[str, Any]] = []
    omissions: list[dict[str, str]] = []
    counters = {"figure": 0, "table": 0}
    for display in report.displays:
        counters[display.kind] += 1
        try:
            source = resolve_display_artifact(display, computation)
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
        except (OSError, ValueError):
            if not allow_partial:
                raise
            omissions.append(
                {
                    "display_id": display.display_id,
                    "kind": display.kind,
                    "reason": "artifact unavailable or invalid",
                }
            )
    manifest = {
        "version": 1,
        "validated": validated,
        "quality_status": quality_status,
        "displays": entries,
        "omissions": omissions,
    }
    write_json(run_dir / "display_manifest.json", manifest)
    return manifest


def describe_available_displays(
    report: ScientificReport,
    computation: ComputationEvidence,
    *,
    validated: bool,
    quality_status: str,
) -> dict[str, Any]:
    """Describe valid displays without mutating a completed historical run."""

    entries: list[dict[str, Any]] = []
    omissions: list[dict[str, str]] = []
    counters = {"figure": 0, "table": 0}
    for display in report.displays:
        counters[display.kind] += 1
        try:
            source = resolve_display_artifact(display, computation)
            entry: dict[str, Any] = {
                **display.model_dump(mode="json"),
                "number": counters[display.kind],
                "path": None,
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
            }
            if display.kind == "figure":
                entry.update(inspect_figure(source))
            else:
                entry.update(read_table_preview(source))
            entries.append(entry)
        except (OSError, ValueError):
            omissions.append(
                {
                    "display_id": display.display_id,
                    "kind": display.kind,
                    "reason": "artifact unavailable or invalid",
                }
            )
    return {
        "version": 1,
        "validated": validated,
        "quality_status": quality_status,
        "virtual": True,
        "displays": entries,
        "omissions": omissions,
    }


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
    quality_status: str | None = None,
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

    local_references = {
        item["source_id"]: item
        for item in (reference_manifest or {}).get("references", [])
    }
    external_sources = [source for source in report.sources if source.url is not None]
    citation_numbers = {
        source.source_id: index
        for index, source in enumerate(external_sources, start=1)
    }

    def citation_target(source_id: str) -> str:
        source = next(item for item in external_sources if item.source_id == source_id)
        local = local_references.get(source_id, {})
        markdown = local.get("markdown")
        return markdown["path"] if markdown else str(source.url)

    def cited_text(section: str, value: str) -> str:
        rendered = value
        for citation in report.inline_citations:
            if citation.section != section:
                continue
            links = ", ".join(
                f"[{citation_numbers[source_id]}]({citation_target(source_id)})"
                for source_id in citation.source_ids
                if source_id in citation_numbers
            )
            if links:
                rendered = rendered.replace(
                    citation.anchor_text,
                    f"{citation.anchor_text} [{links}]",
                    1,
                )
        return rendered

    lines = [
        f"# {report.title}",
        "",
    ]
    validated_statuses = {"supported", "supported_with_comments"}
    if quality_status is not None and quality_status not in validated_statuses:
        lines.extend(
            [
                "> **NOT VALIDATED** — The run quality status is "
                f"`{quality_status}`. Claim labels below are provisional model "
                "output and must not be treated as supported findings.",
                "",
            ]
        )
    lines.extend(
        [
            "## Abstract",
            "",
            cited_text("executive_summary", report.executive_summary),
            "",
            "## Introduction",
            "",
            cited_text("introduction", report.introduction),
            "",
            "## Methods",
            "",
        ]
    )
    lines.extend(f"- {cited_text('methods', method)}" for method in report.methods)
    for entry in by_placement["methods"]:
        lines.extend(_display_markdown(entry))
    lines.extend(["", "## Results", "", cited_text("results", report.results)])
    for entry in by_placement["results"]:
        lines.extend(_display_markdown(entry))
    lines.extend(["", "## Discussion", "", cited_text("discussion", report.discussion)])
    if report.limitations:
        lines.extend(["", "### Limitations", ""])
        lines.extend(f"- {item}" for item in report.limitations)
    for entry in by_placement["discussion"]:
        lines.extend(_display_markdown(entry))
    lines.extend(
        ["", "## Conclusions", "", cited_text("conclusions", report.conclusions)]
    )
    lines.extend(["", "## Evidence ledger", ""])
    for claim in report.claims:
        refs = ", ".join(claim.evidence_refs) or "none"
        claim_status = claim.status.value
        if quality_status is not None and quality_status not in validated_statuses:
            claim_status = f"model-labeled {claim_status}; run not validated"
        lines.append(
            f"- **{claim.claim_id} [{claim_status}]** {claim.text} (evidence: {refs})"
        )
    lines.extend(["", "## Sources", ""])
    for source in external_sources:
        local = local_references.get(source.source_id, {})
        markdown = local.get("markdown")
        pdf = local.get("pdf")
        title = f"[{source.title}]({markdown['path']})" if markdown else source.title
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
        number = citation_numbers[source.source_id]
        lines.append(f"{number}. **{source.source_id}:** {title} — {suffix}")
    artifact_sources = [source for source in report.sources if source.url is None]
    if artifact_sources:
        lines.extend(["", "### Computational and controller evidence", ""])
    for source in artifact_sources:
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
