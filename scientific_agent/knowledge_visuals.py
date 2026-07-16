"""Bounded deterministic raster extraction for Gemma-only knowledge indexing."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import signal
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_VISUAL_ASSETS = 40
MAX_SOURCE_PIXELS = 100_000_000
MAX_VISUAL_BYTES = 4 * 1024 * 1024
MAX_VISUAL_TOTAL_BYTES = 160 * 1024 * 1024
MAX_ARCHIVE_VISUAL_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 2_000
MAX_COMPRESSION_RATIO = 200
MAX_RENDER_DIMENSION = 3_000
MAX_RASTERIZER_OUTPUTS = MAX_VISUAL_ASSETS * 2
RASTERIZER_TIMEOUT_SECONDS = 180
RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
OFFICE_MEDIA = re.compile(r"^(?:word|ppt|xl)/media/[^/]+$")


def is_direct_raster(path: Path) -> bool:
    return path.suffix.casefold() in RASTER_SUFFIXES


def _normalize(source: Path, destination: Path) -> dict[str, Any] | None:
    try:
        with Image.open(source) as opened:
            width, height = opened.size
            if width < 1 or height < 1 or width * height > MAX_SOURCE_PIXELS:
                return None
            image = ImageOps.exif_transpose(opened)
            image.thumbnail(
                (MAX_RENDER_DIMENSION, MAX_RENDER_DIMENSION), Image.Resampling.LANCZOS
            )
            if image.mode not in {"RGB", "L"}:
                background = Image.new("RGB", image.size, "white")
                if "A" in image.getbands():
                    background.paste(image, mask=image.getchannel("A"))
                else:
                    background.paste(image)
                image = background
            elif image.mode == "L":
                image = image.convert("RGB")
            else:
                image = image.copy()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        return None
    for quality in (90, 82, 74, 66):
        image.save(destination, format="JPEG", quality=quality, optimize=True)
        if destination.stat().st_size <= MAX_VISUAL_BYTES:
            break
        image.thumbnail(
            (max(640, image.width * 3 // 4), max(640, image.height * 3 // 4)),
            Image.Resampling.LANCZOS,
        )
    if destination.stat().st_size > MAX_VISUAL_BYTES:
        destination.unlink(missing_ok=True)
        return None
    destination.chmod(0o600)
    return {
        "path": destination,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "bytes": destination.stat().st_size,
        "width": image.width,
        "height": image.height,
        "media_type": mimetypes.guess_type(destination.name)[0] or "image/jpeg",
    }


def _office_candidates(source: Path, scratch: Path) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    try:
        with zipfile.ZipFile(source) as archive:
            archive_members = archive.infolist()
            if len(archive_members) > MAX_ARCHIVE_MEMBERS:
                return []
            if (
                sum(item.file_size for item in archive_members)
                > MAX_ARCHIVE_TOTAL_BYTES
            ):
                return []
            if any(
                item.compress_size
                and item.file_size / item.compress_size > MAX_COMPRESSION_RATIO
                for item in archive_members
            ):
                return []
            members = [
                item
                for item in archive_members
                if not item.is_dir() and OFFICE_MEDIA.fullmatch(item.filename)
            ]
            total = 0
            for item in members[: MAX_VISUAL_ASSETS * 2]:
                suffix = Path(item.filename).suffix.casefold()
                if (
                    suffix not in RASTER_SUFFIXES
                    or item.file_size > MAX_ARCHIVE_VISUAL_BYTES
                ):
                    continue
                total += item.file_size
                if total > MAX_ARCHIVE_TOTAL_BYTES:
                    break
                path = scratch / f"office-{len(candidates):04d}{suffix}"
                path.write_bytes(archive.read(item))
                candidates.append((path, item.filename))
    except (OSError, ValueError, zipfile.BadZipFile):
        return []
    return candidates


def _run_bounded_rasterizer(command: list[str], scratch: Path) -> bool:
    """Run a parser while enforcing output count/volume and process limits."""

    limited = [
        "/usr/bin/prlimit",
        f"--fsize={MAX_ARCHIVE_VISUAL_BYTES}",
        "--as=2147483648",
        "--nproc=16",
        f"--cpu={RASTERIZER_TIMEOUT_SECONDS}",
        "--",
        *command,
    ]
    try:
        process = subprocess.Popen(
            limited,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            start_new_session=True,
        )
    except OSError:
        return False
    deadline = time.monotonic() + RASTERIZER_TIMEOUT_SECONDS
    while process.poll() is None:
        outputs = [item for item in scratch.iterdir() if item.is_file()]
        try:
            total = sum(item.stat().st_size for item in outputs)
        except OSError:
            total = MAX_ARCHIVE_TOTAL_BYTES + 1
        if (
            len(outputs) > MAX_RASTERIZER_OUTPUTS
            or total > MAX_ARCHIVE_TOTAL_BYTES
            or time.monotonic() >= deadline
        ):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
            process.wait(timeout=10)
            return False
        time.sleep(0.1)
    outputs = [item for item in scratch.iterdir() if item.is_file()]
    if (
        process.returncode != 0
        or len(outputs) > MAX_RASTERIZER_OUTPUTS
        or sum(item.stat().st_size for item in outputs) > MAX_ARCHIVE_TOTAL_BYTES
    ):
        return False
    return True


def _pdf_candidates(source: Path, scratch: Path) -> list[tuple[Path, str]]:
    prefix = scratch / "pdf-page"
    if not _run_bounded_rasterizer(
        [
            "/usr/bin/pdftoppm",
            "-f",
            "1",
            "-l",
            str(MAX_VISUAL_ASSETS),
            "-r",
            "180",
            "-jpeg",
            str(source),
            str(prefix),
        ],
        scratch,
    ):
        return []
    return [
        (path, f"PDF page {index + 1}")
        for index, path in enumerate(sorted(scratch.glob("pdf-page-*")))
        if path.is_file() and not path.is_symlink()
    ]


def extract_visual_candidates(source: Path, staging: Path) -> list[dict[str, Any]]:
    """Extract/normalize bounded rasters; model interpretation happens elsewhere."""

    visual_root = staging / "visuals"
    visual_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="evidence-knowledge-visuals-") as temporary:
        scratch = Path(temporary)
        scratch.chmod(0o700)
        if is_direct_raster(source):
            candidates = [(source, source.name)]
        elif source.suffix.casefold() == ".pdf":
            candidates = _pdf_candidates(source, scratch)
        elif source.suffix.casefold() in {".docx", ".pptx", ".xlsx"}:
            candidates = _office_candidates(source, scratch)
        else:
            candidates = []
        results: list[dict[str, Any]] = []
        total = 0
        for candidate, label in candidates:
            if len(results) >= MAX_VISUAL_ASSETS:
                break
            destination = visual_root / f"visual-{len(results):04d}.jpg"
            normalized = _normalize(candidate, destination)
            if normalized is None:
                continue
            total += int(normalized["bytes"])
            if total > MAX_VISUAL_TOTAL_BYTES:
                destination.unlink(missing_ok=True)
                break
            visual_id = (
                "kv-"
                + hashlib.sha256(
                    f"{len(results)}:{normalized['sha256']}:{label}".encode()
                ).hexdigest()[:24]
            )
            results.append(
                {
                    **normalized,
                    "id": visual_id,
                    "ordinal": len(results),
                    "source_label": label[:500],
                    "extractor": "direct-raster-normalizer"
                    if candidate == source and is_direct_raster(source)
                    else "pdftoppm-page-normalizer"
                    if source.suffix.casefold() == ".pdf"
                    else "ooxml-media-normalizer",
                    "extractor_version": "knowledge-visuals-v1",
                }
            )
        return results


def register_document_visuals(library: Any, document_id: str) -> list[dict[str, Any]]:
    """Extract normalized rasters and register only their controller-owned bytes."""

    source = library.source_path(document_id)
    with tempfile.TemporaryDirectory(prefix="evidence-knowledge-register-") as raw:
        temporary = Path(raw)
        document = library.get_document(document_id)
        suffix = Path(document["filename"]).suffix.casefold()
        staged_source = temporary / f"source{suffix}"
        shutil.copyfile(source, staged_source)
        candidates = extract_visual_candidates(staged_source, temporary)
        return [
            library.register_visual_asset(
                document_id,
                Path(item["path"]),
                source_label=(
                    document["filename"]
                    if item["extractor"] == "direct-raster-normalizer"
                    else item["source_label"]
                ),
                sha256=item["sha256"],
            )
            for item in candidates
        ]
