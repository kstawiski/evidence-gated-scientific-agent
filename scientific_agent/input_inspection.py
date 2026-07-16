"""Bounded, value-free structural inspection of immutable run inputs.

This module deliberately reports shape, encoding, types, and missingness without
including cell values or estimating scientific effects.  It can therefore run
before the protocol lock without turning planning into post-hoc result review.
"""

from __future__ import annotations

import csv
import json
import math
import mimetypes
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

from PIL import Image, UnidentifiedImageError

from .provenance import sha256_file
from .schemas import InputColumnProfile, InputFileProfile, InputProfile

MAX_PROFILE_FILES = 200
MAX_TABULAR_ROWS = 250_000
MAX_COLUMNS = 256
MAX_DISTINCT_VALUES = 10_000
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 200
MAX_TEXT_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 128 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200

MISSING_TOKENS = {"", ".", "na", "n/a", "nan", "null", "none", "missing"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?$"
)


class _ColumnAccumulator:
    def __init__(self, name: str) -> None:
        self.name = name
        self.missing = 0
        self.non_missing = 0
        self.types: Counter[str] = Counter()
        self.distinct: set[str] = set()
        self.distinct_capped = False

    def add(self, value: Any) -> None:
        if value is None:
            self.missing += 1
            return
        text = str(value).strip()
        if text.casefold() in MISSING_TOKENS:
            self.missing += 1
            return
        self.non_missing += 1
        self.types[_infer_scalar_type(value, text)] += 1
        if len(self.distinct) < MAX_DISTINCT_VALUES:
            self.distinct.add(text)
        elif text not in self.distinct:
            self.distinct_capped = True

    def finish(self) -> InputColumnProfile:
        total = self.missing + self.non_missing
        ordered_types = [
            name
            for name, _ in sorted(
                self.types.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        return InputColumnProfile(
            name=self.name,
            inferred_types=ordered_types[:8],
            non_missing_count=self.non_missing,
            missing_count=self.missing,
            missing_fraction=(self.missing / total if total else 0.0),
            distinct_non_missing=len(self.distinct),
            distinct_count_capped=self.distinct_capped,
        )


def _infer_scalar_type(value: Any, text: str) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number" if math.isfinite(value) else "non_finite_number"
    folded = text.casefold()
    if folded in {"true", "false", "yes", "no"}:
        return "boolean"
    try:
        int(text)
    except ValueError:
        pass
    else:
        return "integer"
    try:
        number = float(text)
    except ValueError:
        pass
    else:
        return "number" if math.isfinite(number) else "non_finite_number"
    if DATETIME_RE.fullmatch(text):
        return "datetime"
    if DATE_RE.fullmatch(text):
        return "date"
    return "string"


def _deduplicate_headers(
    values: Iterable[Any], width: int
) -> tuple[list[str], list[str]]:
    headers: list[str] = []
    seen: Counter[str] = Counter()
    limitations: list[str] = []
    supplied = list(values)
    for index in range(width):
        raw = str(supplied[index]).strip() if index < len(supplied) else ""
        name = raw or f"unnamed_column_{index + 1}"
        seen[name] += 1
        if seen[name] > 1:
            limitations.append(f"duplicate column name: {name}")
            name = f"{name}__{seen[name]}"
        headers.append(name)
    return headers, list(dict.fromkeys(limitations))


def _encoding(path: Path) -> str:
    sample = path.read_bytes()[: 64 * 1024]
    for candidate in ("utf-8-sig", "utf-8"):
        try:
            sample.decode(candidate)
        except UnicodeDecodeError:
            continue
        return candidate
    return "latin-1"


def _delimiter(path: Path, encoding: str) -> str:
    if path.suffix.casefold() in {".tsv", ".tab"}:
        return "\t"
    sample = path.read_bytes()[: 64 * 1024].decode(encoding, errors="replace")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        return ","


def _profile_delimited(path: Path, base: dict[str, Any]) -> InputFileProfile:
    encoding = _encoding(path)
    delimiter = _delimiter(path, encoding)
    limitations: list[str] = []
    rows = 0
    completed = True
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header_row = next(reader)
        except StopIteration:
            return InputFileProfile(
                **base,
                detected_format="delimited_text",
                media_type="text/csv",
                inspection_status="complete",
                rows_observed=0,
                rows_total=0,
                details={"encoding": encoding, "delimiter": delimiter, "empty": True},
                limitations=[],
            )
        width = min(len(header_row), MAX_COLUMNS)
        if len(header_row) > MAX_COLUMNS:
            limitations.append(
                f"only the first {MAX_COLUMNS} of {len(header_row)} columns were profiled"
            )
        headers, header_limitations = _deduplicate_headers(header_row, width)
        limitations.extend(header_limitations)
        accumulators = [_ColumnAccumulator(name) for name in headers]
        for row in reader:
            if rows >= MAX_TABULAR_ROWS:
                completed = False
                break
            rows += 1
            if len(row) != len(header_row):
                limitations.append(
                    "at least one row has a different field count than the header"
                )
            for index, accumulator in enumerate(accumulators):
                accumulator.add(row[index] if index < len(row) else None)
    if not completed:
        limitations.append(
            f"row and missingness counts cover only the first {MAX_TABULAR_ROWS} data rows"
        )
    return InputFileProfile(
        **base,
        detected_format="delimited_text",
        media_type="text/tab-separated-values" if delimiter == "\t" else "text/csv",
        inspection_status="complete" if completed else "partial",
        rows_observed=rows,
        rows_total=rows if completed else None,
        columns=[item.finish() for item in accumulators],
        details={
            "encoding": encoding,
            "delimiter": "TAB" if delimiter == "\t" else delimiter,
            "declared_columns": len(header_row),
            "values_included_in_profile": False,
        },
        limitations=list(dict.fromkeys(limitations)),
    )


def _profile_records(
    records: list[Any],
) -> tuple[int, list[InputColumnProfile], list[str]]:
    keys: list[str] = []
    for record in records[:MAX_TABULAR_ROWS]:
        if isinstance(record, dict):
            for key in record:
                name = str(key)
                if name not in keys and len(keys) < MAX_COLUMNS:
                    keys.append(name)
    accumulators = [_ColumnAccumulator(name) for name in keys]
    for record in records[:MAX_TABULAR_ROWS]:
        if not isinstance(record, dict):
            continue
        for accumulator in accumulators:
            accumulator.add(record.get(accumulator.name))
    limitations: list[str] = []
    if len(records) > MAX_TABULAR_ROWS:
        limitations.append(
            f"row and missingness counts cover only the first {MAX_TABULAR_ROWS} records"
        )
    return (
        min(len(records), MAX_TABULAR_ROWS),
        [item.finish() for item in accumulators],
        limitations,
    )


def _profile_json(path: Path, base: dict[str, Any]) -> InputFileProfile:
    if path.stat().st_size > MAX_JSON_BYTES:
        return InputFileProfile(
            **base,
            detected_format="json",
            media_type="application/json",
            inspection_status="partial",
            details={"top_level": "not_loaded", "values_included_in_profile": False},
            limitations=[
                f"JSON exceeds the {MAX_JSON_BYTES} byte structural-inspection limit"
            ],
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return InputFileProfile(
            **base,
            detected_format="json",
            media_type="application/json",
            inspection_status="failed",
            details={},
            limitations=[f"JSON parsing failed ({type(exc).__name__})"],
        )
    details: dict[str, Any] = {
        "top_level": type(value).__name__,
        "values_included_in_profile": False,
    }
    if isinstance(value, list):
        observed, columns, limitations = _profile_records(value)
        details["items"] = len(value)
        return InputFileProfile(
            **base,
            detected_format="json",
            media_type="application/json",
            inspection_status="complete" if observed == len(value) else "partial",
            rows_observed=observed,
            rows_total=len(value),
            columns=columns,
            details=details,
            limitations=limitations,
        )
    if isinstance(value, dict):
        details["keys"] = sorted(str(key) for key in value)[:MAX_COLUMNS]
        details["key_count"] = len(value)
    return InputFileProfile(
        **base,
        detected_format="json",
        media_type="application/json",
        inspection_status="complete",
        details=details,
        limitations=[],
    )


def _profile_image(path: Path, base: dict[str, Any]) -> InputFileProfile:
    try:
        with Image.open(path) as image:
            details = {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "frames": int(getattr(image, "n_frames", 1)),
                "pixel_interpretation": "Gemma-only preplanning review",
            }
            media_type = Image.MIME.get(image.format or "", "application/octet-stream")
            detected = (image.format or path.suffix.lstrip(".") or "image").casefold()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        return InputFileProfile(
            **base,
            detected_format="image",
            media_type="application/octet-stream",
            inspection_status="failed",
            details={},
            limitations=[f"image metadata inspection failed ({type(exc).__name__})"],
        )
    return InputFileProfile(
        **base,
        detected_format=detected,
        media_type=media_type,
        inspection_status="complete",
        details=details,
        limitations=[],
    )


def _profile_archive(
    path: Path, base: dict[str, Any], detected_format: str
) -> InputFileProfile:
    try:
        with zipfile.ZipFile(path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            details = {
                "member_count": len(members),
                "members": [
                    {"path": item.filename, "bytes": item.file_size}
                    for item in members[:MAX_ARCHIVE_MEMBERS]
                ],
                "archive_extracted": False,
            }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return InputFileProfile(
            **base,
            detected_format=detected_format,
            media_type="application/zip",
            inspection_status="failed",
            details={},
            limitations=[f"archive inventory failed ({type(exc).__name__})"],
        )
    limitations = []
    if len(members) > MAX_ARCHIVE_MEMBERS:
        limitations.append(
            f"only the first {MAX_ARCHIVE_MEMBERS} of {len(members)} archive members are listed"
        )
    if detected_format == "xlsx":
        limitations.append(
            "workbook sheets and cell missingness require a locked sandbox analysis step"
        )
    return InputFileProfile(
        **base,
        detected_format=detected_format,
        media_type={
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }.get(detected_format, "application/zip"),
        inspection_status="complete"
        if len(members) <= MAX_ARCHIVE_MEMBERS
        else "partial",
        details=details,
        limitations=limitations,
    )


def _validated_archive_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = [item for item in archive.infolist() if not item.is_dir()]
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError("archive has too many members for structural intake")
    total = 0
    for item in members:
        if item.file_size > MAX_ARCHIVE_MEMBER_BYTES:
            raise ValueError("archive member exceeds the intake size limit")
        total += item.file_size
        if total > MAX_ARCHIVE_TOTAL_BYTES:
            raise ValueError("archive exceeds the intake uncompressed-size limit")
        if (
            item.compress_size
            and item.file_size / item.compress_size > MAX_COMPRESSION_RATIO
        ):
            raise ValueError(
                "archive member exceeds the intake compression-ratio limit"
            )
    return members


def _xlsx_column_index(reference: str) -> int | None:
    match = re.match(r"^([A-Z]+)", reference.upper())
    if not match:
        return None
    value = 0
    for character in match.group(1):
        value = value * 26 + ord(character) - 64
    return value - 1


def _xlsx_cell_value(cell: ElementTree.Element, shared: list[str]) -> Any:
    kind = cell.attrib.get("t", "")
    if kind == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//{*}t"))
    node = cell.find("{*}v")
    if node is None or node.text is None:
        return None
    value = node.text
    if kind == "s":
        try:
            return shared[int(value)]
        except (ValueError, IndexError):
            return None
    if kind == "b":
        return value == "1"
    if kind in {"str", "e"}:
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def _profile_xlsx(path: Path, base: dict[str, Any]) -> InputFileProfile:
    try:
        with zipfile.ZipFile(path) as archive:
            members = _validated_archive_members(archive)
            names = {item.filename for item in members}
            shared: list[str] = []
            if "xl/sharedStrings.xml" in names:
                root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
                shared = [
                    "".join(node.text or "" for node in item.findall(".//{*}t"))
                    for item in root.findall(".//{*}si")
                ]
            sheets = sorted(
                name
                for name in names
                if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
            )
            if not sheets:
                raise ValueError("workbook has no worksheets")
            root = ElementTree.fromstring(archive.read(sheets[0]))
    except (
        OSError,
        ValueError,
        zipfile.BadZipFile,
        ElementTree.ParseError,
    ) as exc:
        return InputFileProfile(
            **base,
            detected_format="xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            inspection_status="failed",
            details={},
            limitations=[
                f"workbook structural inspection failed ({type(exc).__name__})"
            ],
        )

    parsed_rows: list[dict[int, Any]] = []
    capped = False
    for row in root.findall(".//{*}sheetData/{*}row"):
        if len(parsed_rows) > MAX_TABULAR_ROWS:
            capped = True
            break
        values: dict[int, Any] = {}
        for cell in row.findall("{*}c"):
            index = _xlsx_column_index(cell.attrib.get("r", ""))
            if index is not None and index < MAX_COLUMNS:
                values[index] = _xlsx_cell_value(cell, shared)
        parsed_rows.append(values)
    if not parsed_rows:
        return InputFileProfile(
            **base,
            detected_format="xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            inspection_status="complete",
            rows_observed=0,
            rows_total=0,
            details={"worksheets": len(sheets), "first_worksheet": sheets[0]},
            limitations=["the first worksheet is empty"],
        )
    width = min(
        max((max(row, default=-1) for row in parsed_rows), default=-1) + 1,
        MAX_COLUMNS,
    )
    headers, header_limitations = _deduplicate_headers(
        [parsed_rows[0].get(index) for index in range(width)], width
    )
    accumulators = [_ColumnAccumulator(name) for name in headers]
    for row in parsed_rows[1:]:
        for index, accumulator in enumerate(accumulators):
            accumulator.add(row.get(index))
    limitations = [
        *header_limitations,
        "the first nonempty worksheet row is treated as the header during intake",
    ]
    if len(sheets) > 1:
        limitations.append(
            "column types and missingness are profiled for the first worksheet; all worksheets remain available to the locked analysis"
        )
    if capped:
        limitations.append(
            f"row and missingness counts cover only the first {MAX_TABULAR_ROWS} data rows"
        )
    data_rows = max(0, len(parsed_rows) - 1)
    return InputFileProfile(
        **base,
        detected_format="xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        inspection_status="partial" if capped or len(sheets) > 1 else "complete",
        rows_observed=data_rows,
        rows_total=None if capped else data_rows,
        columns=[item.finish() for item in accumulators],
        details={
            "worksheets": len(sheets),
            "worksheet_files": sheets[:32],
            "first_worksheet": sheets[0],
            "declared_columns": width,
            "values_included_in_profile": False,
        },
        limitations=limitations,
    )


def _profile_text(path: Path, base: dict[str, Any]) -> InputFileProfile:
    if path.stat().st_size > MAX_TEXT_BYTES:
        return InputFileProfile(
            **base,
            detected_format="text",
            media_type=mimetypes.guess_type(path.name)[0] or "text/plain",
            inspection_status="partial",
            details={"values_included_in_profile": False},
            limitations=[f"text exceeds the {MAX_TEXT_BYTES} byte inspection limit"],
        )
    encoding = _encoding(path)
    text = path.read_text(encoding=encoding, errors="replace")
    return InputFileProfile(
        **base,
        detected_format=path.suffix.lstrip(".").casefold() or "text",
        media_type=mimetypes.guess_type(path.name)[0] or "text/plain",
        inspection_status="complete",
        details={
            "encoding": encoding,
            "lines": len(text.splitlines()),
            "nonempty_lines": sum(bool(line.strip()) for line in text.splitlines()),
            "words": len(text.split()),
            "values_included_in_profile": False,
        },
        limitations=[],
    )


def _inspect_file(path: Path, known_sha256: str | None = None) -> InputFileProfile:
    suffix = path.suffix.casefold()
    base = {
        "path": f"/workspace/{path.name}",
        "sha256": known_sha256 or sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if suffix in {".csv", ".tsv", ".tab"}:
        return _profile_delimited(path, base)
    if suffix in {".json", ".geojson"}:
        return _profile_json(path, base)
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}:
        return _profile_image(path, base)
    if suffix == ".xlsx":
        return _profile_xlsx(path, base)
    if suffix in {".zip", ".docx", ".pptx"}:
        return _profile_archive(path, base, suffix.lstrip("."))
    if suffix in {".txt", ".md", ".rst", ".yaml", ".yml", ".r", ".py", ".sql"}:
        return _profile_text(path, base)
    if suffix == ".pdf":
        return InputFileProfile(
            **base,
            detected_format="pdf",
            media_type="application/pdf",
            inspection_status="partial",
            details={"pixel_interpretation": "Gemma-only preplanning review"},
            limitations=[
                "text extraction and complete page coverage are not guaranteed during intake"
            ],
        )
    return InputFileProfile(
        **base,
        detected_format=suffix.lstrip(".") or "unknown",
        media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        inspection_status="unsupported",
        details={},
        limitations=["format requires a task-specific reader in the locked sandbox"],
    )


def build_input_profile(
    workspace: Path, known_hashes: dict[str, str] | None = None
) -> InputProfile:
    """Inspect top-level immutable inputs with explicit bounded-coverage limits."""

    root = workspace.resolve()
    candidates = [
        path
        for path in sorted(root.iterdir(), key=lambda item: item.name.casefold())
        if path.is_file() and not path.is_symlink()
    ]
    files: list[InputFileProfile] = []
    limitations: list[str] = []
    for path in candidates[:MAX_PROFILE_FILES]:
        try:
            files.append(_inspect_file(path, (known_hashes or {}).get(path.name)))
        except (OSError, csv.Error) as exc:
            files.append(
                InputFileProfile(
                    path=f"/workspace/{path.name}",
                    sha256=(known_hashes or {}).get(path.name) or sha256_file(path),
                    bytes=path.stat().st_size,
                    detected_format=path.suffix.lstrip(".") or "unknown",
                    media_type=mimetypes.guess_type(path.name)[0]
                    or "application/octet-stream",
                    inspection_status="failed",
                    details={},
                    limitations=[
                        f"structural inspection failed ({type(exc).__name__})"
                    ],
                )
            )
    if len(candidates) > MAX_PROFILE_FILES:
        limitations.append(
            f"only the first {MAX_PROFILE_FILES} of {len(candidates)} inputs were structurally profiled"
        )
    limitations.append(
        "intake profiles contain no cell values or effect estimates; semantic coding, units, sentinel missing values, and design metadata still require validated analysis"
    )
    return InputProfile(
        total_files=len(candidates),
        profiled_files=len(files),
        files=files,
        limitations=limitations,
    )
