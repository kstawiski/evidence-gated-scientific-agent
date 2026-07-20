"""Deterministic XLSX export for reader-facing Evidence Bench results."""

from __future__ import annotations

import csv
import io
import math
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from .schemas import ScientificReport


_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_INVALID_SHEET_CHARS = re.compile(r"[\\/*?:\[\]]")
_INVALID_XML_CHARACTERS = re.compile("[^\x09\x0a\x0d\x20-\ud7ff\ue000-\ufffd]")


def _column_name(index: int) -> str:
    value = index
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _cell(reference: str, value: Any, *, style: int = 0) -> str:
    text = _INVALID_XML_CHARACTERS.sub("�", "" if value is None else str(value))
    style_attr = f' s="{style}"' if style else ""
    stripped = text.strip()
    unsigned = stripped.lstrip("+-")
    has_leading_zero_identifier = (
        len(unsigned) > 1 and unsigned.startswith("0") and unsigned[1].isdigit()
    )
    if _NUMBER.fullmatch(stripped) and not has_leading_zero_identifier:
        number = float(stripped)
        if math.isfinite(number):
            return f'<c r="{reference}"{style_attr}><v>{escape(stripped)}</v></c>'
    preserve = ' xml:space="preserve"' if text != text.strip() else ""
    return (
        f'<c r="{reference}" t="inlineStr"{style_attr}><is><t{preserve}>'
        f"{escape(text)}</t></is></c>"
    )


def _worksheet(rows: list[list[Any]], *, freeze_header: bool = True) -> str:
    width = max((len(row) for row in rows), default=1)
    height = max(len(rows), 1)
    rendered_rows: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells = "".join(
            _cell(
                f"{_column_name(column_index)}{row_index}",
                value,
                style=1 if row_index == 1 else 0,
            )
            for column_index, value in enumerate(row, start=1)
        )
        rendered_rows.append(f'<row r="{row_index}">{cells}</row>')
    view = (
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        "</sheetView></sheetViews>"
        if freeze_header and rows
        else '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
    )
    auto_filter = (
        f'<autoFilter ref="A1:{_column_name(width)}{height}"/>'
        if rows and width > 1
        else ""
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{_column_name(width)}{height}"/>{view}'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f"<sheetData>{''.join(rendered_rows)}</sheetData>{auto_filter}</worksheet>"
    )


def _sheet_name(value: str, used: set[str]) -> str:
    safe_value = _INVALID_XML_CHARACTERS.sub("�", value)
    base = _INVALID_SHEET_CHARS.sub(" ", safe_value).strip(" '") or "Results"
    base = base[:31]
    candidate = base
    counter = 2
    while candidate.casefold() in used:
        suffix = f" {counter}"
        candidate = f"{base[: 31 - len(suffix)]}{suffix}"
        counter += 1
    used.add(candidate.casefold())
    return candidate


def _table_rows(path: Path) -> list[list[str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            list(row) for row in csv.reader(handle, delimiter=delimiter, strict=True)
        ]


def build_results_workbook(
    report: ScientificReport,
    table_sources: Iterable[tuple[dict[str, Any], Path]],
    *,
    run_id: str,
    quality_status: str,
) -> bytes:
    """Return a dependency-free XLSX containing report text, claims, and full tables."""

    sheets: list[tuple[str, list[list[Any]]]] = []
    sheets.append(
        (
            "README",
            [
                ["Field", "Value"],
                ["Product", "Evidence Bench"],
                ["Run ID", run_id],
                ["Report title", report.title],
                ["Quality status", quality_status],
                [
                    "Interpretation",
                    (
                        "Validated exploratory result"
                        if quality_status in {"supported", "supported_with_comments"}
                        else "Provisional result; review unresolved findings before scientific use"
                    ),
                ],
                [
                    "Workbook contents",
                    "Report sections, claim ledger, and full registered result tables",
                ],
            ],
        )
    )
    sheets.append(
        (
            "Report",
            [
                ["Section", "Text"],
                ["Executive summary", report.executive_summary],
                ["Results", report.results or report.narrative],
                ["Discussion", report.discussion],
                ["Conclusions", report.conclusions],
                ["Limitations", "\n".join(report.limitations)],
                ["Unresolved issues", "\n".join(report.unresolved_issues)],
            ],
        )
    )
    sheets.append(
        (
            "Claims",
            [
                [
                    "Claim ID",
                    "Type",
                    "Status",
                    "Claim",
                    "Evidence references",
                    "Limitations",
                ],
                *[
                    [
                        claim.claim_id,
                        claim.claim_type,
                        claim.status,
                        claim.text,
                        "; ".join(claim.evidence_refs),
                        "\n".join(claim.limitations),
                    ]
                    for claim in report.claims
                ],
            ],
        )
    )
    for entry, source in table_sources:
        sheets.append(
            (
                str(entry.get("title") or entry.get("display_id") or "Table"),
                _table_rows(source),
            )
        )

    used: set[str] = set()
    named_sheets = [(_sheet_name(name, used), rows) for name, rows in sheets]
    workbook_sheets = "".join(
        f'<sheet name={quoteattr(name)} sheetId="{index}" r:id="rId{index}"/>'
        for index, (name, _) in enumerate(named_sheets, start=1)
    )
    relationships = "".join(
        "<Relationship "
        f'Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(named_sheets) + 1)
    )
    relationships += (
        '<Relationship Id="rIdStyles" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    content_overrides = "".join(
        "<Override "
        f'PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(named_sheets) + 1)
    )

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            f"{content_overrides}</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{workbook_sheets}</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{relationships}</Relationships>",
        )
        archive.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="2"><font><sz val="11"/><name val="Aptos"/></font><font><b/><sz val="11"/><name val="Aptos"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            "</styleSheet>",
        )
        for index, (_, rows) in enumerate(named_sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet(rows))
    return output.getvalue()
