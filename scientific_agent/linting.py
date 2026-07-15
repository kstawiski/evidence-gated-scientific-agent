"""Deterministic plan and claim-evidence checks."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .literature import LiteratureError, load_acquired_article_record
from .reporting import (
    MIN_REPORTED_FIGURE_DPI,
    caption_has_number_prefix,
    extract_figure_ocr,
    excessive_table_precision,
    inspect_figure,
    logical_report_output_key,
    read_table_preview,
    resolve_display_artifact,
)
from .schemas import (
    ArtifactRef,
    DeterministicValidation,
    ComputationEvidence,
    LintFinding,
    PlanLintReport,
    PlanProposal,
    RetrievalEvidence,
    ScientificReport,
    TaskSpec,
)


_WORD = re.compile(r"[a-z0-9]{4,}")
_PROTOCOL_TIMING = re.compile(
    r"\b(?:lock(?:ed|ing)?|prespecif(?:ied|ication))\b.{0,100}"
    r"\b(?:before|prior to)\b.{0,100}\b(?:inspect(?:ion|ing)?|outcome|result)",
    re.IGNORECASE,
)
_METHODOLOGICAL_GENERALIZATION = re.compile(
    r"\b(?:robust to|valid despite|known to|generally reliable|assumption violation)",
    re.IGNORECASE,
)
_METHODOLOGICAL_RECOMMENDATION = re.compile(
    r"\b(?:default (?:choice|strategy|method|procedure)|"
    r"(?:recommend(?:ed|ing)?|recommendation).{0,80}(?:as (?:the )?default|"
    r"prefer|use|adopt|replac(?:e|ed|ing))|"
    r"should be (?:prioritized|preferred|used)|"
    r"should be replaced(?: by| with)?|"
    r"(?:method|test|procedure).{0,60}\bpreferable\b|"
    r"(?:method|test|procedure).{0,40}\bsuperior\b|"
    r"abandon\b.{0,80}\bin favou?r of)\b",
    re.IGNORECASE,
)
_METHOD_RECOMMENDATION_SCOPE = re.compile(
    r"\b(?:within|in|under|for|across|among)\b.{0,120}\b(?:simulat(?:ed|ion|ions)|"
    r"evaluated|studied|conditions?|sample sizes?|distributions?|variance ratios?|"
    r"regimes?|comparisons?|analyses?|datasets?|cohorts?|tasks?)\b",
    re.IGNORECASE,
)
_UNIVERSAL_METHOD_RECOMMENDATION = re.compile(
    r"\b(?:(?:for|across|under|in|regardless of)\s+(?:all|any|every)\b|"
    r"universally\b|always\b|irrespective of\b)",
    re.IGNORECASE,
)
_BOUNDED_UNIVERSAL_METHOD_SCOPE = re.compile(
    r"\b(?:(?:all|any|every)\s+(?:evaluated|studied|simulated|prespecified)\b|"
    r"(?:evaluated|studied|simulated|prespecified)\b.{0,80}\b(?:conditions?|"
    r"sample sizes?|distributions?|variance ratios?|regimes?|comparisons?|"
    r"analyses?|datasets?|cohorts?|tasks?))\b",
    re.IGNORECASE,
)
_PROCEDURE_EQUIVALENCE = re.compile(
    r"(?:\b(?:tests?|methods?|procedures?|results?|p-?values?)\b.{0,80}"
    r"\b(?:equivalent|identical|the same)\b|\b(?:equivalent|identical|the same)\b"
    r".{0,80}\b(?:tests?|methods?|procedures?|results?|p-?values?)\b)",
    re.IGNORECASE,
)
_GROUNDING_STOPWORDS = {
    "abstract",
    "analysis",
    "article",
    "authors",
    "conclusion",
    "evidence",
    "finding",
    "findings",
    "methods",
    "paper",
    "reported",
    "reports",
    "result",
    "results",
    "study",
}
_NUMERIC_CELL = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_PRECISE_LITERATURE_NUMBER = re.compile(
    r"(?<![\w.])(?:0\.\d+|\d+(?:\.\d+)?\s*%)(?![\w.])"
)
_TABLE_KEY_COLUMNS = {"measure", "metric", "outcome", "parameter", "statistic"}
_TABLE_VALUE_COLUMNS = {"estimate", "result", "value"}
_GENERIC_NUMERIC_COLUMNS = _TABLE_VALUE_COLUMNS | {"statistic"}
_EQUATION_OPERANDS = re.compile(
    r"(?P<left>[^\s,;:=()]+)\s*(?<![<>])=(?!=)\s*"
    r"(?P<right>[^\s,;:=().]+)",
    re.UNICODE,
)


def _is_report_output(path: Path, directory: str) -> bool:
    parts = [part.lower() for part in path.parts]
    return any(
        parts[index : index + 2] == ["output", directory]
        for index in range(max(0, len(parts) - 1))
    )


def _terms(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _grounding_terms(text: str) -> set[str]:
    return _terms(text) - _GROUNDING_STOPWORDS


def _equation_operand(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w]+", "", normalized, flags=re.UNICODE)


def _self_equality_examples(text: str, *, limit: int = 3) -> list[str]:
    """Return nontrivial equations whose left and right operands are identical."""

    examples: list[str] = []
    for match in _EQUATION_OPERANDS.finditer(text):
        left = _equation_operand(match.group("left"))
        right = _equation_operand(match.group("right"))
        if (
            left == right
            and len(left) >= 2
            and any(character.isalpha() for character in left)
        ):
            examples.append(match.group(0))
            if len(examples) >= limit:
                break
    return examples


def _metadata_value_matches(field: str, reported: Any, recorded: Any) -> bool:
    if field in {"local_pdf_path", "local_markdown_path"}:
        return (None if reported is None else os.path.normpath(str(reported))) == (
            None if recorded is None else os.path.normpath(str(recorded))
        )
    if field == "url":
        return _normalize_url(str(reported or "")) == _normalize_url(
            str(recorded or "")
        )
    if field in {"doi", "pmcid"}:
        return str(reported or "").lower() == str(recorded or "").lower()
    if field == "title":
        return " ".join(str(reported or "").split()) == " ".join(
            str(recorded or "").split()
        )
    return reported == recorded


def lint_plan(task: TaskSpec, plan: PlanProposal) -> PlanLintReport:
    findings: list[LintFinding] = []
    step_ids = [step.step_id for step in plan.steps]
    if len(step_ids) != len(set(step_ids)):
        findings.append(
            LintFinding(
                code="duplicate_step_id",
                location="steps",
                message="Plan step IDs must be unique.",
            )
        )

    for index, step in enumerate(plan.steps):
        location = f"steps[{index}]"
        if not step.validators:
            findings.append(
                LintFinding(
                    code="missing_validator",
                    location=location,
                    message="Every step must declare at least one validator.",
                )
            )
        if not step.stop_conditions:
            findings.append(
                LintFinding(
                    code="missing_stop_condition",
                    location=location,
                    message="Every step must declare a stopping condition.",
                )
            )
        if step.security_risk == "irreversible":
            findings.append(
                LintFinding(
                    code="irreversible_action",
                    location=location,
                    message="The agent cannot execute irreversible actions.",
                )
            )

    produced_text = " ".join(
        [
            *plan.expected_artifacts,
            *(output for step in plan.steps for output in step.outputs),
        ]
    )
    produced_terms = _terms(produced_text)
    for index, deliverable in enumerate(task.deliverables):
        wanted = _terms(deliverable)
        if wanted and not (wanted & produced_terms):
            findings.append(
                LintFinding(
                    code="unmapped_deliverable",
                    location=f"task.deliverables[{index}]",
                    message=f"No declared output appears to produce: {deliverable}",
                )
            )

    if task.scientific_risk in {"confirmatory", "decision_critical"}:
        combined = " ".join(
            [
                *plan.assumptions,
                *plan.expected_artifacts,
                *(method for step in plan.steps for method in step.methods),
            ]
        ).lower()
        if not any(
            term in combined for term in ("protocol", "preregister", "method lock")
        ):
            findings.append(
                LintFinding(
                    code="missing_method_lock",
                    location="plan",
                    message="Confirmatory work requires a protocol or method lock before results.",
                )
            )

    return PlanLintReport(
        passed=not any(f.blocking for f in findings), findings=findings
    )


def _normalize_url(value: str) -> str:
    return value.rstrip("/")


def _reject_nonfinite_json(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _numeric_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _logical_json_output_key(path: Path) -> str:
    """Identify the same generated JSON output across bounded repair attempts."""

    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    for index in range(len(parts) - 1, -1, -1):
        if lowered[index] == "output" and index + 1 < len(parts):
            return "/".join(lowered[index + 1 :])
    return str(path.resolve())


def _collect_json_numbers(
    value: Any,
    destination: dict[str, list[Decimal]],
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = _numeric_key(str(raw_key))
            child_path = (*path, key) if key else path
            if isinstance(child, Decimal) and child.is_finite():
                for candidate_key in {key, "_".join(child_path)} - {""}:
                    destination.setdefault(candidate_key, []).append(child)
            elif not isinstance(child, bool):
                _collect_json_numbers(child, destination, child_path)
    elif isinstance(value, list):
        for child in value:
            _collect_json_numbers(child, destination, path)


def _machine_json_numbers(
    computation: ComputationEvidence | None,
) -> dict[str, list[Decimal]]:
    """Read numeric fields from the latest version of each generated JSON output."""

    latest: dict[str, Path] = {}
    for artifact in computation.artifacts if computation else []:
        path = Path(artifact.path)
        if (
            path.suffix.lower() != ".json"
            or _is_report_output(path, "figures")
            or _is_report_output(path, "tables")
        ):
            continue
        latest[_logical_json_output_key(path)] = path

    numbers: dict[str, list[Decimal]] = {}
    for path in latest.values():
        if not path.is_file() or path.is_symlink():
            continue
        try:
            value = json.loads(
                path.read_text(encoding="utf-8"),
                parse_float=Decimal,
                parse_int=Decimal,
                parse_constant=_reject_nonfinite_json,
            )
        except (OSError, UnicodeError, ValueError):
            continue
        _collect_json_numbers(value, numbers)
    return numbers


def _numeric_cell_agrees(cell: str, source: Decimal) -> bool:
    """Accept a cell only when it is a valid up-to-four-significant-digit rounding."""

    token = cell.strip()
    if not _NUMERIC_CELL.fullmatch(token):
        return False
    try:
        displayed = Decimal(token)
    except InvalidOperation:
        return False
    if not displayed.is_finite():
        return False
    # A reported zero is never an acceptable rendering of non-zero evidence.
    # This is especially important for very small p-values.
    if displayed.is_zero() or source.is_zero():
        return displayed == source

    mantissa = token.lower().split("e", 1)[0].lstrip("+-").replace(".", "")
    significant = mantissa.lstrip("0")
    digits = min(len(significant) if significant else 1, 4)
    rounding_unit = Decimal(1).scaleb(source.copy_abs().adjusted() - digits + 1)
    return abs(displayed - source) <= rounding_unit / 2


def _table_json_numeric_mismatches(
    preview: dict[str, Any],
    machine_numbers: dict[str, list[Decimal]],
    *,
    example_limit: int = 5,
) -> list[str]:
    """Compare only explicitly key-matched table cells with computational JSON."""

    columns = [str(column) for column in preview.get("columns", [])]
    normalized_columns = [_numeric_key(column) for column in columns]
    rows = preview.get("rows", [])
    examples: list[str] = []

    key_indexes = [
        index
        for index, column in enumerate(normalized_columns)
        if column in _TABLE_KEY_COLUMNS
    ]
    value_indexes = [
        index
        for index, column in enumerate(normalized_columns)
        if column in _TABLE_VALUE_COLUMNS
    ]
    if len(key_indexes) == 1 and len(value_indexes) == 1:
        key_index, value_index = key_indexes[0], value_indexes[0]
        if key_index != value_index:
            for row_number, row in enumerate(rows, start=1):
                if max(key_index, value_index) >= len(row):
                    continue
                key = _numeric_key(str(row[key_index]))
                cell = str(row[value_index]).strip()
                candidates = machine_numbers.get(key, [])
                if (
                    candidates
                    and _NUMERIC_CELL.fullmatch(cell)
                    and not any(
                        _numeric_cell_agrees(cell, value) for value in candidates
                    )
                ):
                    values = ", ".join(str(value) for value in candidates[:3])
                    examples.append(
                        f"row {row_number}, {row[key_index]}={cell}; JSON: {values}"
                    )
                    if len(examples) >= example_limit:
                        return examples
            return examples

    for column_index, key in enumerate(normalized_columns):
        if key in _GENERIC_NUMERIC_COLUMNS:
            continue
        candidates = machine_numbers.get(key, [])
        if not candidates:
            continue
        for row_number, row in enumerate(rows, start=1):
            if column_index >= len(row):
                continue
            cell = str(row[column_index]).strip()
            if _NUMERIC_CELL.fullmatch(cell) and not any(
                _numeric_cell_agrees(cell, value) for value in candidates
            ):
                values = ", ".join(str(value) for value in candidates[:3])
                examples.append(
                    f"row {row_number}, column {columns[column_index]}={cell}; "
                    f"JSON: {values}"
                )
                if len(examples) >= example_limit:
                    return examples
    return examples


def _figure_ocr_semantic_findings(
    ocr: dict[str, Any], machine_numbers: dict[str, list[Decimal]]
) -> list[tuple[str, str]]:
    """Detect high-confidence semantic display defects from rendered labels."""

    if not ocr.get("available"):
        return []
    text = " ".join(str(ocr.get("text", "")).casefold().split())
    findings: list[tuple[str, str]] = []
    p_values = [
        value
        for key, values in machine_numbers.items()
        if key == "p" or key.endswith("p_value") or key.endswith("pvalue")
        for value in values
        if value.is_finite()
    ]
    if any(not value.is_zero() for value in p_values) and re.search(
        r"\bp\s*=\s*0(?:[.,]0+)?\b", text
    ):
        findings.append(
            (
                "figure_zero_rounded_nonzero_p_value",
                "The rendered figure labels a nonzero computed p-value as zero; "
                "use scientific notation or an honest inequality.",
            )
        )
    if (
        re.search(r"\bmean\s+(?:difference|diff)\b", text)
        and re.search(r"\b(?:hedges|cohen)\s+[gd]\b", text)
        and re.search(r"\beffect\s+size\b", text)
    ):
        findings.append(
            (
                "figure_mixed_incompatible_effect_scales",
                "The rendered figure presents an unstandardized mean difference "
                "and a standardized effect under one generic effect-size axis; "
                "use separate, explicitly scaled panels or omit the secondary effect.",
            )
        )
    return findings


def validate_report(
    report: ScientificReport,
    retrieval: RetrievalEvidence | None = None,
    computation: ComputationEvidence | None = None,
    required_languages: tuple[str, ...] = (),
    require_reconciliation: bool = False,
    require_pubmed_literature: bool = False,
    controller_artifacts: tuple[ArtifactRef, ...] = (),
    controller_dates: tuple[str, ...] = (),
) -> DeterministicValidation:
    findings: list[LintFinding] = []
    if require_pubmed_literature:
        retrieval_tools = set(retrieval.tools if retrieval else ())
        if "search_pubmed" not in retrieval_tools:
            findings.append(
                LintFinding(
                    code="pubmed_search_missing",
                    location="retrieval_evidence",
                    message=(
                        "Biomedical and health-science analyses must run a typed "
                        "PubMed search and record its query artifact."
                    ),
                )
            )
        if "acquire_pubmed_article" not in retrieval_tools:
            findings.append(
                LintFinding(
                    code="pubmed_article_not_acquired",
                    location="retrieval_evidence",
                    message=(
                        "Acquire at least one relevant PubMed record into the "
                        "workspace, including verified local Markdown and an explicit "
                        "PDF availability status."
                    ),
                )
            )
        if not any(
            source.pmid and source.local_markdown_path for source in report.sources
        ):
            findings.append(
                LintFinding(
                    code="pubmed_source_not_cited",
                    location="sources",
                    message=(
                        "The article must cite at least one locally acquired PubMed "
                        "record when PubMed support is required."
                    ),
                )
            )
    records = computation.records if computation else []
    for record in records:
        if record.status != "succeeded":
            continue
        if any(
            _is_report_output(Path(artifact.path), "figures")
            for artifact in record.artifacts
        ):
            try:
                render_stderr = Path(record.stderr_path).read_text(
                    encoding="utf-8", errors="replace"
                )[:64_000]
            except OSError:
                render_stderr = ""
            if "results might be incorrect" in render_stderr.lower():
                findings.append(
                    LintFinding(
                        code="figure_render_warning",
                        location=record.stderr_path,
                        message=(
                            "The plotting engine reported that layout results might "
                            "be incorrect; regenerate and inspect the figure."
                        ),
                    )
                )
        for artifact in record.artifacts:
            path = Path(artifact.path)
            if (
                artifact.description != "sandbox-generated analysis artifact"
                or path.suffix.lower() != ".json"
            ):
                continue
            try:
                json.loads(
                    path.read_text(encoding="utf-8"),
                    parse_constant=_reject_nonfinite_json,
                )
            except (OSError, UnicodeError, ValueError) as exc:
                findings.append(
                    LintFinding(
                        code="invalid_generated_json",
                        location=str(path),
                        message=(
                            "Generated JSON must be strict UTF-8 JSON and may not "
                            f"contain NaN or Infinity: {type(exc).__name__}."
                        ),
                    )
                )
    for language in required_languages:
        successful_outputs = [
            artifact
            for record in records
            if record.language == language and record.status == "succeeded"
            for artifact in record.artifacts
            if artifact.description == "sandbox-generated analysis artifact"
        ]
        if not successful_outputs:
            findings.append(
                LintFinding(
                    code="required_computation_language_missing",
                    location="computation_evidence",
                    message=(
                        f"The locked task requires {language}, but no successful "
                        "execution from that language produced an analysis artifact."
                    ),
                )
            )
    if require_reconciliation:
        candidates = [
            artifact
            for record in records
            if record.status == "succeeded"
            for artifact in record.artifacts
            if artifact.description == "sandbox-generated analysis artifact"
            and any(
                marker in Path(artifact.path).name.lower()
                for marker in ("reconciliation", "crosscheck", "cross-check")
            )
        ]
        if not candidates:
            findings.append(
                LintFinding(
                    code="required_reconciliation_artifact_missing",
                    location="computation_evidence",
                    message=(
                        "The locked cross-language task requires a generated "
                        "machine-readable reconciliation artifact."
                    ),
                )
            )
        else:
            verdicts: list[bool] = []
            for artifact in candidates:
                path = Path(artifact.path)
                if path.suffix.lower() != ".json" or not path.is_file():
                    continue
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not isinstance(value, dict):
                    continue
                for key in (
                    "all_pass",
                    "passed",
                    "within_tolerance",
                    "reconciliation_passed",
                ):
                    verdict = value.get(key)
                    if isinstance(verdict, bool):
                        verdicts.append(verdict)
                        break
            if not verdicts:
                findings.append(
                    LintFinding(
                        code="reconciliation_artifact_invalid",
                        location="computation_evidence",
                        message=(
                            "The reconciliation artifact must be JSON with a top-level "
                            "boolean all_pass, passed, within_tolerance, or "
                            "reconciliation_passed verdict."
                        ),
                    )
                )
            elif not verdicts[-1]:
                findings.append(
                    LintFinding(
                        code="cross_language_reconciliation_failed",
                        location="computation_evidence",
                        message=(
                            "The latest generated reconciliation artifact reports "
                            "that the prespecified tolerance was not met."
                        ),
                    )
                )
            elif not all(verdicts[:-1]):
                findings.append(
                    LintFinding(
                        code="superseded_reconciliation_failure",
                        location="computation_evidence",
                        message=(
                            "An earlier reconciliation failed before a later "
                            "successful correction; both remain in provenance."
                        ),
                        blocking=False,
                    )
                )
    source_ids = [source.source_id for source in report.sources]
    if len(source_ids) != len(set(source_ids)):
        findings.append(
            LintFinding(
                code="duplicate_source_id",
                location="sources",
                message="Source IDs must be unique.",
            )
        )
    known = set(source_ids)
    sources_by_id = {source.source_id: source for source in report.sources}
    controller_paths = {
        os.path.normpath(artifact.path) for artifact in controller_artifacts
    }
    methods_text = " ".join(report.methods)
    if _PROTOCOL_TIMING.search(methods_text) and not any(
        source.artifact_path is not None
        and os.path.normpath(source.artifact_path) in controller_paths
        for source in report.sources
    ):
        findings.append(
            LintFinding(
                code="protocol_timing_without_controller_artifact",
                location="methods",
                message=(
                    "A claim that the protocol was locked before outcome inspection "
                    "must cite the controller-generated protocol artifact; a later "
                    "sandbox analysis plan cannot establish timing."
                ),
            )
        )
    if (
        "study design is unspecified" in report.introduction.lower()
        and "following an intervention" in report.introduction.lower()
    ):
        findings.append(
            LintFinding(
                code="unspecified_design_intervention_framing",
                location="introduction",
                message=(
                    "An unspecified design cannot be framed as following an "
                    "intervention; describe the measurement interval without "
                    "implying exposure assignment."
                ),
            )
        )
    discussion_and_limits = " ".join((report.discussion, *report.limitations)).lower()
    if (
        re.search(
            r"\bwithout\b.{0,80}\b(?:domain|units?|scale)\b", discussion_and_limits
        )
        and "clinical" in discussion_and_limits
    ):
        findings.append(
            LintFinding(
                code="unknown_domain_clinical_framing",
                location="discussion",
                message=(
                    "When the outcome domain or units are unknown, report unknown "
                    "scientific or practical relevance without introducing clinical "
                    "importance."
                ),
            )
        )
    retrieval_artifacts = {
        os.path.normpath(path) for path in (retrieval.artifacts if retrieval else [])
    }
    acquired_text_by_source: dict[str, str] = {}
    for index, source in enumerate(report.sources):
        if source.doi is not None and source.source_type == "web_page":
            findings.append(
                LintFinding(
                    code="doi_source_misclassified_as_web_page",
                    location=f"sources[{index}].source_type",
                    message=(
                        "A DOI-bearing scholarly source cannot be downgraded to a "
                        "generic web page to bypass literature-acquisition checks; "
                        "classify the article honestly and acquire its local record "
                        "or remove unsupported dependent claims."
                    ),
                )
            )
        if (
            source.url is not None
            and source.source_type in {"primary_study", "review"}
            and source.local_markdown_path is None
        ):
            findings.append(
                LintFinding(
                    code="literature_source_not_locally_acquired",
                    location=f"sources[{index}]",
                    message=(
                        "A cited primary study or review must have a "
                        "controller-verified local Markdown record so users can "
                        "preview the exact evidence; a search result or browser "
                        "page alone is only a lead."
                    ),
                )
            )
        if source.pmid and source.local_markdown_path is None:
            findings.append(
                LintFinding(
                    code="pubmed_source_without_local_text",
                    location=f"sources[{index}]",
                    message=(
                        "A cited PubMed record requires a locally acquired Markdown "
                        "or explicit abstract record for browser preview."
                    ),
                )
            )
        for field_name, value, suffix in (
            ("local_pdf_path", source.local_pdf_path, ".pdf"),
            ("local_markdown_path", source.local_markdown_path, ".md"),
        ):
            if value is None:
                continue
            normalized = os.path.normpath(value)
            if normalized not in retrieval_artifacts:
                findings.append(
                    LintFinding(
                        code="local_literature_artifact_not_retrieved",
                        location=f"sources[{index}].{field_name}",
                        message=(
                            "Local literature files must be returned by a successful "
                            f"typed acquisition tool: {value}"
                        ),
                    )
                )
                continue
            path = Path(value)
            if path.suffix.lower() != suffix or not path.is_file() or path.is_symlink():
                findings.append(
                    LintFinding(
                        code="invalid_local_literature_artifact",
                        location=f"sources[{index}].{field_name}",
                        message=f"Local literature artifact has an invalid type: {value}",
                    )
                )
                continue
            if suffix == ".pdf":
                try:
                    with path.open("rb") as handle:
                        signature = handle.read(1024)
                    if path.stat().st_size < 10_000 or b"%PDF-" not in signature:
                        raise ValueError
                except (OSError, ValueError):
                    findings.append(
                        LintFinding(
                            code="invalid_local_literature_pdf",
                            location=f"sources[{index}].{field_name}",
                            message="Local literature PDF failed signature or size checks.",
                        )
                    )
        if source.pmid and source.full_text_status is None:
            findings.append(
                LintFinding(
                    code="pubmed_acquisition_status_missing",
                    location=f"sources[{index}].full_text_status",
                    message="A cited PubMed record requires an explicit acquisition status.",
                )
            )
        if source.pmid:
            try:
                acquired, _, markdown = load_acquired_article_record(source, retrieval)
                acquired_text_by_source[source.source_id] = markdown.lower()
                article = acquired["article"]
                acquisition = acquired["acquisition"]
                recorded_values = {
                    "pmid": article.get("pmid"),
                    "pmcid": article.get("pmcid"),
                    "doi": article.get("doi"),
                    "title": article.get("title"),
                    "url": article.get("canonical_url"),
                    "citekey": acquisition.get("citekey"),
                    "license": acquisition.get("license"),
                    "rights_status": acquisition.get("rights_status"),
                    "terms_warning": acquisition.get("terms_warning"),
                    "retracted": acquisition.get("retracted"),
                    "full_text_status": acquisition.get("status"),
                    "local_pdf_path": acquisition.get("pdf_path"),
                    "local_markdown_path": acquisition.get("markdown_path"),
                }
                source_values = {
                    "pmid": source.pmid,
                    "pmcid": source.pmcid,
                    "doi": source.doi,
                    "title": source.title,
                    "url": source.url,
                    "citekey": source.citekey,
                    "license": source.license,
                    "rights_status": source.rights_status,
                    "terms_warning": source.terms_warning,
                    "retracted": source.retracted,
                    "full_text_status": source.full_text_status,
                    "local_pdf_path": source.local_pdf_path,
                    "local_markdown_path": source.local_markdown_path,
                }
                for field_name, recorded in recorded_values.items():
                    reported = source_values[field_name]
                    if not _metadata_value_matches(field_name, reported, recorded):
                        findings.append(
                            LintFinding(
                                code="pubmed_acquisition_metadata_mismatch",
                                location=f"sources[{index}].{field_name}",
                                message=(
                                    f"SourceRecord {field_name} does not match the "
                                    "controller-recorded PubMed acquisition metadata."
                                ),
                            )
                        )
                passage_terms = _grounding_terms(source.supporting_passage)
                if passage_terms and not any(
                    term in acquired_text_by_source[source.source_id]
                    for term in passage_terms
                ):
                    findings.append(
                        LintFinding(
                            code="pubmed_supporting_passage_not_grounded",
                            location=f"sources[{index}].supporting_passage",
                            message=(
                                "The stated supporting passage has no informative "
                                "term overlap with the acquired article text."
                            ),
                        )
                    )
            except (LiteratureError, KeyError, TypeError, OSError) as exc:
                findings.append(
                    LintFinding(
                        code="pubmed_acquisition_metadata_invalid",
                        location=f"sources[{index}]",
                        message=str(exc),
                    )
                )
    claim_ids = [claim.claim_id for claim in report.claims]
    if len(claim_ids) != len(set(claim_ids)):
        findings.append(
            LintFinding(
                code="duplicate_claim_id",
                location="claims",
                message="Claim IDs must be unique.",
            )
        )

    provenance_text = " ".join(
        [
            report.executive_summary,
            report.introduction,
            *report.methods,
            report.results,
            report.discussion,
            report.conclusions,
            report.narrative,
            *report.limitations,
        ]
    ).lower()
    self_equalities = _self_equality_examples(provenance_text)
    if self_equalities:
        findings.append(
            LintFinding(
                code="tautological_equation",
                location="report",
                message=(
                    "The article contains an equality with the same nontrivial "
                    "operand on both sides; correct the scientific notation or "
                    "remove the tautology: " + "; ".join(self_equalities)
                ),
            )
        )
    if _METHODOLOGICAL_RECOMMENDATION.search(provenance_text) and not any(
        _METHODOLOGICAL_RECOMMENDATION.search(claim.text) for claim in report.claims
    ):
        findings.append(
            LintFinding(
                code="methodological_recommendation_missing_claim",
                location="report",
                message=(
                    "A default, superiority, or method-selection recommendation "
                    "appears in the article without a matching ClaimRecord. Add a "
                    "scoped evidence-linked claim or remove the recommendation."
                ),
            )
        )
    if re.search(
        r"\b(?:hash(?:es|ing)?|manifest|provenance)\b.{0,100}"
        r"\b(?:deferred|unavailable|not generated|cannot be generated)\b",
        provenance_text,
    ):
        findings.append(
            LintFinding(
                code="false_provenance_deferral",
                location="report",
                message=(
                    "The controller always generates the provenance manifest; "
                    "the report must not claim hashing is deferred or unavailable."
                ),
            )
        )

    display_ids = [display.display_id for display in report.displays]
    if len(display_ids) != len(set(display_ids)):
        findings.append(
            LintFinding(
                code="duplicate_display_id",
                location="displays",
                message="Display IDs must be unique.",
            )
        )
    display_paths = {
        os.path.normpath(display.artifact_path) for display in report.displays
    }
    # Computation evidence is append-only. A repair may therefore generate a
    # corrected artifact at the same logical /output path in a later attempt.
    # Keep the earlier version in provenance, but require the report to register
    # only the latest version rather than making a bad display impossible to
    # supersede.
    final_outputs_by_key: dict[str, Path] = {}
    for artifact in computation.artifacts if computation else []:
        path = Path(artifact.path)
        key = logical_report_output_key(path)
        if key is not None:
            final_outputs_by_key[key] = path
    final_outputs = list(final_outputs_by_key.values())
    for path in final_outputs:
        if os.path.normpath(str(path)) not in display_paths:
            findings.append(
                LintFinding(
                    code="unregistered_report_artifact",
                    location=str(path),
                    message=(
                        "Every successful artifact below output/figures or "
                        "output/tables must be registered as a report display."
                    ),
                )
            )

    section_text = {
        "methods": " ".join(report.methods),
        "results": report.results,
        "discussion": report.discussion,
    }
    display_counters = {"figure": 0, "table": 0}
    mention_positions: dict[str, list[int]] = {"figure": [], "table": []}
    article_text = "\n".join(
        [" ".join(report.methods), report.results, report.discussion]
    )
    machine_numbers = _machine_json_numbers(computation)
    known_claim_ids = set(claim_ids)
    known_source_ids = set(source_ids)
    for index, display in enumerate(report.displays):
        location = f"displays[{index}]"
        display_counters[display.kind] += 1
        number = display_counters[display.kind]
        label = "Figure" if display.kind == "figure" else "Table"
        mention = re.search(
            rf"\b{label}\s+{number}\b",
            section_text[display.placement],
            re.IGNORECASE,
        )
        if mention is None:
            findings.append(
                LintFinding(
                    code="display_not_mentioned",
                    location=location,
                    message=(
                        f"{label} {number} must be mentioned in its "
                        f"{display.placement} section."
                    ),
                )
            )
        overall = re.search(rf"\b{label}\s+{number}\b", article_text, re.IGNORECASE)
        if overall is not None:
            mention_positions[display.kind].append(overall.start())
        if caption_has_number_prefix(display.caption) or caption_has_number_prefix(
            display.title
        ):
            findings.append(
                LintFinding(
                    code="model_supplied_display_number",
                    location=location,
                    message="The controller, not the model, assigns display numbers.",
                )
            )
        unknown_claims = sorted(set(display.claim_ids) - known_claim_ids)
        if unknown_claims:
            findings.append(
                LintFinding(
                    code="display_unknown_claim",
                    location=location,
                    message=f"Display references unknown claims: {', '.join(unknown_claims)}",
                )
            )
        unknown_sources = sorted(set(display.evidence_refs) - known_source_ids)
        if unknown_sources:
            findings.append(
                LintFinding(
                    code="display_unknown_evidence_ref",
                    location=location,
                    message=(
                        "Display references unknown evidence sources: "
                        + ", ".join(unknown_sources)
                    ),
                )
            )
        try:
            artifact_path = resolve_display_artifact(
                display, computation or ComputationEvidence()
            )
            if display.kind == "figure":
                if len(display.alt_text.strip()) < 10:
                    findings.append(
                        LintFinding(
                            code="figure_alt_text_missing",
                            location=f"{location}.alt_text",
                            message="Figures require meaningful alternative text.",
                        )
                    )
                figure_metadata = inspect_figure(artifact_path)
                reported_dpi = figure_metadata.get("dpi")
                if (
                    isinstance(reported_dpi, (int, float))
                    # PNG stores pixels/metre, so an exact 300-DPI save commonly
                    # round-trips through Pillow as 299.9994.
                    and reported_dpi + 0.5 < MIN_REPORTED_FIGURE_DPI
                ):
                    findings.append(
                        LintFinding(
                            code="figure_dpi_below_minimum",
                            location=location,
                            message=(
                                f"Figure reports {reported_dpi:.1f} DPI; final "
                                f"scientific rasters require at least "
                                f"{MIN_REPORTED_FIGURE_DPI:.0f} DPI."
                            ),
                        )
                    )
                figure_ocr = extract_figure_ocr(artifact_path)
                for code, message in _figure_ocr_semantic_findings(
                    figure_ocr, machine_numbers
                ):
                    findings.append(
                        LintFinding(code=code, location=location, message=message)
                    )
            else:
                table_preview = read_table_preview(artifact_path)
                precision_examples = excessive_table_precision(table_preview)
                if precision_examples:
                    findings.append(
                        LintFinding(
                            code="table_excessive_precision",
                            location=location,
                            message=(
                                "Reader-facing tables must use scientific display "
                                "precision; preserve full precision in JSON. Examples: "
                                + "; ".join(precision_examples)
                            ),
                        )
                    )
                contradiction_examples = _table_json_numeric_mismatches(
                    table_preview, machine_numbers
                )
                if contradiction_examples:
                    findings.append(
                        LintFinding(
                            code="table_machine_result_contradiction",
                            location=location,
                            message=(
                                "Reader-facing numeric cells contradict exact-key "
                                "machine-readable JSON results. Preserve display "
                                "rounding but do not change the result: "
                                + "; ".join(contradiction_examples)
                            ),
                        )
                    )
        except ValueError as exc:
            findings.append(
                LintFinding(
                    code="invalid_display_artifact",
                    location=location,
                    message=str(exc),
                )
            )

    for kind, positions in mention_positions.items():
        if positions != sorted(positions):
            findings.append(
                LintFinding(
                    code="display_mentions_out_of_order",
                    location="report",
                    message=f"{kind.title()} mentions must follow controller numbering.",
                )
            )

    for index, claim in enumerate(report.claims):
        location = f"claims[{index}]"
        missing = sorted(set(claim.evidence_refs) - known)
        if missing:
            findings.append(
                LintFinding(
                    code="unknown_evidence_ref",
                    location=location,
                    message=f"Claim references unknown sources: {', '.join(missing)}",
                )
            )
        if (
            claim.claim_type not in {"hypothesis"}
            and claim.status.value in {"supported", "partially_supported"}
            and not claim.evidence_refs
        ):
            findings.append(
                LintFinding(
                    code="supported_without_evidence",
                    location=location,
                    message="A supported non-hypothesis claim must cite evidence.",
                )
            )
        if claim.claim_type == "hypothesis" and claim.status.value == "supported":
            findings.append(
                LintFinding(
                    code="hypothesis_marked_supported",
                    location=location,
                    message="A hypothesis must not be labeled supported without reclassification.",
                )
            )

        if claim.status.value in {
            "supported",
            "partially_supported",
        }:
            referenced_sources = [
                sources_by_id[source_id]
                for source_id in claim.evidence_refs
                if source_id in sources_by_id
            ]
            if _METHODOLOGICAL_RECOMMENDATION.search(claim.text):
                scoped = _METHOD_RECOMMENDATION_SCOPE.search(claim.text)
                universal = _UNIVERSAL_METHOD_RECOMMENDATION.search(claim.text)
                bounded_universal = _BOUNDED_UNIVERSAL_METHOD_SCOPE.search(claim.text)
                if not scoped or (universal and not bounded_universal):
                    findings.append(
                        LintFinding(
                            code="methodological_recommendation_unscoped",
                            location=location,
                            message=(
                                "A method-selection recommendation must name the "
                                "studied simulation regime, distributions, sample-size "
                                "range, or variance-ratio conditions; do not convert "
                                "bounded evidence into a universal default."
                            ),
                        )
                    )
                if not any(
                    source.artifact_path or source.local_markdown_path
                    for source in referenced_sources
                ):
                    findings.append(
                        LintFinding(
                            code="methodological_recommendation_not_locally_grounded",
                            location=location,
                            message=(
                                "A method-selection recommendation cannot rely only "
                                "on unacquired web summaries; link a local computation "
                                "or acquired source passage, or narrow/remove it."
                            ),
                        )
                    )
            if _PROCEDURE_EQUIVALENCE.search(claim.text) and not any(
                source.artifact_path or source.local_markdown_path
                for source in referenced_sources
            ):
                findings.append(
                    LintFinding(
                        code="procedure_equivalence_not_verified",
                        location=location,
                        message=(
                            "A claim that complete tests, p-values, or results are "
                            "equivalent needs a reproducible calculation or an exact "
                            "locally acquired supporting passage; equality of an "
                            "assumption alone is insufficient."
                        ),
                    )
                )
            if any(source.retracted is True for source in referenced_sources):
                findings.append(
                    LintFinding(
                        code="retracted_source_used_as_support",
                        location=location,
                        message=(
                            "A retracted article cannot support a claim; it may be "
                            "discussed only as retracted or contradicted evidence."
                        ),
                    )
                )
            if claim.claim_type == "literature_supported":
                claim_terms = _grounding_terms(claim.text)
                for source in referenced_sources:
                    acquired_text = acquired_text_by_source.get(source.source_id)
                    if (
                        source.pmid
                        and acquired_text is not None
                        and claim_terms
                        and not any(term in acquired_text for term in claim_terms)
                    ):
                        findings.append(
                            LintFinding(
                                code="literature_claim_not_lexically_grounded",
                                location=location,
                                message=(
                                    f"Claim {claim.claim_id} has no informative term "
                                    f"overlap with acquired PubMed source "
                                    f"{source.source_id}; independent semantic audit "
                                    "cannot substitute for this gross-mismatch check."
                                ),
                            )
                        )
                precise_numbers = {
                    token.replace(" ", "")
                    for token in _PRECISE_LITERATURE_NUMBER.findall(claim.text)
                }
                acquired_linked_texts = [
                    acquired_text_by_source[source.source_id].replace(" ", "")
                    for source in referenced_sources
                    if source.source_id in acquired_text_by_source
                ]
                unsupported_numbers = sorted(
                    token
                    for token in precise_numbers
                    if acquired_linked_texts
                    and not any(token in text for text in acquired_linked_texts)
                )
                if unsupported_numbers:
                    findings.append(
                        LintFinding(
                            code="literature_claim_number_not_grounded",
                            location=location,
                            message=(
                                "Precise literature-derived values must appear in "
                                "at least one linked locally acquired article: "
                                + ", ".join(unsupported_numbers)
                            ),
                        )
                    )
            if claim.claim_type == "computed" and _PROTOCOL_TIMING.search(claim.text):
                findings.append(
                    LintFinding(
                        code="protocol_timing_not_computed",
                        location=location,
                        message=(
                            "A sandbox-generated analysis artifact cannot establish "
                            "that the protocol was locked before outcome inspection; "
                            "describe controller protocol provenance in Methods instead."
                        ),
                    )
                )
            if (
                claim.claim_type == "inference"
                and _METHODOLOGICAL_GENERALIZATION.search(claim.text)
                and not any(source.url for source in referenced_sources)
            ):
                findings.append(
                    LintFinding(
                        code="methodological_generalization_without_source",
                        location=location,
                        message=(
                            "A general claim about method robustness or validity needs "
                            "retrieved literature evidence, not only a computation artifact; "
                            "otherwise preserve it as an unresolved limitation."
                        ),
                    )
                )
            if (
                claim.claim_type == "computed"
                and referenced_sources
                and not any(source.artifact_path for source in referenced_sources)
            ):
                findings.append(
                    LintFinding(
                        code="computed_without_artifact",
                        location=location,
                        message="A computed claim must cite a sandbox-generated artifact.",
                    )
                )
            if (
                claim.claim_type == "literature_supported"
                and referenced_sources
                and not any(source.url for source in referenced_sources)
            ):
                findings.append(
                    LintFinding(
                        code="literature_without_url",
                        location=location,
                        message="A literature-supported claim must cite a retrieved URL.",
                    )
                )
            for source_id in claim.evidence_refs:
                source = sources_by_id.get(source_id)
                if source is None:
                    continue
                if source.url is not None:
                    if retrieval is None or retrieval.successful_calls == 0:
                        findings.append(
                            LintFinding(
                                code="supported_without_retrieval",
                                location=location,
                                message=(
                                    "A claim with an external source requires a "
                                    "successful retrieval tool call."
                                ),
                            )
                        )
                    if retrieval is None:
                        continue
                    retrieved_urls = {_normalize_url(url) for url in retrieval.urls}
                    source_url = _normalize_url(str(source.url))
                    if source_url not in retrieved_urls:
                        findings.append(
                            LintFinding(
                                code="source_url_not_retrieved",
                                location=f"{location}.evidence_refs",
                                message=(
                                    "Source URL was not present in retrieval output: "
                                    f"{source_url}"
                                ),
                            )
                        )
                    if retrieval.retrieval_dates and not any(
                        source.retrieved_at.startswith(date)
                        for date in retrieval.retrieval_dates
                    ):
                        findings.append(
                            LintFinding(
                                code="source_retrieval_date_mismatch",
                                location=f"sources[{source_id}].retrieved_at",
                                message=(
                                    "Source retrieval date does not match any recorded "
                                    f"tool-call date: {source.retrieved_at}"
                                ),
                            )
                        )
                elif source.artifact_path is not None:
                    artifact_path = os.path.normpath(source.artifact_path)
                    is_controller_artifact = artifact_path in controller_paths
                    if not is_controller_artifact and (
                        computation is None or computation.successful_calls == 0
                    ):
                        findings.append(
                            LintFinding(
                                code="supported_without_computation",
                                location=location,
                                message=(
                                    "A claim with an artifact source requires a "
                                    "successful sandbox computation."
                                ),
                            )
                        )
                    known_artifacts = {
                        os.path.normpath(artifact.path)
                        for artifact in (computation.artifacts if computation else [])
                    } | controller_paths
                    if artifact_path not in known_artifacts:
                        findings.append(
                            LintFinding(
                                code="source_artifact_not_generated",
                                location=f"{location}.evidence_refs",
                                message=(
                                    "Source artifact was not produced by a successful "
                                    f"sandbox run: {source.artifact_path}"
                                ),
                            )
                        )
                    evidence_dates = (
                        set(controller_dates)
                        if is_controller_artifact
                        else {
                            record.started_at[:10]
                            for record in (computation.records if computation else [])
                            if record.status == "succeeded"
                        }
                    )
                    if evidence_dates and not any(
                        source.retrieved_at.startswith(date) for date in evidence_dates
                    ):
                        findings.append(
                            LintFinding(
                                code=(
                                    "source_controller_date_mismatch"
                                    if is_controller_artifact
                                    else "source_computation_date_mismatch"
                                ),
                                location=f"sources[{source_id}].retrieved_at",
                                message=(
                                    "Artifact evidence date does not match its recorded "
                                    f"controller or computation date: {source.retrieved_at}"
                                ),
                            )
                        )

    return DeterministicValidation(
        passed=not any(f.blocking for f in findings), findings=findings
    )
