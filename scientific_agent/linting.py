"""Deterministic plan and claim-evidence checks."""

from __future__ import annotations

import ast
import json
import hashlib
import math
import os
import re
import unicodedata
import zipfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .literature import LiteratureError, load_acquired_article_record
from .reporting import (
    FIGURE_MEDIA_TYPES,
    MIN_REPORTED_FIGURE_DPI,
    TABLE_DELIMITERS,
    caption_has_number_prefix,
    extract_figure_ocr,
    excessive_table_precision,
    figure_annotation_overlap_candidates,
    inspect_figure,
    logical_report_output_key,
    read_table_preview,
    resolve_display_artifact,
)

from .schemas import (
    ArtifactRef,
    DeterministicValidation,
    ComputationEvidence,
    ComputationRecord,
    LintFinding,
    PlanLintReport,
    PlanProposal,
    RetrievalEvidence,
    ScientificReport,
    TaskSpec,
)


RECONCILIATION_JSON_MAX_BYTES = 1024 * 1024


def _bounded_json_object(path: Path) -> dict[str, Any] | None:
    """Read one strict, bounded JSON object."""

    try:
        if not path.is_file() or path.stat().st_size > RECONCILIATION_JSON_MAX_BYTES:
            return None
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _json_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _json_path_number(document: Any, json_path: str) -> float | None:
    if not json_path or not re.fullmatch(
        r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", json_path
    ):
        return None
    current = document
    for component in json_path.split("."):
        if not isinstance(current, dict) or component not in current:
            return None
        current = current[component]
    return _json_number(current)


def _nested_json_objects(value: Any, path: str = "$"):
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from _nested_json_objects(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _nested_json_objects(child, f"{path}[{index}]")


def _inferential_consistency_findings(path: Path, document: Any) -> list[LintFinding]:
    """Check that reported t, df, and p-value tuples are arithmetically possible."""

    try:
        from scipy.stats import t as student_t
    except ImportError:  # pragma: no cover - release runtimes include SciPy
        student_t = None
    findings: list[LintFinding] = []
    for json_path, node in _nested_json_objects(document):
        alias_groups = {
            "t statistic": ("t_statistic", "welch_t_statistic"),
            "degrees of freedom": (
                "degrees_freedom",
                "degrees_of_freedom",
                "welch_df",
                "df",
            ),
            "p-value": (
                "p_value",
                "p_value_two_sided",
                "welch_p_value",
                "welch_pvalue",
            ),
        }
        present = {
            label: [(key, node[key]) for key in aliases if key in node]
            for label, aliases in alias_groups.items()
        }
        if not all(present.values()):
            continue
        raw_values = tuple(value for values in present.values() for _, value in values)
        if all(value is None for value in raw_values):
            continue

        invalid_alias = False
        alias_conflict_recorded = False
        values_by_label: dict[str, float] = {}
        for label, aliases in present.items():
            numeric = [(key, _json_number(value)) for key, value in aliases]
            if any(value is None for _, value in numeric):
                invalid_alias = True
                break
            concrete = [(key, value) for key, value in numeric if value is not None]
            reference = concrete[0][1]
            if any(
                not math.isclose(value, reference, rel_tol=1e-12, abs_tol=1e-15)
                for _, value in concrete[1:]
            ):
                findings.append(
                    LintFinding(
                        code="inferential_statistic_inconsistent",
                        location=f"{path}:{json_path}",
                        message=(
                            f"Conflicting aliases were reported for {label}: "
                            + ", ".join(f"{key}={value:.8g}" for key, value in concrete)
                            + "."
                        ),
                    )
                )
                invalid_alias = True
                alias_conflict_recorded = True
                break
            values_by_label[label] = reference
        if invalid_alias:
            if not alias_conflict_recorded:
                findings.append(
                    LintFinding(
                        code="inferential_statistic_inconsistent",
                        location=f"{path}:{json_path}",
                        message=(
                            "A t statistic, degrees of freedom, and p-value must be "
                            "finite JSON numbers from the same test object; display "
                            "strings belong in separate fields."
                        ),
                    )
                )
            continue

        statistic = values_by_label["t statistic"]
        degrees_freedom = values_by_label["degrees of freedom"]
        reported_p = values_by_label["p-value"]
        if degrees_freedom <= 0 or not 0 <= reported_p <= 1:
            findings.append(
                LintFinding(
                    code="inferential_statistic_inconsistent",
                    location=f"{path}:{json_path}",
                    message="Degrees of freedom and p-values must lie in valid ranges.",
                )
            )
            continue
        if student_t is None:
            findings.append(
                LintFinding(
                    code="inferential_validator_unavailable",
                    location=f"{path}:{json_path}",
                    message=(
                        "SciPy is required to validate a reported t statistic, "
                        "degrees of freedom, and p-value; install the analysis "
                        "dependencies instead of accepting this tuple unchecked."
                    ),
                )
            )
            continue
        lower_tail = float(student_t.cdf(statistic, degrees_freedom))
        upper_tail = float(student_t.sf(statistic, degrees_freedom))
        two_sided = min(1.0, 2.0 * float(student_t.sf(abs(statistic), degrees_freedom)))
        raw_alternative = node.get("alternative")
        alternative = re.sub(
            r"[\s_.]+", "-", str(raw_alternative or "").strip().casefold()
        )
        if alternative in {"less", "lower", "left", "left-tailed"}:
            valid_probabilities = (lower_tail,)
        elif alternative in {"greater", "upper", "right", "right-tailed"}:
            valid_probabilities = (upper_tail,)
        elif alternative in {"two-sided", "two-tailed"}:
            valid_probabilities = (two_sided,)
        elif not alternative:
            valid_probabilities = (lower_tail, upper_tail, two_sided)
        else:
            findings.append(
                LintFinding(
                    code="inferential_statistic_inconsistent",
                    location=f"{path}:{json_path}",
                    message=(
                        "The reported alternative hypothesis is not recognized; "
                        "use less, greater, or two-sided so the p-value tail can "
                        "be validated."
                    ),
                )
            )
            continue
        matches_valid_tail = any(
            math.isclose(
                reported_p,
                expected,
                rel_tol=1e-6,
                abs_tol=1e-300,
            )
            for expected in valid_probabilities
        )
        if not matches_valid_tail:
            findings.append(
                LintFinding(
                    code="inferential_statistic_inconsistent",
                    location=f"{path}:{json_path}",
                    message=(
                        "The reported t statistic, degrees of freedom, and p-value "
                        "cannot represent either a one- or two-sided Student-t test: "
                        f"t={statistic:.8g}, df={degrees_freedom:.8g}, "
                        f"p={reported_p:.8g}; expected lower-tail {lower_tail:.8g}, "
                        f"upper-tail {upper_tail:.8g}, or two-sided {two_sided:.8g}."
                    ),
                )
            )
    return findings


def reconciliation_verdict(
    path: Path,
    computation: ComputationEvidence | None = None,
) -> bool | None:
    """Independently validate one cross-language reconciliation artifact.

    A model-authored top-level boolean is not evidence. Each comparison must bind
    Python and R values to successful, content-hashed JSON artifacts; this function
    reloads those values and recomputes the declared difference and verdict.
    """

    value = _bounded_json_object(path)
    if value is None or computation is None:
        return None
    try:
        reconciliation_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None
    normalized_reconciliation_path = os.path.normpath(str(path))
    if not any(
        record.status == "succeeded"
        and any(
            os.path.normpath(artifact.path) == normalized_reconciliation_path
            and artifact.sha256 == reconciliation_digest
            and artifact.description == "sandbox-generated analysis artifact"
            for artifact in record.artifacts
        )
        for record in computation.records
    ):
        return None
    top_level_verdicts = [
        candidate
        for key in (
            "all_pass",
            "passed",
            "within_tolerance",
            "reconciliation_passed",
        )
        if isinstance((candidate := value.get(key)), bool)
    ]
    verdict = top_level_verdicts[0] if top_level_verdicts else None
    comparisons = value.get("comparisons")
    if (
        verdict is None
        or any(candidate is not verdict for candidate in top_level_verdicts[1:])
        or not isinstance(comparisons, list)
        or not comparisons
    ):
        return None

    artifacts_by_language_and_hash: dict[tuple[str, str], Path] = {}
    for record in computation.records:
        if record.status != "succeeded":
            continue
        for artifact in record.artifacts:
            if (
                artifact.description == "sandbox-generated analysis artifact"
                and artifact.sha256
                and re.fullmatch(r"[0-9a-fA-F]{64}", artifact.sha256)
                and Path(artifact.path).suffix.lower() == ".json"
            ):
                artifacts_by_language_and_hash[
                    (record.language, artifact.sha256.lower())
                ] = Path(artifact.path)

    observed_passes: list[bool] = []
    bound_documents: dict[tuple[str, str], dict[str, object]] = {}
    bound_document_pairs: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for comparison in comparisons:
        if (
            not isinstance(comparison, dict)
            or not str(comparison.get("metric", "")).strip()
        ):
            return None
        sources: dict[str, float] = {}
        comparison_document_keys: dict[str, tuple[str, str]] = {}
        for side in ("python", "r"):
            source = comparison.get(side)
            if not isinstance(source, dict) or source.get("language") != side:
                return None
            digest = source.get("artifact_sha256")
            json_path = source.get("json_path")
            declared = _json_number(source.get("value"))
            if (
                not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
                or not isinstance(json_path, str)
                or declared is None
            ):
                return None
            source_path = artifacts_by_language_and_hash.get((side, digest.lower()))
            if source_path is None:
                return None
            try:
                if (
                    hashlib.sha256(source_path.read_bytes()).hexdigest()
                    != digest.lower()
                ):
                    return None
            except OSError:
                return None
            document = _bounded_json_object(source_path)
            observed = _json_path_number(document, json_path) if document else None
            if observed is None or not math.isclose(
                declared, observed, rel_tol=1e-12, abs_tol=1e-12
            ):
                return None
            if document is not None:
                document_key = (side, digest.lower())
                previous = bound_documents.get(document_key)
                if previous is not None and previous != document:
                    return None
                bound_documents[document_key] = document
                comparison_document_keys[side] = document_key
            sources[side] = observed

        if set(comparison_document_keys) == {"python", "r"}:
            bound_document_pairs.add(
                (
                    comparison_document_keys["python"],
                    comparison_document_keys["r"],
                )
            )

        tolerance = _json_number(comparison.get("tolerance"))
        declared_difference = _json_number(comparison.get("absolute_difference"))
        declared_pass = comparison.get("passed")
        if (
            tolerance is None
            or tolerance < 0
            or declared_difference is None
            or declared_difference < 0
            or not isinstance(declared_pass, bool)
        ):
            return None
        observed_difference = abs(sources["python"] - sources["r"])
        if not math.isclose(
            declared_difference,
            observed_difference,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            return None
        observed_pass = observed_difference <= tolerance
        if declared_pass is not observed_pass:
            return None
        observed_passes.append(observed_pass)

    # Shared structural input checks with the same JSON path have the same
    # declared meaning in both language artifacts. Do not certify numerical
    # agreement while those machine-readable conclusions contradict one another.
    # Method-dependent booleans such as normality-test decisions are intentionally
    # excluded because different implementations can use different tests.
    def diagnostic_booleans(document: dict[str, object]) -> dict[str, bool]:
        found: dict[str, bool] = {}

        def visit(value: object, path: str) -> None:
            if isinstance(value, bool):
                found[path] = value
            elif isinstance(value, dict):
                for key, child in value.items():
                    visit(child, f"{path}.{key}" if path else str(key))

        for section in ("diagnostics", "data_quality", "validation"):
            value = document.get(section)
            if isinstance(value, dict):
                visit(value, section)
        return found

    for python_key, r_key in bound_document_pairs:
        python_diagnostics = diagnostic_booleans(bound_documents[python_key])
        r_diagnostics = diagnostic_booleans(bound_documents[r_key])
        if any(
            python_diagnostics[path] is not r_diagnostics[path]
            for path in python_diagnostics.keys() & r_diagnostics.keys()
            if re.search(
                r"(?:^|_)(?:missing|duplicate|duplicated|schema|empty|finite|nan|inf|infinite)(?:_|$)",
                path.rsplit(".", 1)[-1].lower(),
            )
        ):
            observed_passes.append(False)

    observed_verdict = all(observed_passes)
    return verdict if verdict is observed_verdict else None


_WORD = re.compile(r"[a-z0-9]{4,}")
_ARBITRARY_SEMANTIC_ARM_MAPPING = re.compile(
    r"\b(?:lexicograph\w*|alphabet\w*|numeric\w*|row|categor\w*)\b"
    r".{0,120}\b(?:control|treatment|intervention|comparison|reference)\b"
    r"|\b(?:control|treatment|intervention|comparison|reference)\b"
    r".{0,120}\b(?:lexicograph\w*|alphabet\w*|numeric\w*|row|categor\w*)\b"
    r"|\b(?:assign|map|designat|classif|infer)\w*\b.{0,120}"
    r"\b(?:based on|according to|using|higher|lower|larger|smaller|"
    r"maximum|minimum)\b.{0,80}\b(?:baseline|outcome|covariate|"
    r"group size|sample size|missing\w*|effect\w*|response)\b.{0,120}"
    r"\b(?:control|treatment|intervention|comparison|reference)\b"
    r"|\b(?:control|treatment|intervention|comparison|reference)\b"
    r".{0,80}\b(?:assign|map|designat|classif|infer)\w*\b.{0,120}"
    r"\b(?:based on|according to|using|higher|lower|larger|smaller|"
    r"maximum|minimum)\b.{0,80}\b(?:baseline|outcome|covariate|"
    r"group size|sample size|missing\w*|effect\w*|response)\b",
    re.IGNORECASE,
)
_ASSUMPTION_DIAGNOSTIC = re.compile(
    r"\b(?:shapiro(?:-wilk)?|normality(?:\s+test)?|outliers?|"
    r"standard deviations?)\b",
    re.IGNORECASE,
)
_MAPPING_LITERAL = re.compile(r"\{[^{}\n]{1,1000}\}")
_SEMANTIC_ROLE_NAMES = {
    "case",
    "comparator",
    "control",
    "exposed",
    "intervention",
    "placebo",
    "reference",
    "treatment",
    "unexposed",
}
_INVALID_HEDGES_J_PARENTHESES = re.compile(
    r"(?:1\s*-\s*)?3\s*/\s*\(\s*4\s*\*\s*\(\s*(?:n|N)\s*-\s*9\s*\)\s*\)",
)
_DESIGN_CLASSIFICATION_ASSERTION = re.compile(
    r"\b(?:is|was|are|were|assum(?:e[sd]?|ing)(?:\s+to\s+be)?|"
    r"classif(?:y|ies|ied|ying)(?:\s+as)?|treat(?:s|ed|ing)?\s+as)\s+"
    r"(?:strictly\s+)?(?:an?\s+)?"
    r"(?P<design>observational|randomi[sz]ed|experimental|synthetic|representative)\b",
    re.IGNORECASE,
)
_TASK_DESIGN_CLASSIFICATION = {
    "observational": re.compile(
        r"\bobservational\s+(?:cohort|data|dataset|design|study)\b", re.IGNORECASE
    ),
    "randomized": re.compile(
        r"\brandomi[sz]ed\s+(?:allocation|controlled\s+trial|design|study|trial)\b",
        re.IGNORECASE,
    ),
    "experimental": re.compile(
        r"\bexperimental\s+(?:data|dataset|design|study)\b", re.IGNORECASE
    ),
    "synthetic": re.compile(
        r"\bsynthetic\s+(?:data|dataset|fixture|study)\b", re.IGNORECASE
    ),
    "representative": re.compile(
        r"\brepresentative\s+(?:cohort|population|sample)\b", re.IGNORECASE
    ),
}
_EXPLICIT_DIAGNOSTIC_ACTION = re.compile(
    r"\b(?:stop|halt|abort|terminate|exclude|drop|omit|remove)\w*\b"
    r".{0,120}\b(?:shapiro(?:-wilk)?|normality(?:\s+test)?|outliers?|"
    r"standard deviations?)\b"
    r"|\b(?:shapiro(?:-wilk)?|normality(?:\s+test)?|outliers?|"
    r"standard deviations?)\b.{0,120}"
    r"\b(?:stop|halt|abort|terminate|exclude|drop|omit|remove)\w*\b",
    re.IGNORECASE,
)
_PROTOCOL_TIMING = re.compile(
    r"\b(?:lock(?:ed|ing)?|prespecif(?:ied|ication))\b.{0,100}"
    r"\b(?:before|prior to)\b.{0,100}\b(?:inspect(?:ion|ing)?|outcome|result)",
    re.IGNORECASE,
)
_PROTOCOL_BEFORE_DATA_INSPECTION = re.compile(
    r"\b(?:lock(?:ed|ing)?|prespecif(?:ied|ication))\b.{0,120}"
    r"\b(?:before|prior to)\b.{0,40}\bdata inspection\b",
    re.IGNORECASE,
)
_AI_ROLE_UNDERSTATEMENT = re.compile(
    r"\bAI\b.{0,100}\b(?:only|solely)\b.{0,120}"
    r"\b(?:draft(?:ing|ed)?|writ(?:ing|ten)|artifact registration)\b",
    re.IGNORECASE,
)
_BALANCED_DESIGN_REASSURANCE = re.compile(
    r"\bbalanced\b.{0,80}\b(?:design|groups?|sample sizes?)\b.{0,140}"
    r"\b(?:mitigat(?:e[sd]?|ing)|protect(?:s|ed|ing)?|robust)\b.{0,100}"
    r"\b(?:type\s*i\s*error|non[- ]?normal(?:ity)?|normality|assumption)\b|"
    r"\bbalanced\b.{0,80}\b(?:mitigat(?:e[sd]?|ing)|protect(?:s|ed|ing)?)\b"
    r".{0,100}\btype\s*i\s*error\b|"
    r"\b(?:normality|non[- ]?normality|distributional\s+assumptions?)\b"
    r".{0,180}\bbalanced\b.{0,80}\b(?:design|groups?|sample sizes?)\b"
    r".{0,140}\b(?:mitigat(?:e[sd]?|ing)|reassur\w*|"
    r"reduc\w*\s+(?:concern|sensitivity))\b",
    re.IGNORECASE,
)
_UNQUALIFIED_RESULT_ROBUSTNESS = re.compile(
    r"\b(?:analysis|association|contrast|estimate|finding|result)s?\b.{0,80}"
    r"\b(?:is|are|was|were)\s+(?:statistically\s+)?robust\b|"
    r"\brobust\b.{0,50}\b(?:analysis|association|contrast|estimate|finding|result)s?\b",
    re.IGNORECASE,
)
_GROUP_MEAN_UNIFORMITY_OVERCLAIM = re.compile(
    r"\b(?:group|condition|arm)\b.{0,100}\b"
    r"(?:show(?:s|ed|ing)?|exhibit(?:s|ed|ing)?)\b.{0,30}\buniform\b"
    r".{0,30}\b(?:increase|decrease|change|response)\b|"
    r"\buniform\b.{0,30}\b(?:increase|decrease|change|response)\b"
    r".{0,80}\b(?:group|condition|arm|mean)\b",
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
_PLANNED_INPUT_FILENAME = re.compile(
    r"(?<![A-Za-z0-9._-])([A-Za-z0-9][A-Za-z0-9._-]*\."
    r"(?:csv|tsv|txt|json|jsonl|parquet|feather|xlsx?|ods|rds|rdata|"
    r"png|jpe?g|webp|tiff?|svg|pdf|docx?|pptx?|zip|tar|gz|bz2|xz|"
    r"py|r|rmd|qmd|ipynb|fasta|fastq|bam|sam|vcf|bed|gff3?|gtf))\b",
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
_CITATION_CORRESPONDENCE_STOPWORDS = _GROUNDING_STOPWORDS | {
    "after",
    "also",
    "among",
    "before",
    "between",
    "during",
    "following",
    "from",
    "have",
    "only",
    "source",
    "that",
    "their",
    "these",
    "this",
    "those",
    "using",
    "were",
    "with",
    "without",
}
_CITATION_NUMBER = re.compile(
    r"(?<![\w.])(?P<number>[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)"
    r"(?:[eE][+-]?\d+)?)(?P<percent>\s*%)?(?!(?:\w|[.,]\d))"
)
_NUMERIC_CELL = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_PRECISE_LITERATURE_NUMBER = re.compile(
    r"(?<![\w.])(?:0\.\d+|\d+(?:\.\d+)?\s*%)(?![\w.])"
)
_TABLE_KEY_COLUMNS = {"measure", "metric", "outcome", "parameter", "statistic"}
_TABLE_VALUE_COLUMNS = {"estimate", "result", "value"}
_GENERIC_NUMERIC_COLUMNS = _TABLE_VALUE_COLUMNS | {"statistic"}
_GENERIC_WIDE_ESTIMATE_COLUMNS = _TABLE_VALUE_COLUMNS | {"difference", "overall"}
_GROUP_HEADER_TOKENS = {
    "case",
    "cases",
    "comparator",
    "comparators",
    "control",
    "controls",
    "exposed",
    "intervention",
    "interventions",
    "placebo",
    "placebos",
    "treatment",
    "treatments",
    "unexposed",
}
_EQUATION_OPERANDS = re.compile(
    r"(?P<left>[^\s,;:=()]+)\s*(?<![<>])=(?!=)\s*"
    r"(?P<right>[^\s,;:=().]+)",
    re.UNICODE,
)
_COMPUTED_DIAGNOSTIC_TERMS = {
    "shapiro": re.compile(r"\bshapiro(?:-wilk)?\b", re.IGNORECASE),
    "levene": re.compile(r"\blevene(?:'s)?\b", re.IGNORECASE),
}
_OVERSTATED_SENSITIVITY_LANGUAGE = re.compile(
    r"\b(?:confirm(?:s|ed|ing)?\s+(?:the\s+)?robustness|"
    r"(?:baseline|age|covariates?).{0,60}\bdid not (?:materially )?confound|"
    r"validat(?:e[sd]?|ing) (?:the )?(?:analytical|analysis) pipeline|"
    r"confirm(?:s|ed|ing)? algorithmic equivalence|"
    r"sensitivity analys(?:is|es).{0,50}\bconfirm(?:s|ed|ing)? stability)\b",
    re.IGNORECASE,
)
_ASSUMPTION_ACCEPTANCE_LANGUAGE = re.compile(
    r"\b(?:normality|homoscedasticity|equal[- ]variance).{0,100}"
    r"\bassumptions? (?:are |were )?(?:met|satisfied|supported)\b|"
    r"\bassumptions? (?:are |were )?(?:met|satisfied|supported).{0,100}"
    r"\b(?:shapiro|levene|normality|homoscedasticity|equal[- ]variance)\b",
    re.IGNORECASE,
)
_WELCH_NORMALITY_OVERCLAIM = re.compile(
    r"\b(?:welch(?:'s)?(?:\s+(?:t[- ]?test|procedure|test))?).{0,140}"
    r"\b(?:accommodat(?:e[sd]?|ing)|account(?:s|ed|ing)?\s+for|correct(?:s|ed|ing)?\s+for|"
    r"handle(?:s|d|ing)?|remain(?:s|ed)?\s+applicable|robust)\b.{0,100}"
    r"\b(?:non[- ]?normal(?:ity)?|normality|gaussian)\b|"
    r"\b(?:non[- ]?normal(?:ity)?|normality|gaussian)\b.{0,140}"
    r"\b(?:welch(?:'s)?(?:\s+(?:t[- ]?test|procedure|test))?).{0,100}"
    r"\b(?:accommodat(?:e[sd]?|ing)|handle(?:s|d|ing)?|remain(?:s|ed)?\s+applicable|robust)\b",
    re.IGNORECASE,
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


def _citation_correspondence_terms(text: str) -> set[str]:
    """Return conservative content terms for an anchor-to-claim gross check."""

    decomposed = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return set(_WORD.findall(folded)) - _CITATION_CORRESPONDENCE_STOPWORDS


def _citation_correspondence_numbers(text: str) -> set[str]:
    """Normalize numerical tokens without treating formatting as disagreement."""

    values: set[str] = set()
    for match in _CITATION_NUMBER.finditer(text):
        raw = match.group("number").replace(",", ".")
        try:
            normalized = format(Decimal(raw).normalize(), "f")
        except InvalidOperation:
            continue
        values.add(normalized + ("%" if match.group("percent") else ""))
    return values


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


def lint_plan(
    task: TaskSpec,
    plan: PlanProposal,
    *,
    controller_method_lock: bool = False,
) -> PlanLintReport:
    findings: list[LintFinding] = []
    plan_text_parts = [
        plan.objective,
        *plan.assumptions,
        *plan.required_data,
        *plan.alternatives_considered,
        *plan.foreseeable_failure_modes,
        *plan.expected_artifacts,
        *plan.unresolved_questions,
        *plan.estimated_resources,
    ]
    for step in plan.steps:
        plan_text_parts.extend(
            [
                step.objective,
                *step.inputs,
                *step.outputs,
                *step.methods,
                *(validator.description for validator in step.validators),
                *step.stop_conditions,
            ]
        )
    plan_text = " ".join(plan_text_parts)
    observed_role_columns = [
        (source.path, column.name, set(column.candidate_role_labels))
        for source in (task.input_profile.files if task.input_profile else [])
        for column in source.columns
        if column.candidate_role_labels_complete and column.candidate_role_labels
    ]
    for match in _MAPPING_LITERAL.finditer(plan_text):
        try:
            mapping = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        if not isinstance(mapping, dict) or not mapping:
            continue
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in mapping.items()
        ):
            continue
        mapped_roles = {value.casefold() for value in mapping.values()}
        if len(mapped_roles & _SEMANTIC_ROLE_NAMES) < 2:
            continue
        mapping_keys = {key.casefold() for key in mapping}
        if any(
            mapping_keys == {label.casefold() for label in labels}
            for _path, _name, labels in observed_role_columns
        ):
            continue
        observed = (
            "; ".join(
                f"{path}:{name}={sorted(labels)}"
                for path, name, labels in observed_role_columns
            )
            or "no complete candidate role-label set was exposed"
        )
        findings.append(
            LintFinding(
                code="role_mapping_not_grounded_in_input_profile",
                location="plan",
                message=(
                    "An explicit semantic role mapping must use exactly the bounded "
                    "category labels recorded by the immutable input profile; critic "
                    f"examples are not data. Observed role columns: {observed}."
                ),
            )
        )
    if "hedges" in task.objective.casefold() and _INVALID_HEDGES_J_PARENTHESES.search(
        plan_text
    ):
        findings.append(
            LintFinding(
                code="invalid_hedges_j_parentheses",
                location="plan",
                message=(
                    "The Hedges small-sample correction is 1 - 3/(4*N - 9), "
                    "where N is the total sample size; 1 - 3/(4*(N - 9)) is a "
                    "different and invalid formula. Preserve the task-specified "
                    "parentheses exactly."
                ),
            )
        )
    task_design_text = " ".join([task.objective, *task.constraints])
    asserted_designs = {
        (
            "randomized"
            if match.group("design").casefold() in {"randomised", "randomized"}
            else match.group("design").casefold()
        )
        for match in _DESIGN_CLASSIFICATION_ASSERTION.finditer(plan_text)
    }
    for design in sorted(asserted_designs):
        if _TASK_DESIGN_CLASSIFICATION[design].search(task_design_text):
            continue
        findings.append(
            LintFinding(
                code="unsupported_plan_design_classification",
                location="plan",
                message=(
                    f"The plan classifies the input as {design}, but the user task "
                    "and controller profile do not establish that design. State that "
                    "allocation/sampling design is unspecified and constrain causal "
                    "and generalizability claims instead."
                ),
            )
        )
    step_ids = [step.step_id for step in plan.steps]
    if len(step_ids) != len(set(step_ids)):
        findings.append(
            LintFinding(
                code="duplicate_step_id",
                location="steps",
                message="Plan step IDs must be unique.",
            )
        )

    available_names = {
        Path(item.path).name.casefold() for item in task.available_inputs
    }
    produced_names = {
        Path(item).name.casefold()
        for item in [
            *plan.expected_artifacts,
            *(output for step in plan.steps for output in step.outputs),
        ]
    }
    objective_names = {
        match.group(1).casefold()
        for match in _PLANNED_INPUT_FILENAME.finditer(task.objective)
    }
    acquisition_text = " ".join(
        [
            *plan.required_data,
            *(step.objective for step in plan.steps),
            *(method for step in plan.steps for method in step.methods),
        ]
    ).casefold()
    declares_acquisition = any(
        term in acquisition_text
        for term in ("acquire", "download", "fetch", "import", "retrieve")
    )
    task_text = " ".join([task.objective, *task.constraints])
    task_authorizes_diagnostic_action = bool(
        _EXPLICIT_DIAGNOSTIC_ACTION.search(task_text)
    )
    declared_inputs = [
        *(
            ("required_data", index, item)
            for index, item in enumerate(plan.required_data)
        ),
        *(
            (f"steps[{step_index}].inputs", input_index, item)
            for step_index, step in enumerate(plan.steps)
            for input_index, item in enumerate(step.inputs)
        ),
    ]
    for field, index, value in declared_inputs:
        for match in _PLANNED_INPUT_FILENAME.finditer(value):
            name = match.group(1).casefold()
            if (
                name in available_names
                or name in produced_names
                or (name in objective_names and declares_acquisition)
            ):
                continue
            findings.append(
                LintFinding(
                    code="unknown_plan_input_artifact",
                    location=f"{field}[{index}]",
                    message=(
                        f"Plan input {match.group(1)} is neither an immutable "
                        "uploaded input nor a declared plan output; use an exact "
                        "available filename or a generic input description."
                    ),
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
        if (
            task.scientific_risk in {"confirmatory", "decision_critical"}
            and not task_authorizes_diagnostic_action
        ):
            for stop_index, stop_condition in enumerate(step.stop_conditions):
                if _ASSUMPTION_DIAGNOSTIC.search(stop_condition):
                    findings.append(
                        LintFinding(
                            code="data_dependent_primary_analysis_stop",
                            location=f"{location}.stop_conditions[{stop_index}]",
                            message=(
                                "A normality or outlier diagnostic cannot "
                                "automatically halt, exclude observations from, "
                                "or replace a locked primary analysis unless the "
                                "user-supplied protocol explicitly authorizes that "
                                "decision rule. Keep the primary analysis, report "
                                "the diagnostic, and predefine a sensitivity "
                                "analysis instead."
                            ),
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
        step_text = " ".join([step.objective, *step.methods, *step.stop_conditions])
        if _ARBITRARY_SEMANTIC_ARM_MAPPING.search(step_text):
            findings.append(
                LintFinding(
                    code="arbitrary_semantic_arm_mapping",
                    location=location,
                    message=(
                        "Semantic control/treatment identity cannot be inferred "
                        "from lexical, alphabetical, numeric, row, or category "
                        "order, or from observed baselines, outcomes, covariates, "
                        "group sizes, missingness, or effect direction/magnitude. "
                        "Predefine accepted normalized role labels and stop for "
                        "explicit mapping when labels are unrecognized or ambiguous."
                    ),
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

    if (
        task.scientific_risk in {"confirmatory", "decision_critical"}
        and not controller_method_lock
    ):
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


def _authoritative_json_outputs(
    records: list[ComputationRecord],
) -> dict[tuple[str, str], str]:
    """Return the latest successful path for each language/logical JSON output."""

    latest: dict[tuple[str, str], str] = {}
    for record in records:
        if record.status != "succeeded":
            continue
        for artifact in record.artifacts:
            path = Path(artifact.path)
            if (
                artifact.description == "sandbox-generated analysis artifact"
                and path.suffix.lower() == ".json"
            ):
                latest[(record.language, _logical_json_output_key(path))] = (
                    os.path.normpath(str(path))
                )
    return latest


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

    if computation and computation.records:
        latest = {
            key: Path(path)
            for key, path in _authoritative_json_outputs(computation.records).items()
        }
    else:
        latest = {}
        for artifact in computation.artifacts if computation else []:
            path = Path(artifact.path)
            if (
                path.suffix.lower() != ".json"
                or _is_report_output(path, "figures")
                or _is_report_output(path, "tables")
            ):
                continue
            latest[("", _logical_json_output_key(path))] = path

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


def _reported_degrees_of_freedom_findings(
    results_text: str,
    machine_numbers: dict[str, list[Decimal]],
) -> list[LintFinding]:
    candidates = {
        value
        for key in (
            "df",
            "df_welch",
            "welch_df",
            "degrees_freedom",
            "degrees_of_freedom",
        )
        for value in machine_numbers.get(key, [])
    }
    if not candidates:
        return []
    findings = []
    for match in re.finditer(
        r"\bdf\s*(?:[=:≈~]\s*)?([0-9]+(?:\.[0-9]+)?)",
        results_text,
        flags=re.IGNORECASE,
    ):
        reported = match.group(1)
        if any(_numeric_cell_agrees(reported, candidate) for candidate in candidates):
            continue
        findings.append(
            LintFinding(
                code="reported_degrees_of_freedom_not_in_machine_results",
                location="results",
                message=(
                    f"Reported df={reported} does not match any degrees of freedom "
                    "in the latest successful machine-readable results; correct the "
                    "article from the authoritative computation artifact."
                ),
            )
        )
    return findings


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
    if len(key_indexes) == 1:
        key_index = key_indexes[0]
        for row_number, row in enumerate(rows, start=1):
            if key_index >= len(row):
                continue
            row_key = _numeric_key(str(row[key_index]))
            if not row_key:
                continue
            for column_index, column_key in enumerate(normalized_columns):
                if column_index == key_index or column_index >= len(row):
                    continue
                cell = str(row[column_index]).strip()
                if not _NUMERIC_CELL.fullmatch(cell):
                    continue
                if column_key in _GENERIC_WIDE_ESTIMATE_COLUMNS:
                    candidates = machine_numbers.get(row_key, [])
                else:
                    # A group-labelled column is not itself a metric. Require an
                    # explicit compound JSON identity (for example,
                    # groups.treatment.baseline_mean) before comparing the cell.
                    # This prevents a Treatment column from being compared with
                    # unrelated JSON values stored under a bare `treatment` key.
                    compound_keys = {
                        f"{column_key}_{row_key}",
                        f"{row_key}_{column_key}",
                    }
                    candidates = [
                        value
                        for machine_key, values in machine_numbers.items()
                        if any(
                            machine_key == compound
                            or machine_key.endswith(f"_{compound}")
                            for compound in compound_keys
                        )
                        for value in values
                    ]
                if candidates and not any(
                    _numeric_cell_agrees(cell, value) for value in candidates
                ):
                    values = ", ".join(str(value) for value in candidates[:3])
                    cell_label = (
                        f"{row[key_index]}={cell}"
                        if column_key in _TABLE_VALUE_COLUMNS
                        else f"{row[key_index]} / {columns[column_index]}={cell}"
                    )
                    examples.append(f"row {row_number}, {cell_label}; JSON: {values}")
                    if len(examples) >= example_limit:
                        return examples
        return examples

    for column_index, key in enumerate(normalized_columns):
        if key in _GENERIC_NUMERIC_COLUMNS or (
            set(key.split("_")) & _GROUP_HEADER_TOKENS
        ):
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


def _ambiguous_group_table_rows(
    preview: dict[str, Any], *, example_limit: int = 5
) -> list[str]:
    """Find overall estimates incorrectly placed beneath one group header."""

    columns = [str(column).strip() for column in preview.get("columns", [])]
    normalized = [_numeric_key(column) for column in columns]
    group_indexes = {
        index
        for index, header in enumerate(normalized[1:], start=1)
        if set(header.split("_")) & _GROUP_HEADER_TOKENS
    }
    if len(group_indexes) < 2:
        return []
    overall_markers = (
        "estimand",
        "difference",
        "p_value",
        "pvalue",
        "cohen",
        "hedges",
        "adjusted_effect",
        "r_squared",
        "rsquared",
        "confidence_interval",
        "95_ci",
    )
    examples = []
    for row_number, row in enumerate(preview.get("rows", []), start=1):
        if not row:
            continue
        label = _numeric_key(str(row[0]))
        if not any(marker in label for marker in overall_markers):
            continue
        populated = [
            index for index, value in enumerate(row[1:], start=1) if str(value).strip()
        ]
        if len(populated) != 1:
            continue
        index = populated[0]
        if index >= len(columns) or index not in group_indexes:
            continue
        examples.append(f"row {row_number} ({row[0]}) under {columns[index]}")
        if len(examples) >= example_limit:
            break
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
    if (
        re.search(r"\bmean\s+(?:difference|diff)\b", text)
        and re.search(r"\bd\s*=\s*[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)", text)
        and not re.search(r"\bcohen(?:'s)?\s+d\b", text)
    ):
        findings.append(
            (
                "figure_ambiguous_bare_d_label",
                "The rendered figure labels a mean-difference display with bare "
                "'d = ...', which is readily confused with Cohen d. Spell out "
                "'mean difference' for the raw estimand or 'Cohen d' for the "
                "standardized effect.",
            )
        )
    return findings


def _figure_caption_semantic_findings(
    caption: str, alt_text: str, ocr: dict[str, Any]
) -> list[tuple[str, str]]:
    """Reject high-confidence claims about annotations absent from the raster."""

    if not ocr.get("available"):
        return []
    narrative = " ".join(f"{caption} {alt_text}".casefold().split())
    visible = " ".join(str(ocr.get("text", "")).casefold().split())
    r_squared = r"(?:adjusted\s+)?r(?:\s*[- ]?squared|\s*[²2])"
    claims_visible_annotation = re.search(
        rf"\b(?:annotation|label|panel|figure)s?\b.{{0,100}}\b{r_squared}\b",
        narrative,
    ) or re.search(
        rf"\b{r_squared}\b.{{0,100}}\b(?:annotation|label|panel|figure)s?\b",
        narrative,
    )
    if claims_visible_annotation and not re.search(rf"\b{r_squared}\b", visible):
        return [
            (
                "figure_caption_claims_missing_annotation",
                "The caption or alt text says the figure contains an R-squared "
                "annotation, but no such label is recoverable from the rendered "
                "raster; correct the caption or add the promised annotation.",
            )
        ]
    return []


def _figure_source_semantic_findings(
    artifact_path: Path,
    computation: ComputationEvidence | None,
) -> list[tuple[str, str]]:
    """Reject high-confidence Python plotting-source semantic defects."""

    if computation is None:
        return []
    normalized_figure = os.path.normpath(str(artifact_path))
    source_paths: list[Path] = []
    for record in computation.records:
        if record.status != "succeeded" or not any(
            os.path.normpath(artifact.path) == normalized_figure
            for artifact in record.artifacts
        ):
            continue
        source_paths.extend(
            Path(artifact.path)
            for artifact in record.artifacts
            if artifact.description == "python analysis source"
            and Path(artifact.path).suffix.lower() == ".py"
        )

    for source_path in source_paths:
        try:
            source_bytes = source_path.read_bytes()
            if len(source_bytes) > 512 * 1024:
                continue
            tree = ast.parse(source_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        effect_x_axes: set[str] = set()
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "set_xlabel"
                and isinstance(node.func.value, ast.Name)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                continue
            label = node.args[0].value.casefold()
            if any(
                term in label
                for term in (
                    "mean difference",
                    "difference in mean",
                    "difference in change",
                    "effect estimate",
                    "contrast",
                )
            ):
                effect_x_axes.add(node.func.value.id)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if (
                node.func.attr == "set_xticks"
                and node.args
                and isinstance(node.args[0], (ast.List, ast.Tuple))
            ):
                tick_values = [
                    item.value
                    for item in node.args[0].elts
                    if isinstance(item, ast.Constant)
                    and isinstance(item.value, (int, float))
                    and not isinstance(item.value, bool)
                ]
                if len(tick_values) == len(node.args[0].elts) and len(
                    set(tick_values)
                ) < len(tick_values):
                    return [
                        (
                            "figure_duplicate_category_positions",
                            "The plotting source assigns duplicate x-axis tick "
                            "positions to distinct categories, so groups overlap. "
                            "Use one unique categorical position per group and "
                            "regenerate the figure.",
                        )
                    ]
            if node.func.attr == "scatter":
                y_expression: ast.AST | None = (
                    node.args[1] if len(node.args) > 1 else None
                )
                if y_expression is None:
                    y_expression = next(
                        (
                            keyword.value
                            for keyword in node.keywords
                            if keyword.arg == "y"
                        ),
                        None,
                    )
                if isinstance(y_expression, (ast.BinOp, ast.AugAssign)):
                    identifiers = {
                        child.id.casefold()
                        for child in ast.walk(y_expression)
                        if isinstance(child, ast.Name)
                    }
                    if any("jitter" in identifier for identifier in identifiers):
                        return [
                            (
                                "figure_numeric_axis_jitter",
                                "The plotting source adds jitter to the quantitative "
                                "y-axis values in a scatter plot, so displayed raw "
                                "observations no longer equal the source data. Apply "
                                "jitter only to the categorical position axis and "
                                "regenerate the figure.",
                            )
                        ]

            if not (
                node.func.attr in {"plot", "errorbar"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in effect_x_axes
                and len(node.args) >= 2
            ):
                continue

            def is_zero_position(expression: ast.AST) -> bool:
                if (
                    isinstance(expression, (ast.List, ast.Tuple))
                    and len(expression.elts) == 1
                ):
                    expression = expression.elts[0]
                return (
                    isinstance(expression, ast.Constant)
                    and isinstance(expression.value, (int, float))
                    and not isinstance(expression.value, bool)
                    and float(expression.value) == 0.0
                )

            def contains_effect_estimate(expression: ast.AST) -> bool:
                return any(
                    any(
                        term in child.id.casefold()
                        for term in (
                            "mean_diff",
                            "effect_estimate",
                            "contrast_estimate",
                            "hedges_g",
                        )
                    )
                    for child in ast.walk(expression)
                    if isinstance(child, ast.Name)
                )

            xerr_expression = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "xerr"),
                None,
            )
            yerr_expression = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "yerr"),
                None,
            )
            y_names = {
                child.id
                for child in ast.walk(node.args[1])
                if isinstance(child, ast.Name)
            }
            xerr_names = (
                {
                    child.id
                    for child in ast.walk(xerr_expression)
                    if isinstance(child, ast.Name)
                }
                if xerr_expression is not None
                else set()
            )
            transposed_interval = bool(y_names & xerr_names)
            if is_zero_position(node.args[0]) and (
                contains_effect_estimate(node.args[1])
                or transposed_interval
                or yerr_expression is not None
            ):
                return [
                    (
                        "figure_effect_axis_transposed",
                        "The plotting source labels the x-axis as an effect estimate "
                        "but places the estimate on the y coordinate and zero on x. "
                        "Place the estimate and its interval on the labeled x scale, "
                        "use a constant categorical y position, and regenerate the "
                        "figure.",
                    )
                ]
    return []


def validate_report(
    report: ScientificReport,
    retrieval: RetrievalEvidence | None = None,
    computation: ComputationEvidence | None = None,
    required_languages: tuple[str, ...] = (),
    require_reconciliation: bool = False,
    require_pubmed_literature: bool = False,
    require_inline_citations: bool = False,
    required_output_extensions: tuple[str, ...] = (),
    required_display_kinds: tuple[str, ...] = (),
    controller_artifacts: tuple[ArtifactRef, ...] = (),
    controller_dates: tuple[str, ...] = (),
) -> DeterministicValidation:
    findings: list[LintFinding] = []
    valid_reconciliation_paths: set[str] = set()
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
    generated_artifacts = [
        Path(artifact.path)
        for artifact in (computation.artifacts if computation else [])
        if artifact.description == "sandbox-generated analysis artifact"
    ]
    present_display_kinds = {display.kind for display in report.displays}
    for required_kind in required_display_kinds:
        if required_kind not in present_display_kinds:
            findings.append(
                LintFinding(
                    code="required_report_display_missing",
                    location="displays",
                    message=(
                        f"The locked plan requires a reader-facing {required_kind}, "
                        "but the report does not register one. Generate the exact "
                        "artifact and embed it with a caption and evidence links."
                    ),
                )
            )
    for required_suffix in required_output_extensions:
        candidates = [
            path
            for path in generated_artifacts
            if path.suffix.casefold() == required_suffix.casefold()
            and path.is_file()
            and not path.is_symlink()
        ]
        if not candidates:
            findings.append(
                LintFinding(
                    code="requested_output_artifact_missing",
                    location="computation_evidence",
                    message=(
                        f"The user requested a {required_suffix} output, but no "
                        "successful sandbox artifact of that type was produced."
                    ),
                )
            )
            continue
        if required_suffix.casefold() == ".pptx":
            try:
                with zipfile.ZipFile(candidates[-1]) as archive:
                    names = set(archive.namelist())
                required_members = {"[Content_Types].xml", "ppt/presentation.xml"}
                if not required_members.issubset(names):
                    raise ValueError("required presentation members are absent")
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                findings.append(
                    LintFinding(
                        code="requested_pptx_invalid",
                        location=str(candidates[-1]),
                        message=f"The requested PPTX is not structurally valid: {exc}",
                    )
                )
            visual_previews = [
                path
                for path in generated_artifacts
                if "visual-review" in path.parts
                and path.suffix.casefold() in FIGURE_MEDIA_TYPES
                and path.is_file()
                and not path.is_symlink()
            ]
            if not visual_previews:
                findings.append(
                    LintFinding(
                        code="requested_pptx_preview_missing",
                        location=str(candidates[-1]),
                        message=(
                            "A requested PPTX requires at least one deterministic "
                            "slide preview below output/visual-review for Gemma-only "
                            "visual inspection."
                        ),
                    )
                )
        elif required_suffix.casefold() == ".zip":
            try:
                with zipfile.ZipFile(candidates[-1]) as archive:
                    if not any(not item.is_dir() for item in archive.infolist()):
                        raise ValueError("the archive contains no files")
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                findings.append(
                    LintFinding(
                        code="requested_zip_invalid",
                        location=str(candidates[-1]),
                        message=f"The requested result ZIP is invalid: {exc}",
                    )
                )
        elif required_suffix.casefold() == ".ipynb":
            try:
                notebook = (
                    json.loads(candidates[-1].read_text(encoding="utf-8"))
                    if candidates[-1].stat().st_size <= 64 * 1024 * 1024
                    else None
                )
            except (OSError, UnicodeError, ValueError):
                notebook = None
            if (
                notebook is None
                or not isinstance(notebook.get("cells"), list)
                or not isinstance(notebook.get("nbformat"), int)
            ):
                findings.append(
                    LintFinding(
                        code="requested_notebook_invalid",
                        location=str(candidates[-1]),
                        message=(
                            "The requested notebook must be strict JSON with cells "
                            "and an integer nbformat."
                        ),
                    )
                )
    authoritative_json = _authoritative_json_outputs(records)
    authoritative_figures: dict[str, str] = {}
    for artifact in computation.artifacts if computation else []:
        path = Path(artifact.path)
        key = logical_report_output_key(path)
        if (
            key is not None
            and key.startswith("figures/")
            and path.suffix.casefold() in FIGURE_MEDIA_TYPES
        ):
            authoritative_figures[key] = os.path.normpath(str(path))
    for record in records:
        if record.status != "succeeded":
            continue
        if any(
            (key := logical_report_output_key(Path(artifact.path))) is not None
            and authoritative_figures.get(key) == os.path.normpath(artifact.path)
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
            artifact_findings: list[LintFinding] = []
            try:
                generated_document = json.loads(
                    path.read_text(encoding="utf-8"),
                    parse_constant=_reject_nonfinite_json,
                )
            except (OSError, UnicodeError, ValueError) as exc:
                artifact_findings.append(
                    LintFinding(
                        code="invalid_generated_json",
                        location=str(path),
                        message=(
                            "Generated JSON must be strict UTF-8 JSON and may not "
                            f"contain NaN or Infinity: {type(exc).__name__}."
                        ),
                    )
                )
            else:
                artifact_findings.extend(
                    _inferential_consistency_findings(path, generated_document)
                )
            is_authoritative = authoritative_json.get(
                (record.language, _logical_json_output_key(path))
            ) == os.path.normpath(str(path))
            if is_authoritative:
                findings.extend(artifact_findings)
            else:
                findings.extend(
                    finding.model_copy(
                        update={
                            "code": f"superseded_{finding.code}",
                            "message": (
                                "A later successful artifact superseded this finding; "
                                f"historical issue retained for audit: {finding.message}"
                            ),
                            "blocking": False,
                        }
                    )
                    for finding in artifact_findings
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
                if path.suffix.lower() != ".json":
                    continue
                verdict = reconciliation_verdict(path, computation)
                if verdict is None:
                    continue
                verdicts.append(verdict)
                if verdict:
                    valid_reconciliation_paths.add(os.path.normpath(str(path)))
            if not verdicts:
                findings.append(
                    LintFinding(
                        code="reconciliation_artifact_invalid",
                        location="computation_evidence",
                        message=(
                            "The reconciliation JSON must bind each Python/R value "
                            "to a successful hashed JSON artifact and JSON path, then "
                            "declare a tolerance, absolute difference, per-comparison "
                            "pass, and consistent top-level verdict."
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
    if require_reconciliation and valid_reconciliation_paths:
        for index, claim in enumerate(report.claims):
            if not re.search(
                r"\b(?:cross[- ]language|independent\s+r\b.{0,80}\b(?:reproduc|match)|absolute\s+difference)",
                claim.text,
                re.IGNORECASE,
            ):
                continue
            cited_paths = {
                os.path.normpath(source.artifact_path)
                for source_id in claim.evidence_refs
                if (source := sources_by_id.get(source_id)) is not None
                and source.artifact_path is not None
            }
            if cited_paths.isdisjoint(valid_reconciliation_paths):
                findings.append(
                    LintFinding(
                        code="cross_language_claim_missing_reconciliation_source",
                        location=f"claims[{index}]",
                        message=(
                            "A cross-language agreement claim must cite the successful "
                            "reconciliation JSON, not only one language's output."
                        ),
                    )
                )
    controller_paths = {
        os.path.normpath(artifact.path) for artifact in controller_artifacts
    }
    methods_text = " ".join(report.methods)
    report_text = " ".join(
        (
            report.executive_summary,
            report.introduction,
            methods_text,
            report.results,
            report.discussion,
            report.conclusions,
            *(claim.text for claim in report.claims),
        )
    )
    report_text = " ".join(report_text.split())
    machine_numbers = _machine_json_numbers(computation)
    findings.extend(
        _reported_degrees_of_freedom_findings(report.results, machine_numbers)
    )
    if _OVERSTATED_SENSITIVITY_LANGUAGE.search(report_text):
        findings.append(
            LintFinding(
                code="sensitivity_analysis_overclaim",
                location="report",
                message=(
                    "Similarity between a primary and sensitivity estimate does not "
                    "prove absence of confounding, algorithmic equivalence, pipeline "
                    "validity, robustness, or stability. Report the observed numerical "
                    "agreement and its scope directly."
                ),
            )
        )
    if _ASSUMPTION_ACCEPTANCE_LANGUAGE.search(report_text):
        findings.append(
            LintFinding(
                code="diagnostic_nonrejection_overclaim",
                location="report",
                message=(
                    "A nonsignificant Shapiro-Wilk or Levene test does not establish "
                    "that an assumption is met. State that the diagnostic did not "
                    "detect a departure, report its limited power, and retain the "
                    "relevant assumption as a limitation."
                ),
            )
        )
    if _WELCH_NORMALITY_OVERCLAIM.search(report_text):
        findings.append(
            LintFinding(
                code="welch_normality_overclaim",
                location="report",
                message=(
                    "Welch's adjustment addresses unequal variances, not departure "
                    "from normality. Remove claims that Welch itself accommodates "
                    "non-normality; retain the diagnostic as a limitation or support "
                    "a separately scoped robustness analysis with direct evidence."
                ),
            )
        )
    if re.search(
        r"\bbaseline\b.{0,180}\b(?:differ|imbalance|p\s*[=<]).{0,180}"
        r"\bjustif(?:y|ies|ied|ying)\b.{0,80}\bprespecif",
        report_text,
        re.IGNORECASE,
    ):
        findings.append(
            LintFinding(
                code="posthoc_result_cannot_justify_prespecification",
                location="report",
                message=(
                    "An observed baseline result cannot justify that an analysis "
                    "was prespecified. Cite the task/protocol for timing and describe "
                    "the baseline result only as interpretation context."
                ),
            )
        )
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
    if _PROTOCOL_BEFORE_DATA_INSPECTION.search(report_text):
        findings.append(
            LintFinding(
                code="protocol_timing_overstates_input_blinding",
                location="report",
                message=(
                    "Evidence Bench profiles inputs before planning and protocol lock. "
                    "Do not claim the protocol preceded data inspection; state only "
                    "the controller-supported timing, such as before outcome analysis "
                    "or result-producing execution."
                ),
            )
        )
    if _AI_ROLE_UNDERSTATEMENT.search(report_text):
        findings.append(
            LintFinding(
                code="ai_role_understated",
                location="methods",
                message=(
                    "Do not say AI was used only for drafting or artifact registration. "
                    "Qwen also supported planning, code generation, and analysis, while "
                    "Gemma performed independent review and deterministic software "
                    "executed and validated the work."
                ),
            )
        )
    assumption_text = " ".join((report_text, *report.limitations))
    if _BALANCED_DESIGN_REASSURANCE.search(assumption_text):
        findings.append(
            LintFinding(
                code="balanced_design_assumption_reassurance",
                location="report",
                message=(
                    "Balance alone does not establish protection from non-normality or "
                    "Type I error inflation. Remove the reassurance, retain the observed "
                    "diagnostic as a limitation, or cite and apply a directly relevant "
                    "methodological robustness analysis."
                ),
            )
        )
    if _UNQUALIFIED_RESULT_ROBUSTNESS.search(report_text):
        findings.append(
            LintFinding(
                code="unqualified_result_robustness",
                location="report",
                message=(
                    "Do not summarize a result as robust. State the exact observed "
                    "reproducibility or sensitivity evidence and its scope instead."
                ),
            )
        )
    if _GROUP_MEAN_UNIFORMITY_OVERCLAIM.search(report_text):
        findings.append(
            LintFinding(
                code="group_mean_uniformity_overclaim",
                location="report",
                message=(
                    "A group mean does not establish a uniform individual response. "
                    "Describe the group mean and observed dispersion directly; use "
                    "individual-level language only when every corresponding value "
                    "is explicitly verified by a successful artifact."
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
    knowledge_passages_by_url: dict[str, tuple[Any, str]] = {}
    knowledge_passages_by_document_path: dict[str, tuple[Any, str]] = {}
    knowledge_visuals_by_url: dict[str, Any] = {}
    knowledge_snapshot_documents: dict[str, tuple[str, str]] = {}
    knowledge_snapshot_metadata: dict[str, dict[str, Any]] = {}
    snapshot_record = next(
        (
            artifact
            for artifact in controller_artifacts
            if Path(artifact.path).name == "knowledge_snapshot.json"
        ),
        None,
    )
    snapshot_artifact = (
        Path(snapshot_record.path) if snapshot_record is not None else None
    )
    if snapshot_artifact is not None:
        try:
            if (
                snapshot_artifact.is_symlink()
                or not snapshot_artifact.is_file()
                or hashlib.sha256(snapshot_artifact.read_bytes()).hexdigest()
                != snapshot_record.sha256
            ):
                raise ValueError("snapshot artifact hash mismatch")
            snapshot = json.loads(snapshot_artifact.read_text(encoding="utf-8"))
            expected = snapshot.pop("snapshot_sha256")
            observed = hashlib.sha256(
                json.dumps(
                    snapshot, sort_keys=True, separators=(",", ":"), default=str
                ).encode()
            ).hexdigest()
            if observed != expected or (
                retrieval
                and retrieval.knowledge_snapshot_sha256
                and retrieval.knowledge_snapshot_sha256 != expected
            ):
                raise ValueError("snapshot hash mismatch")
            knowledge_snapshot_documents = {
                str(item["document_id"]): (
                    str(item["original_sha256"]),
                    str(item["content_sha256"]),
                )
                for item in snapshot.get("documents", [])
            }
            knowledge_snapshot_metadata = {
                str(item["document_id"]): item for item in snapshot.get("documents", [])
            }
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            findings.append(
                LintFinding(
                    code="knowledge_snapshot_invalid",
                    location="knowledge_snapshot.json",
                    message="The controller knowledge snapshot failed integrity validation.",
                )
            )
    for passage in retrieval.knowledge_passages if retrieval else []:
        location = f"knowledge_passages[{passage.passage_id}]"
        path = Path(passage.artifact_path)
        normalized = os.path.normpath(passage.artifact_path)
        try:
            expected_document = knowledge_snapshot_documents.get(passage.document_id)
            if expected_document is None:
                raise ValueError("passage document is absent from the snapshot")
            expected_metadata = knowledge_snapshot_metadata.get(passage.document_id)
            if expected_metadata is None:
                raise ValueError(
                    "passage document metadata is absent from the snapshot"
                )
            if snapshot_artifact is None:
                raise ValueError("knowledge snapshot artifact is absent")
            run_root = snapshot_artifact.parent.resolve()

            def verified_run_file(candidate: Path, expected_sha256: str) -> bytes:
                resolved = candidate.resolve()
                if (
                    candidate.is_symlink()
                    or not candidate.is_file()
                    or (resolved != run_root and run_root not in resolved.parents)
                    or os.path.normpath(str(candidate)) not in retrieval_artifacts
                ):
                    raise ValueError("knowledge artifact escaped the run directory")
                content = candidate.read_bytes()
                if hashlib.sha256(content).hexdigest() != expected_sha256:
                    raise ValueError("knowledge artifact hash mismatch")
                return content

            artifact_text = path.read_text(encoding="utf-8")
            if not artifact_text.startswith("---\n"):
                raise ValueError("knowledge passage frontmatter is absent")
            header, separator, body = artifact_text[4:].partition("\n---\n\n")
            if not separator:
                raise ValueError("knowledge passage frontmatter is malformed")
            artifact_record: dict[str, Any] = {}
            for line in header.splitlines():
                key, delimiter, encoded_value = line.partition(": ")
                if not delimiter or key in artifact_record:
                    raise ValueError("knowledge passage frontmatter is malformed")
                artifact_record[key] = json.loads(encoded_value)
            marker = "# Exact untrusted source passage\n\n"
            if not body.startswith(marker):
                raise ValueError("knowledge passage body is malformed")
            source_text = body[len(marker) :]
            if source_text.endswith("\n"):
                source_text = source_text[:-1]
            artifact_bytes = verified_run_file(path, passage.artifact_sha256)
            document_bytes = verified_run_file(
                Path(passage.document_text_path), passage.document_text_sha256
            )
            verified_run_file(
                Path(passage.document_original_path), passage.document_original_sha256
            )
            document_text = document_bytes.decode("utf-8")
            exact_source_text = document_text[passage.char_start : passage.char_end]
            exact_chunk_sha256 = hashlib.sha256(
                exact_source_text.encode("utf-8")
            ).hexdigest()
            expected_chunk_id = (
                "kc-"
                + hashlib.sha256(
                    (
                        f"{passage.document_id}:{passage.chunk_ordinal}:"
                        f"{passage.char_start}:{passage.char_end}:"
                        f"{exact_chunk_sha256}"
                    ).encode()
                ).hexdigest()[:24]
            )
            expected_passage_id = (
                "kp-"
                + hashlib.sha256(
                    (
                        f"{retrieval.knowledge_snapshot_sha256}:"
                        f"{passage.document_id}:{expected_chunk_id}:"
                        f"{exact_chunk_sha256}"
                    ).encode()
                ).hexdigest()[:24]
            )
            valid = bool(
                artifact_bytes
                and passage.content_sha256 == expected_document[1]
                and passage.document_text_sha256 == expected_document[1]
                and passage.document_original_sha256 == expected_document[0]
                and (
                    passage.canonical_url is None
                    or passage.canonical_url == expected_metadata.get("canonical_url")
                )
                and (
                    passage.source_type is None
                    or passage.source_type == expected_metadata.get("source_type")
                )
                and passage.title == expected_metadata.get("title")
                and passage.char_end <= len(document_text)
                and source_text == exact_source_text
                and passage.chunk_sha256 == exact_chunk_sha256
                and passage.chunk_id == expected_chunk_id
                and passage.passage_id == expected_passage_id
                and artifact_record.get("passage_id") == passage.passage_id
                and artifact_record.get("document_id") == passage.document_id
                and artifact_record.get("chunk_id") == passage.chunk_id
                and artifact_record.get("chunk_ordinal") == passage.chunk_ordinal
                and artifact_record.get("char_start") == passage.char_start
                and artifact_record.get("char_end") == passage.char_end
                and artifact_record.get("chunk_sha256") == passage.chunk_sha256
                and artifact_record.get("source_url") == passage.source_url
                and _normalize_url(passage.source_url).endswith(
                    f"/{passage.passage_id}"
                )
                and len(source_text) == passage.char_end - passage.char_start
                and normalized in retrieval_artifacts
            )
        except (OSError, UnicodeError, IndexError, TypeError, ValueError):
            valid = False
            source_text = ""
        if not valid:
            findings.append(
                LintFinding(
                    code="knowledge_passage_integrity_failed",
                    location=location,
                    message=(
                        "A knowledge citation must resolve to exact hash-checked "
                        "passage bytes from this run's immutable snapshot."
                    ),
                )
            )
            continue
        normalized_url = _normalize_url(passage.source_url)
        if normalized_url in knowledge_passages_by_url:
            findings.append(
                LintFinding(
                    code="duplicate_knowledge_passage_url",
                    location=location,
                    message="Knowledge passage URLs must identify exactly one passage.",
                )
            )
            continue
        knowledge_passages_by_url[normalized_url] = (passage, source_text)
        document_path = os.path.normpath(passage.document_text_path)
        existing_document = knowledge_passages_by_document_path.get(document_path)
        if existing_document is None:
            knowledge_passages_by_document_path[document_path] = (passage, source_text)
        elif existing_document[0].document_id == passage.document_id:
            knowledge_passages_by_document_path[document_path] = (
                existing_document[0],
                existing_document[1] + "\n" + source_text,
            )
    for visual in retrieval.knowledge_visuals if retrieval else []:
        location = f"knowledge_visuals[{visual.knowledge_visual_id}]"
        path = Path(visual.artifact_path)
        normalized = os.path.normpath(visual.artifact_path)
        try:
            expected_document = knowledge_snapshot_documents.get(visual.document_id)
            if expected_document is None:
                raise ValueError("visual document is absent from the snapshot")
            if snapshot_artifact is None:
                raise ValueError("knowledge snapshot artifact is absent")
            run_root = snapshot_artifact.parent.resolve()
            resolved = path.resolve()
            expected_visual_id = (
                "kvp-"
                + hashlib.sha256(
                    (
                        f"{retrieval.knowledge_snapshot_sha256}:"
                        f"{visual.document_id}:{visual.visual_id}:"
                        f"{visual.visual_sha256}"
                    ).encode()
                ).hexdigest()[:24]
            )
            valid = bool(
                path.is_absolute()
                and not path.is_symlink()
                and path.is_file()
                and run_root in resolved.parents
                and resolved.parent == (run_root / "knowledge" / "visuals").resolve()
                and path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}
                and normalized in retrieval_artifacts
                and hashlib.sha256(path.read_bytes()).hexdigest()
                == visual.artifact_sha256
                and visual.artifact_sha256 == visual.visual_sha256
                and visual.document_original_sha256 == expected_document[0]
                and visual.snapshot_sha256 == retrieval.knowledge_snapshot_sha256
                and visual.knowledge_visual_id == expected_visual_id
                and resolved.name.startswith(f"{visual.knowledge_visual_id}.")
                and _normalize_url(visual.source_url).endswith(
                    f"/visuals/{visual.knowledge_visual_id}"
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            valid = False
        if not valid:
            findings.append(
                LintFinding(
                    code="knowledge_visual_integrity_failed",
                    location=location,
                    message=(
                        "A knowledge visual citation must resolve to exact "
                        "hash-checked raster bytes copied into this run's immutable "
                        "snapshot provenance."
                    ),
                )
            )
            continue
        normalized_url = _normalize_url(visual.source_url)
        if normalized_url in knowledge_visuals_by_url:
            findings.append(
                LintFinding(
                    code="duplicate_knowledge_visual_url",
                    location=location,
                    message="Knowledge visual URLs must identify exactly one raster.",
                )
            )
            continue
        knowledge_visuals_by_url[normalized_url] = visual
    acquired_text_by_source: dict[str, str] = {}
    for index, source in enumerate(report.sources):
        knowledge_match = (
            knowledge_passages_by_url.get(_normalize_url(str(source.url)))
            if source.url is not None
            else None
        )
        if (
            knowledge_match is None
            and source.url is not None
            and source.local_markdown_path is not None
        ):
            document_match = knowledge_passages_by_document_path.get(
                os.path.normpath(source.local_markdown_path)
            )
            if document_match is not None:
                document_metadata = knowledge_snapshot_metadata.get(
                    document_match[0].document_id, {}
                )
                canonical_url = document_match[
                    0
                ].canonical_url or document_metadata.get("canonical_url")
                if canonical_url and _normalize_url(str(source.url)) == _normalize_url(
                    str(canonical_url)
                ):
                    knowledge_match = document_match
        if knowledge_match is not None:
            document_metadata = knowledge_snapshot_metadata.get(
                knowledge_match[0].document_id, {}
            )
            if source.title != document_metadata.get("title"):
                findings.append(
                    LintFinding(
                        code="knowledge_source_title_mismatch",
                        location=f"sources[{index}].title",
                        message=(
                            "Knowledge SourceRecord title must match the immutable "
                            "document generation."
                        ),
                    )
                )
            if source.source_type != document_metadata.get("source_type"):
                findings.append(
                    LintFinding(
                        code="knowledge_source_type_mismatch",
                        location=f"sources[{index}].source_type",
                        message=(
                            "Knowledge SourceRecord type must match the immutable "
                            "document generation."
                        ),
                    )
                )
        knowledge_visual_match = (
            knowledge_visuals_by_url.get(_normalize_url(str(source.url)))
            if source.url is not None
            else None
        )
        if knowledge_match is not None:
            acquired_text_by_source[source.source_id] = knowledge_match[1].casefold()
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
            and knowledge_match is None
            and knowledge_visual_match is None
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
        if (
            source.pmid
            and source.local_markdown_path is None
            and knowledge_match is None
            and knowledge_visual_match is None
        ):
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
        if (
            source.pmid
            and source.full_text_status is None
            and knowledge_match is None
            and knowledge_visual_match is None
        ):
            findings.append(
                LintFinding(
                    code="pubmed_acquisition_status_missing",
                    location=f"sources[{index}].full_text_status",
                    message="A cited PubMed record requires an explicit acquisition status.",
                )
            )
        if source.pmid and knowledge_match is None and knowledge_visual_match is None:
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
        if knowledge_match is not None:
            _, source_text = knowledge_match
            passage_terms = _grounding_terms(source.supporting_passage)
            if passage_terms and not any(
                term in source_text.casefold() for term in passage_terms
            ):
                findings.append(
                    LintFinding(
                        code="knowledge_supporting_passage_not_grounded",
                        location=f"sources[{index}].supporting_passage",
                        message=(
                            "The stated supporting passage has no informative "
                            "term overlap with the exact snapshotted knowledge text."
                        ),
                    )
                )
            precise = _PRECISE_LITERATURE_NUMBER.findall(source.supporting_passage)
            missing_numbers = [
                item
                for item in precise
                if item.replace(" ", "") not in source_text.replace(" ", "")
            ]
            if missing_numbers:
                findings.append(
                    LintFinding(
                        code="knowledge_supporting_number_not_grounded",
                        location=f"sources[{index}].supporting_passage",
                        message=(
                            "Precise values attributed to a knowledge passage "
                            "must occur in its exact snapshotted bytes: "
                            + ", ".join(missing_numbers)
                        ),
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

    citation_ids = [citation.citation_id for citation in report.inline_citations]
    if len(citation_ids) != len(set(citation_ids)):
        findings.append(
            LintFinding(
                code="duplicate_inline_citation_id",
                location="inline_citations",
                message="Inline citation IDs must be unique.",
            )
        )
    citation_anchors = [
        (citation.section, citation.anchor_text) for citation in report.inline_citations
    ]
    if len(citation_anchors) != len(set(citation_anchors)):
        findings.append(
            LintFinding(
                code="duplicate_inline_citation_anchor",
                location="inline_citations",
                message=(
                    "Use one InlineCitation per anchored sentence and place all "
                    "direct sources in that record."
                ),
            )
        )
    article_sections = {
        "executive_summary": report.executive_summary,
        "introduction": report.introduction,
        "methods": "\n".join(report.methods),
        "results": report.results,
        "discussion": report.discussion,
        "conclusions": report.conclusions,
    }
    for left_index, left in enumerate(report.inline_citations):
        section_text = article_sections[left.section]
        left_start = section_text.find(left.anchor_text)
        if left_start < 0:
            continue
        left_end = left_start + len(left.anchor_text)
        for right_index, right in enumerate(
            report.inline_citations[left_index + 1 :], start=left_index + 1
        ):
            if right.section != left.section:
                continue
            right_start = section_text.find(right.anchor_text)
            if right_start < 0:
                continue
            right_end = right_start + len(right.anchor_text)
            if max(left_start, right_start) < min(left_end, right_end):
                findings.append(
                    LintFinding(
                        code="overlapping_inline_citation_anchors",
                        location=(
                            f"inline_citations[{left_index}],"
                            f"inline_citations[{right_index}]"
                        ),
                        message=(
                            "Inline citation anchors in one article section must "
                            "not overlap; use one combined citation or separate "
                            "non-overlapping exact anchors."
                        ),
                    )
                )
    citations_by_claim: dict[str, set[str]] = {}
    for index, citation in enumerate(report.inline_citations):
        location = f"inline_citations[{index}]"
        occurrences = article_sections[citation.section].count(citation.anchor_text)
        if occurrences != 1:
            findings.append(
                LintFinding(
                    code="inline_citation_anchor_not_unique",
                    location=f"{location}.anchor_text",
                    message=(
                        "The citation anchor must occur exactly once in its declared "
                        f"article section; found {occurrences}."
                    ),
                )
            )
        unknown_sources = sorted(set(citation.source_ids) - known)
        if unknown_sources:
            findings.append(
                LintFinding(
                    code="inline_citation_unknown_source",
                    location=f"{location}.source_ids",
                    message=(
                        "Inline citation references unknown sources: "
                        + ", ".join(unknown_sources)
                    ),
                )
            )
        nonliterature_sources = sorted(
            source_id
            for source_id in citation.source_ids
            if source_id in sources_by_id and sources_by_id[source_id].url is None
        )
        if nonliterature_sources:
            findings.append(
                LintFinding(
                    code="inline_citation_not_literature",
                    location=f"{location}.source_ids",
                    message=(
                        "Article-style inline citations must resolve to knowledge "
                        "or retrieved literature URLs, not computation artifacts: "
                        + ", ".join(nonliterature_sources)
                    ),
                )
            )
        unknown_claims = sorted(set(citation.claim_ids) - set(claim_ids))
        if unknown_claims:
            findings.append(
                LintFinding(
                    code="inline_citation_unknown_claim",
                    location=f"{location}.claim_ids",
                    message=(
                        "Inline citation references unknown claims: "
                        + ", ".join(unknown_claims)
                    ),
                )
            )
        for claim_id in citation.claim_ids:
            citations_by_claim.setdefault(claim_id, set()).update(citation.source_ids)
            claim = next(
                (item for item in report.claims if item.claim_id == claim_id), None
            )
            if claim is not None and not set(citation.source_ids).issubset(
                claim.evidence_refs
            ):
                findings.append(
                    LintFinding(
                        code="inline_citation_claim_source_mismatch",
                        location=location,
                        message=(
                            f"Citation sources for {claim_id} must be direct "
                            "evidence_refs on that claim."
                        ),
                    )
                )
            if claim is not None:
                anchor_terms = _citation_correspondence_terms(citation.anchor_text)
                claim_terms = _citation_correspondence_terms(claim.text)
                anchor_numbers = _citation_correspondence_numbers(citation.anchor_text)
                claim_numbers = _citation_correspondence_numbers(claim.text)
                if not (anchor_terms & claim_terms or anchor_numbers & claim_numbers):
                    findings.append(
                        LintFinding(
                            code="inline_citation_claim_anchor_mismatch",
                            location=location,
                            message=(
                                f"Citation anchor has no informative lexical or "
                                f"numerical correspondence with linked claim "
                                f"{claim_id}. Place the citation on claim-bearing "
                                "text or correct the claim link; Gemma remains "
                                "responsible for semantic entailment review."
                            ),
                        )
                    )
    if require_inline_citations:
        for index, claim in enumerate(report.claims):
            external_refs = {
                source_id
                for source_id in claim.evidence_refs
                if source_id in sources_by_id
                and sources_by_id[source_id].url is not None
            }
            missing_inline = external_refs - citations_by_claim.get(
                claim.claim_id, set()
            )
            if external_refs and missing_inline:
                findings.append(
                    LintFinding(
                        code="literature_claim_missing_inline_citation",
                        location=f"claims[{index}]",
                        message=(
                            "Every knowledge- or literature-backed claim must be "
                            "cited in the article body. Missing inline sources: "
                            + ", ".join(sorted(missing_inline))
                        ),
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
    # Computation evidence is append-only. A repair may generate a corrected
    # display at the same logical /output path in a later attempt. Keep every
    # historical version in provenance, but require registration only for the
    # latest valid display candidate at each logical key. An unrelated newer file
    # cannot make an older figure or table disappear.
    final_outputs_by_key: dict[str, Path] = {}
    for artifact in computation.artifacts if computation else []:
        path = Path(artifact.path)
        key = logical_report_output_key(path)
        if key is None:
            continue
        output_kind = key.split("/", 1)[0]
        suffix = path.suffix.casefold()
        is_display_candidate = (
            output_kind == "figures" and suffix in FIGURE_MEDIA_TYPES
        ) or (output_kind == "tables" and suffix in TABLE_DELIMITERS)
        if not is_display_candidate:
            findings.append(
                LintFinding(
                    code="non_display_artifact_in_reader_facing_folder",
                    location=str(path),
                    message=(
                        "This artifact cannot be embedded as a report display. "
                        "Keep full-precision JSON and diagnostics below /output/data "
                        "or /output/validation, and create a PNG/JPEG/WebP figure or "
                        "rounded CSV/TSV table when a reader-facing display is needed."
                    ),
                    blocking=False,
                )
            )
            continue
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
            if logical_report_output_key(artifact_path) is None:
                findings.append(
                    LintFinding(
                        code="display_not_reader_facing_output",
                        location=location,
                        message=(
                            "A ReportDisplay must be a deliberate artifact within "
                            "/output/figures or /output/tables; uploaded, "
                            "extracted, and visual-review evidence remains a source "
                            "artifact rather than an inline display."
                        ),
                    )
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
                annotation_overlap = figure_annotation_overlap_candidates(
                    artifact_path,
                    figure_ocr,
                    width=int(figure_metadata["width"]),
                    height=int(figure_metadata["height"]),
                )
                if annotation_overlap:
                    examples = "; ".join(
                        (
                            f"{item['text']!r} box is "
                            f"{item['height_vs_median']:.2f}x the median text height "
                            f"with {item['chromatic_pixel_fraction']:.1%} chromatic pixels"
                        )
                        for item in annotation_overlap[:3]
                    )
                    findings.append(
                        LintFinding(
                            code="figure_annotation_data_overlap",
                            location=location,
                            message=(
                                "Rendered OCR/raster geometry indicates that a colored "
                                "data mark or interval crosses annotation text. Move the "
                                "annotation away from the plotted geometry and regenerate "
                                f"the raster. Candidates: {examples}"
                            ),
                        )
                    )
                for code, message in _figure_ocr_semantic_findings(
                    figure_ocr, machine_numbers
                ):
                    findings.append(
                        LintFinding(code=code, location=location, message=message)
                    )
                for code, message in _figure_caption_semantic_findings(
                    display.caption, display.alt_text, figure_ocr
                ):
                    findings.append(
                        LintFinding(code=code, location=location, message=message)
                    )
                for code, message in _figure_source_semantic_findings(
                    artifact_path, computation
                ):
                    findings.append(
                        LintFinding(code=code, location=location, message=message)
                    )
            else:
                table_preview = read_table_preview(artifact_path)
                ambiguous_rows = _ambiguous_group_table_rows(table_preview)
                if ambiguous_rows:
                    findings.append(
                        LintFinding(
                            code="table_ambiguous_overall_estimate_column",
                            location=location,
                            message=(
                                "Overall estimates must not appear beneath one group "
                                "header. Add a neutral Estimate/Overall column or use "
                                "a separate estimand table. Examples: "
                                + "; ".join(ambiguous_rows)
                            ),
                        )
                    )
                precision_examples = excessive_table_precision(table_preview)
                if precision_examples:
                    findings.append(
                        LintFinding(
                            code="table_excessive_precision",
                            location=location,
                            message=(
                                "Reader-facing tables must use scientific display "
                                "precision of at most four significant digits (not "
                                "four decimal places); preserve full precision in "
                                "JSON. For example, use 10.9 or 10.90 instead of "
                                "10.897, 0.9801 instead of 0.980132, and 5.000 "
                                "instead of 5.0000. Offending cells: "
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
                    source.artifact_path
                    or source.local_markdown_path
                    or (
                        source.url is not None
                        and _normalize_url(str(source.url)) in knowledge_passages_by_url
                    )
                    or (
                        source.url is not None
                        and _normalize_url(str(source.url)) in knowledge_visuals_by_url
                    )
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
                source.artifact_path
                or source.local_markdown_path
                or (
                    source.url is not None
                    and _normalize_url(str(source.url)) in knowledge_passages_by_url
                )
                or (
                    source.url is not None
                    and _normalize_url(str(source.url)) in knowledge_visuals_by_url
                )
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
                if re.search(r"\bguidelines?\b", claim.text, re.IGNORECASE) and not any(
                    source.source_type == "guideline" for source in referenced_sources
                ):
                    findings.append(
                        LintFinding(
                            code="literature_source_type_mismatch",
                            location=location,
                            message=(
                                "A claim attributed to reporting guidelines must cite "
                                "a source classified and acquired as a guideline; do "
                                "not relabel an unrelated primary study or review."
                            ),
                        )
                    )
                claim_terms = _grounding_terms(claim.text)
                for source in referenced_sources:
                    acquired_text = acquired_text_by_source.get(source.source_id)
                    if (
                        acquired_text is not None
                        and claim_terms
                        and not any(term in acquired_text for term in claim_terms)
                    ):
                        findings.append(
                            LintFinding(
                                code="literature_claim_not_lexically_grounded",
                                location=location,
                                message=(
                                    f"Claim {claim.claim_id} has no informative term "
                                    f"overlap with exact locally acquired source "
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
            if claim.claim_type == "computed":
                artifact_texts: list[str] = []
                for source in referenced_sources:
                    if source.artifact_path is None:
                        continue
                    path = Path(source.artifact_path)
                    try:
                        if (
                            path.suffix.lower() in {".json", ".csv", ".tsv", ".txt"}
                            and path.is_file()
                            and not path.is_symlink()
                            and path.stat().st_size <= RECONCILIATION_JSON_MAX_BYTES
                        ):
                            artifact_texts.append(
                                path.read_text(
                                    encoding="utf-8", errors="replace"
                                ).lower()
                            )
                    except OSError:
                        continue
                missing_diagnostics = sorted(
                    label
                    for label, pattern in _COMPUTED_DIAGNOSTIC_TERMS.items()
                    if pattern.search(claim.text)
                    and artifact_texts
                    and not any(label in text for text in artifact_texts)
                )
                if missing_diagnostics:
                    findings.append(
                        LintFinding(
                            code="computed_diagnostic_not_in_artifact",
                            location=location,
                            message=(
                                "A named diagnostic must occur in a directly cited "
                                "machine-readable computation artifact. Missing: "
                                + ", ".join(missing_diagnostics)
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
