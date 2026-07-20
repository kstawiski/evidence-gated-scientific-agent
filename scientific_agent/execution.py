"""Typed, resource-bounded Python and R execution through bubblewrap."""

from __future__ import annotations

import ast
import json
import os
import re
import signal
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import httpx

from .config import SandboxSettings
from .provenance import sha256_bytes, sha256_file, utc_now, write_json
from .schemas import ArtifactRef, ComputationEvidence, ComputationRecord


Language = Literal["python", "r"]
R_ANALYSIS_BASELINE_PACKAGES = (
    "ggplot2",
    "ragg",
    "systemfonts",
    "svglite",
    "patchwork",
    "cowplot",
    "ggrepel",
    "ggbeeswarm",
    "ggridges",
    "scales",
    "viridisLite",
    "colorspace",
    "pheatmap",
    "ComplexHeatmap",
    "dplyr",
    "tidyr",
    "tibble",
    "purrr",
    "stringr",
    "forcats",
    "lubridate",
    "readr",
    "readxl",
    "openxlsx",
    "survival",
    "survminer",
    "broom",
    "rstatix",
    "emmeans",
    "lme4",
    "pROC",
    "glmnet",
    "cmprsk",
    "survey",
    "mice",
    "data.table",
    "jsonlite",
)
R_ANALYSIS_BASELINE_MINIMUM_VERSIONS = {"patchwork": "1.2.0"}
RETURN_TEXT_BYTES = 32 * 1024
PREVIEW_TEXT_BYTES = 8 * 1024
PREVIEW_SUFFIXES = {".csv", ".json", ".md", ".tsv", ".txt"}
PRIOR_EXECUTION_REFERENCE = re.compile(r"/prior/(?P<execution_id>exec-[0-9]{3})/")


def _reject_nonfinite_json(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _python_static_violations(code: str) -> list[str]:
    """Reject a small set of unambiguously invalid scientific API calls."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    assignment_candidates: dict[str, list[ast.AST]] = {}
    errorbar_container_names: set[str] = set()
    for candidate in ast.walk(tree):
        if not (
            isinstance(candidate, ast.Assign)
            and len(candidate.targets) == 1
            and isinstance(candidate.targets[0], ast.Name)
        ):
            continue
        name = candidate.targets[0].id
        assignment_candidates.setdefault(name, []).append(candidate.value)
        if (
            isinstance(candidate.value, ast.Call)
            and isinstance(candidate.value.func, ast.Attribute)
            and candidate.value.func.attr == "errorbar"
        ):
            errorbar_container_names.add(name)
    single_assignments = {
        name: values[0]
        for name, values in assignment_candidates.items()
        if len(values) == 1
    }

    def numeric_literal(expression: ast.AST | None) -> float | None:
        if isinstance(expression, (ast.List, ast.Tuple)) and len(expression.elts) == 1:
            expression = expression.elts[0]
        if (
            isinstance(expression, ast.Constant)
            and isinstance(expression.value, (int, float))
            and not isinstance(expression.value, bool)
        ):
            return float(expression.value)
        if (
            isinstance(expression, ast.UnaryOp)
            and isinstance(expression.op, ast.USub)
            and isinstance(expression.operand, ast.Constant)
            and isinstance(expression.operand.value, (int, float))
            and not isinstance(expression.operand.value, bool)
        ):
            return -float(expression.operand.value)
        return None

    def is_literal_zero(expression: ast.AST | None) -> bool:
        return numeric_literal(expression) == 0.0

    def resolved_numeric_literal(
        expression: ast.AST,
        before_line: int | None = None,
        seen_names: frozenset[str] = frozenset(),
    ) -> float | None:
        """Resolve exact numeric constants through simple prior assignments."""

        literal = numeric_literal(expression)
        if literal is not None:
            return literal
        if isinstance(expression, ast.Name) and expression.id not in seen_names:
            assignments = [
                candidate
                for candidate in assignment_candidates.get(expression.id, ())
                if before_line is None
                or getattr(candidate, "lineno", before_line) < before_line
            ]
            if assignments:
                latest = max(assignments, key=lambda item: getattr(item, "lineno", -1))
                return resolved_numeric_literal(
                    latest,
                    getattr(latest, "lineno", before_line),
                    seen_names | {expression.id},
                )
        if isinstance(expression, ast.BinOp):
            left = resolved_numeric_literal(expression.left, before_line, seen_names)
            right = resolved_numeric_literal(expression.right, before_line, seen_names)
            if left is None or right is None:
                return None
            if isinstance(expression.op, ast.Add):
                return left + right
            if isinstance(expression.op, ast.Sub):
                return left - right
            if isinstance(expression.op, ast.Mult):
                return left * right
        return None

    def could_be_visible_label(expression: ast.AST | None) -> bool:
        if expression is None:
            return False
        if isinstance(expression, ast.Constant):
            return bool(
                isinstance(expression.value, str)
                and expression.value.strip()
                and not expression.value.lstrip().startswith("_")
            )
        # A computed label cannot be proven empty or Matplotlib-hidden statically.
        return True

    def categorical_position_center(
        expression: ast.AST,
        before_line: int | None = None,
        seen_names: frozenset[str] = frozenset(),
    ) -> float | None:
        literal = numeric_literal(expression)
        if literal is not None:
            return literal
        if isinstance(expression, ast.Name) and expression.id not in seen_names:
            assignments = [
                candidate
                for candidate in assignment_candidates.get(expression.id, ())
                if before_line is None
                or getattr(candidate, "lineno", before_line) < before_line
            ]
            if assignments:
                latest = max(assignments, key=lambda item: getattr(item, "lineno", -1))
                return categorical_position_center(
                    latest,
                    getattr(latest, "lineno", before_line),
                    seen_names | {expression.id},
                )
        if isinstance(expression, ast.BinOp) and isinstance(
            expression.op, (ast.Add, ast.Sub)
        ):
            left = categorical_position_center(expression.left, before_line, seen_names)
            right = categorical_position_center(
                expression.right, before_line, seen_names
            )
            if left is not None and right is not None:
                return (
                    left + right if isinstance(expression.op, ast.Add) else left - right
                )
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "uniform"
            and len(expression.args) >= 2
        ):
            lower = numeric_literal(expression.args[0])
            upper = numeric_literal(expression.args[1])
            if lower is not None and upper is not None:
                return (lower + upper) / 2.0
        return None

    def extremum_items(
        expression: ast.AST, before_line: int | None
    ) -> tuple[ast.AST, ...]:
        """Expand a literal or previously assigned list passed to min()/max()."""

        resolved = expression
        if isinstance(expression, ast.Name):
            assignments = [
                candidate
                for candidate in assignment_candidates.get(expression.id, ())
                if before_line is None
                or getattr(candidate, "lineno", before_line) < before_line
            ]
            if assignments:
                resolved = max(
                    assignments, key=lambda item: getattr(item, "lineno", -1)
                )
        if isinstance(resolved, (ast.List, ast.Tuple)):
            return tuple(resolved.elts)
        return (expression,)

    sign_cache: dict[tuple[int, int | None, frozenset[str]], tuple[bool, bool]] = {}

    def bound_signs(
        expression: ast.AST,
        before_line: int | None = None,
        seen_names: frozenset[str] = frozenset(),
    ) -> tuple[bool, bool]:
        """Return whether an expression is provably nonpositive and nonnegative."""

        cache_key = (id(expression), before_line, seen_names)
        cached = sign_cache.get(cache_key)
        if cached is not None:
            return cached
        literal = numeric_literal(expression)
        if literal is not None:
            result = (literal <= 0.0, literal >= 0.0)
        elif isinstance(expression, ast.Name) and expression.id not in seen_names:
            assignments = [
                candidate
                for candidate in assignment_candidates.get(expression.id, ())
                if before_line is None
                or getattr(candidate, "lineno", before_line) < before_line
            ]
            if assignments:
                latest = max(assignments, key=lambda item: getattr(item, "lineno", -1))
                result = bound_signs(
                    latest,
                    getattr(latest, "lineno", before_line),
                    seen_names | {expression.id},
                )
            else:
                result = (False, False)
        elif isinstance(expression, ast.UnaryOp) and isinstance(
            expression.op, ast.USub
        ):
            nonpositive, nonnegative = bound_signs(
                expression.operand, before_line, seen_names
            )
            result = (nonnegative, nonpositive)
        elif (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id == "abs"
            and len(expression.args) == 1
        ):
            nonpositive, nonnegative = bound_signs(
                expression.args[0], before_line, seen_names
            )
            result = (nonpositive and nonnegative, True)
        elif isinstance(expression, ast.BinOp):
            left_nonpositive, left_nonnegative = bound_signs(
                expression.left, before_line, seen_names
            )
            right_nonpositive, right_nonnegative = bound_signs(
                expression.right, before_line, seen_names
            )
            if isinstance(expression.op, ast.Add):
                result = (
                    left_nonpositive and right_nonpositive,
                    left_nonnegative and right_nonnegative,
                )
            elif isinstance(expression.op, ast.Sub):
                result = (
                    left_nonpositive and right_nonnegative,
                    left_nonnegative and right_nonpositive,
                )
            elif isinstance(expression.op, ast.Mult):
                result = (
                    (left_nonpositive and right_nonnegative)
                    or (left_nonnegative and right_nonpositive),
                    (left_nonnegative and right_nonnegative)
                    or (left_nonpositive and right_nonpositive),
                )
            else:
                result = (False, False)
        elif (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id in {"min", "max"}
        ):
            item_signs = [
                bound_signs(item, before_line, seen_names)
                for argument in expression.args
                for item in extremum_items(argument, before_line)
            ]
            if not item_signs:
                result = (False, False)
            elif expression.func.id == "min":
                result = (
                    any(nonpositive for nonpositive, _ in item_signs),
                    all(nonnegative for _, nonnegative in item_signs),
                )
            else:
                result = (
                    all(nonpositive for nonpositive, _ in item_signs),
                    any(nonnegative for _, nonnegative in item_signs),
                )
        else:
            result = (False, False)
        sign_cache[cache_key] = result
        return result

    def bound_proves_at_most_zero(
        expression: ast.AST,
        before_line: int | None = None,
        seen_names: frozenset[str] = frozenset(),
    ) -> bool:
        return bound_signs(expression, before_line, seen_names)[0]

    def bound_proves_at_least_zero(
        expression: ast.AST,
        before_line: int | None = None,
        seen_names: frozenset[str] = frozenset(),
    ) -> bool:
        return bound_signs(expression, before_line, seen_names)[1]

    def is_errorbar_caplines_expression(expression: ast.AST) -> bool:
        return bool(
            isinstance(expression, ast.Subscript)
            and isinstance(expression.value, ast.Name)
            and expression.value.id in errorbar_container_names
            and numeric_literal(expression.slice) == 1.0
        )

    capline_tuple_names = {
        name
        for name, expression in single_assignments.items()
        if is_errorbar_caplines_expression(expression)
    }

    def x_limits_prove_zero_visible(call: ast.Call) -> bool:
        lower: ast.AST | None = None
        upper: ast.AST | None = None
        if len(call.args) >= 2:
            lower, upper = call.args[:2]
        elif (
            len(call.args) == 1
            and isinstance(call.args[0], (ast.List, ast.Tuple))
            and len(call.args[0].elts) == 2
        ):
            lower, upper = call.args[0].elts
        for item in call.keywords:
            if item.arg in {"left", "xmin"}:
                lower = item.value
            elif item.arg in {"right", "xmax"}:
                upper = item.value
        return bool(
            lower is not None
            and upper is not None
            and bound_proves_at_most_zero(lower, getattr(call, "lineno", None))
            and bound_proves_at_least_zero(upper, getattr(call, "lineno", None))
        )

    violations: set[str] = set()
    effect_x_axes = {
        ast.dump(node.func.value, include_attributes=False)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_xlabel"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and re.search(
            r"\b(?:effect|mean\s+difference|contrast|estimate)\b",
            node.args[0].value,
            re.IGNORECASE,
        )
    }
    labeled_artist_axes: set[str] = set()
    null_reference_axes: set[str] = set()
    labeled_scatter_centers: dict[str, list[float]] = {}
    categorical_tick_counts: dict[str, int] = {}
    for candidate in ast.walk(tree):
        if not isinstance(candidate, ast.Call) or not isinstance(
            candidate.func, ast.Attribute
        ):
            continue
        axis_key = ast.dump(candidate.func.value, include_attributes=False)
        label = next(
            (item.value for item in candidate.keywords if item.arg == "label"),
            None,
        )
        if candidate.func.attr in {
            "bar",
            "errorbar",
            "fill_between",
            "hlines",
            "plot",
            "scatter",
        } and could_be_visible_label(label):
            labeled_artist_axes.add(axis_key)
        if (
            candidate.func.attr == "scatter"
            and could_be_visible_label(label)
            and candidate.args
        ):
            center = categorical_position_center(
                candidate.args[0], getattr(candidate, "lineno", None)
            )
            if center is not None:
                labeled_scatter_centers.setdefault(axis_key, []).append(center)
        if (
            candidate.func.attr == "set_xticks"
            and candidate.args
            and isinstance(candidate.args[0], (ast.List, ast.Tuple))
        ):
            tick_values = [numeric_literal(item) for item in candidate.args[0].elts]
            if all(value is not None for value in tick_values):
                categorical_tick_counts[axis_key] = len(set(tick_values))
        if candidate.func.attr == "axvline":
            x_expression = (
                candidate.args[0]
                if candidate.args
                else next(
                    (
                        item.value
                        for item in candidate.keywords
                        if item.arg in {"x", "xmin"}
                    ),
                    None,
                )
            )
            if is_literal_zero(x_expression):
                null_reference_axes.add(axis_key)
    if any(
        categorical_tick_counts.get(axis_key, 0) >= 2
        and len(centers) >= 2
        and len({round(center, 12) for center in centers}) < len(centers)
        for axis_key, centers in labeled_scatter_centers.items()
    ):
        violations.add(
            "Scientific categorical scatter groups share a jitter center despite "
            "distinct x-axis categories; add each group's declared category center "
            "to its jitter so raw observations align with their own tick and mean"
        )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "twinx",
            "twiny",
            "secondary_xaxis",
            "secondary_yaxis",
        }:
            violations.add(
                "Scientific display preflight rejects secondary/twin axes; "
                "place the contrast or differently scaled estimand in a separate "
                "labeled panel"
            )
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_xticks"
            and node.args
            and isinstance(node.args[0], (ast.List, ast.Tuple))
        ):
            axis_key = ast.dump(node.func.value, include_attributes=False)
            if axis_key in effect_x_axes and not node.args[0].elts:
                violations.add(
                    "Scientific effect-estimate axes require visible numeric x ticks; "
                    "do not remove the scale with set_xticks([])"
                )
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
                violations.add(
                    "Scientific categorical axes require unique tick positions; "
                    "duplicate x ticks overlap groups and make the display misleading"
                )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "legend":
            axis_key = ast.dump(node.func.value, include_attributes=False)
            explicit_handles = bool(node.args) or any(
                item.arg in {"handles", "labels"} for item in node.keywords
            )
            if not explicit_handles and axis_key not in labeled_artist_axes:
                violations.add(
                    "Matplotlib legend() has no labeled artists on this axis; omit "
                    "the empty legend or add an honest label to a plotted artist"
                )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get_segments":
            receiver = node.func.value
            if (
                isinstance(receiver, ast.Name) and receiver.id in capline_tuple_names
            ) or is_errorbar_caplines_expression(receiver):
                violations.add(
                    "Matplotlib ErrorbarContainer caplines are a tuple of Line2D "
                    "artists and do not support get_segments(); inspect each "
                    "capline with get_xdata()/get_ydata() or inspect barlinecols"
                )
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get_segments", "get_xdata", "get_ydata"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in errorbar_container_names
        ):
            violations.add(
                "Matplotlib ErrorbarContainer does not expose line coordinate "
                "methods directly; inspect container[0] with get_xdata()/get_ydata(), "
                "individual container[1] caplines, or container[2] barlinecols"
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "set_xlim":
            axis_key = ast.dump(node.func.value, include_attributes=False)
            if (
                axis_key in effect_x_axes
                and axis_key in null_reference_axes
                and not x_limits_prove_zero_visible(node)
            ):
                violations.add(
                    "Matplotlib effect-axis limits may clip the intended zero/null "
                    "reference; include zero explicitly when computing set_xlim()"
                )
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "errorbar":
            continue
        keywords = {item.arg: item.value for item in node.keywords if item.arg}
        if "linewidths" in keywords:
            violations.add(
                "Matplotlib errorbar rejects linewidths=; use linewidth= or elinewidth="
            )
        x_expression = node.args[0] if node.args else keywords.get("x")
        y_expression = node.args[1] if len(node.args) >= 2 else keywords.get("y")
        if x_expression is not None and y_expression is not None and "xerr" in keywords:
            if (
                isinstance(x_expression, (ast.List, ast.Tuple))
                and len(x_expression.elts) == 1
            ):
                x_expression = x_expression.elts[0]
            x_is_zero = (
                resolved_numeric_literal(x_expression, getattr(node, "lineno", None))
                == 0.0
            )
            y_names = {
                item.id for item in ast.walk(y_expression) if isinstance(item, ast.Name)
            }
            xerr_names = {
                item.id
                for item in ast.walk(keywords["xerr"])
                if isinstance(item, ast.Name)
            }
            if x_is_zero and y_names & xerr_names:
                violations.add(
                    "Matplotlib effect interval is transposed: xerr is derived from "
                    "the value plotted on y while x is fixed at zero; plot the "
                    "estimate on x and use a constant categorical y position"
                )
        if x_expression is not None and y_expression is not None and "yerr" in keywords:
            if (
                isinstance(x_expression, (ast.List, ast.Tuple))
                and len(x_expression.elts) == 1
            ):
                x_expression = x_expression.elts[0]
            x_is_zero = (
                resolved_numeric_literal(x_expression, getattr(node, "lineno", None))
                == 0.0
            )
            axis_key = ast.dump(node.func.value, include_attributes=False)
            if x_is_zero and axis_key in effect_x_axes:
                violations.add(
                    "Matplotlib effect interval is transposed: an axis labeled for "
                    "the effect estimate fixes x at zero and places the interval in "
                    "yerr; plot the estimate and confidence interval on x with a "
                    "constant categorical y position"
                )
        scalar_x = isinstance(x_expression, ast.Constant)
        if not scalar_x:
            continue
        for name in ("xerr", "yerr"):
            value = keywords.get(name)
            if not isinstance(value, (ast.List, ast.Tuple)) or len(value.elts) != 2:
                continue
            if any(isinstance(item, (ast.List, ast.Tuple)) for item in value.elts):
                continue
            violations.add(
                f"Matplotlib singleton asymmetric {name} must have shape (2, 1); "
                f"use [[lower], [upper]] rather than [lower, upper]"
            )
    return sorted(violations)


def _unavailable_prior_reference_violations(
    code: str, successful_execution_ids: set[str]
) -> list[str]:
    referenced = {
        match.group("execution_id")
        for match in PRIOR_EXECUTION_REFERENCE.finditer(code)
    }
    return [
        (
            f"/prior/{execution_id} is not a successful execution in the current "
            "attempt; use /history/attempt-N/exec-ID/output only for the exact "
            "registered prior-attempt artifact"
        )
        for execution_id in sorted(referenced - successful_execution_ids)
    ]


def _read_bounded(path: Path) -> str:
    data = path.read_bytes()[:RETURN_TEXT_BYTES]
    text = data.decode("utf-8", errors="replace")
    if path.stat().st_size > RETURN_TEXT_BYTES:
        text += f"\n...[truncated; full log: {path}]"
    return text


def _artifact(path: Path, description: str) -> ArtifactRef:
    return ArtifactRef(
        path=str(path.resolve()),
        sha256=sha256_file(path),
        description=description,
    )


def _output_previews(artifacts: list[ArtifactRef]) -> dict[str, str]:
    previews: dict[str, str] = {}
    for artifact in artifacts:
        path = Path(artifact.path)
        if path.suffix.lower() not in PREVIEW_SUFFIXES:
            continue
        data = path.read_bytes()[:PREVIEW_TEXT_BYTES]
        preview = data.decode("utf-8", errors="replace")
        if path.stat().st_size > PREVIEW_TEXT_BYTES:
            preview += "\n...[truncated]"
        previews[artifact.path] = preview
    return previews


@dataclass
class AnalysisExecutor:
    workspace: Path
    root: Path
    settings: SandboxSettings
    environment_dir: Path | None = None
    history_dir: Path | None = None
    cancel_event: threading.Event | None = None
    _records: list[ComputationRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.workspace = self.workspace.resolve()
        self.root = self.root.resolve()
        if self.environment_dir is not None:
            self.environment_dir = self.environment_dir.resolve()
        self.history_dir = (self.history_dir or self.root.parent).resolve()
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.root, 0o700)

    def _required_paths(self, language: Language) -> list[Path]:
        common = [self.settings.bwrap, self.settings.prlimit, Path("/usr")]
        if language == "python":
            return [
                *common,
                self.settings.python,
                self.settings.python_prefix,
                self.settings.python_packages,
            ]
        return [*common, self.settings.rscript, self.settings.r_library, Path("/etc/R")]

    def _validate_runtime(self, language: Language) -> None:
        missing = [
            str(path) for path in self._required_paths(language) if not path.exists()
        ]
        if missing:
            raise RuntimeError(
                f"sandbox runtime paths are missing: {', '.join(missing)}"
            )

    def _bwrap_command(
        self,
        language: Language,
        script: Path,
        output_dir: Path,
        workspace_packages: Path | None = None,
    ) -> list[str]:
        bwrap = [
            str(self.settings.bwrap),
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/bin",
            "/bin",
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/lib64",
            "/lib64",
            "--dir",
            "/etc",
            "--ro-bind",
            "/etc/alternatives",
            "/etc/alternatives",
            "--ro-bind",
            "/etc/ld.so.cache",
            "/etc/ld.so.cache",
            "--dir",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/tmp/home",
            "--dir",
            "/workspace",
            "--ro-bind",
            str(self.workspace),
            "/workspace",
            "--dir",
            "/prior",
            "--ro-bind",
            str(self.root),
            "/prior",
            "--ro-bind",
            str(self.history_dir),
            "/history",
            "--dir",
            "/analysis",
            "--ro-bind",
            str(script),
            f"/analysis/{script.name}",
            "--dir",
            "/output",
            "--bind",
            str(output_dir),
            "/output",
            "--clearenv",
            "--setenv",
            "HOME",
            "/tmp/home",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "OMP_NUM_THREADS",
            "1",
            "--setenv",
            "OPENBLAS_NUM_THREADS",
            "1",
            "--setenv",
            "MKL_NUM_THREADS",
            "1",
            "--chdir",
            "/output",
        ]
        if language == "python":
            try:
                python_rel = self.settings.python.resolve().relative_to(
                    self.settings.python_prefix.resolve()
                )
            except ValueError as exc:
                raise RuntimeError(
                    "sandbox Python executable must be inside its configured prefix"
                ) from exc
            sandbox_python = Path("/opt/python-runtime") / python_rel
            python_path_setup = "sys.path.insert(0,'/opt/python-packages');"
            bwrap.extend(["--dir", "/opt"])
            if workspace_packages is not None and workspace_packages.is_dir():
                bwrap.extend(
                    [
                        "--ro-bind",
                        str(workspace_packages),
                        "/opt/workspace-python-packages",
                    ]
                )
                python_path_setup += (
                    "sys.path.insert(0,'/opt/workspace-python-packages');"
                )
            bwrap.extend(
                [
                    "--ro-bind",
                    str(self.settings.python_prefix),
                    "/opt/python-runtime",
                    "--ro-bind",
                    str(self.settings.python_packages),
                    "/opt/python-packages",
                    "--setenv",
                    "PATH",
                    "/opt/python-runtime/bin:/usr/bin",
                    "--setenv",
                    "PYTHONHASHSEED",
                    "0",
                    "--setenv",
                    "MPLCONFIGDIR",
                    "/tmp/matplotlib",
                    str(sandbox_python),
                    "-I",
                    "-c",
                    (
                        "import runpy,sys;"
                        "import resource;"
                        f"resource.setrlimit(resource.RLIMIT_NPROC,({self.settings.max_processes},{self.settings.max_processes}));"
                        f"{python_path_setup}"
                        f"runpy.run_path('/analysis/{script.name}',run_name='__main__')"
                    ),
                ]
            )
        else:
            r_libraries = "/opt/R-library"
            bwrap.extend(["--dir", "/opt"])
            if workspace_packages is not None and workspace_packages.is_dir():
                bwrap.extend(
                    [
                        "--ro-bind",
                        str(workspace_packages),
                        "/opt/workspace-R-library",
                    ]
                )
                r_libraries = "/opt/workspace-R-library:/opt/R-library"
            bwrap.extend(
                [
                    "--ro-bind",
                    "/etc/R",
                    "/etc/R",
                    "--ro-bind",
                    str(self.settings.r_library),
                    "/opt/R-library",
                    "--setenv",
                    "PATH",
                    "/usr/bin",
                    "--setenv",
                    "R_LIBS_USER",
                    r_libraries,
                    "/usr/bin/bash",
                    "-c",
                    (
                        f"ulimit -u {self.settings.max_processes}; "
                        f"exec /usr/bin/Rscript --vanilla /analysis/{script.name}"
                    ),
                ]
            )
        return bwrap

    def _snapshot_environment(
        self,
        language: Language,
        call_dir: Path,
    ) -> tuple[Path | None, dict[str, str], list[ArtifactRef]]:
        if self.environment_dir is None:
            return None, {}, []
        current = self.environment_dir / language
        if not current.exists():
            return None, {}, []
        generation = current.resolve()
        generations = (self.environment_dir / ".generations").resolve()
        if generation.parent != generations:
            raise RuntimeError("workspace package generation escaped its environment")
        packages = generation / "packages"
        lock = generation / "lock.json"
        if not packages.is_dir() or not lock.is_file():
            raise RuntimeError("workspace package generation is incomplete")
        lock_copy = call_dir / f"environment-{language}-lock.json"
        lock_copy.write_bytes(lock.read_bytes())
        lock_copy.chmod(0o600)
        digest = sha256_file(lock_copy)
        return (
            packages,
            {language: digest},
            [_artifact(lock_copy, "workspace package environment lock")],
        )

    def _limited_command(self, command: list[str], timeout_seconds: int) -> list[str]:
        return [
            str(self.settings.prlimit),
            f"--cpu={timeout_seconds + 1}",
            f"--as={self.settings.max_memory_bytes}",
            f"--fsize={self.settings.max_file_bytes}",
            "--nofile=1024",
            "--core=0",
            "--",
            *command,
        ]

    def _inspect_outputs(self, output_dir: Path) -> tuple[list[ArtifactRef], list[str]]:
        artifacts: list[ArtifactRef] = []
        violations: list[str] = []
        total_bytes = 0
        for path in sorted(output_dir.rglob("*")):
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode):
                violations.append(f"non-regular output rejected: {path}")
                continue
            total_bytes += info.st_size
            if info.st_size > self.settings.max_file_bytes:
                violations.append(f"output file exceeds per-file limit: {path}")
                continue
            relative = path.relative_to(output_dir)
            if (
                relative.parts
                and relative.parts[0].casefold() == "tables"
                and path.suffix.casefold() in {".csv", ".tsv"}
            ):
                from .reporting import read_table_preview

                try:
                    read_table_preview(path)
                except ValueError as exc:
                    violations.append(f"invalid reader table {path}: {exc}")
            if path.suffix.casefold() in {".json", ".ipynb"}:
                try:
                    json.loads(
                        path.read_text(encoding="utf-8"),
                        parse_constant=_reject_nonfinite_json,
                    )
                except (OSError, UnicodeError, ValueError) as exc:
                    violations.append(
                        f"generated output must be strict JSON {path}: {exc}"
                    )
            if path.suffix.casefold() == ".jsonl":
                try:
                    for line_number, line in enumerate(
                        path.read_text(encoding="utf-8").splitlines(), 1
                    ):
                        if line.strip():
                            try:
                                json.loads(
                                    line,
                                    parse_constant=_reject_nonfinite_json,
                                )
                            except ValueError as exc:
                                raise ValueError(f"line {line_number}: {exc}") from exc
                except (OSError, UnicodeError, ValueError) as exc:
                    violations.append(
                        f"generated output must be strict JSON Lines {path}: {exc}"
                    )
            artifacts.append(_artifact(path, "sandbox-generated analysis artifact"))
        if total_bytes > self.settings.max_output_bytes:
            violations.append(
                f"total output exceeds {self.settings.max_output_bytes} bytes"
            )
            artifacts = []
        return artifacts, violations

    def execute(
        self,
        language: Language,
        code: str,
        timeout_seconds: int = 120,
    ) -> dict:
        """Execute one Python or R script inside the fixed sandbox profile."""

        execution_id = f"exec-{len(self._records) + 1:03d}"
        started_at = utc_now()
        started = time.monotonic()
        code_bytes = code.encode("utf-8")
        extension = "py" if language == "python" else "R"
        call_dir = self.root / execution_id
        output_dir = call_dir / "output"
        call_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        output_dir.mkdir(mode=0o700)
        script = call_dir / f"analysis.{extension}"
        stdout_path = call_dir / "stdout.txt"
        stderr_path = call_dir / "stderr.txt"
        script.write_bytes(code_bytes)
        script.chmod(0o600)
        environment_locks: dict[str, str] = {}
        environment_artifacts: list[ArtifactRef] = []
        workspace_packages: Path | None = None

        status: Literal[
            "succeeded", "failed", "timed_out", "cancelled", "policy_denied"
        ]
        exit_code: int | None = None
        violations: list[str] = []
        if not code.strip():
            violations.append("code must not be empty")
        if len(code_bytes) > self.settings.max_code_bytes:
            violations.append(f"code exceeds {self.settings.max_code_bytes} byte limit")
        if len(self._records) >= self.settings.max_calls_per_attempt:
            violations.append("analysis call budget exhausted")
        if language == "python":
            violations.extend(_python_static_violations(code))
        violations.extend(
            _unavailable_prior_reference_violations(
                code,
                {
                    record.execution_id
                    for record in self._records
                    if record.status == "succeeded"
                },
            )
        )
        try:
            (
                workspace_packages,
                environment_locks,
                environment_artifacts,
            ) = self._snapshot_environment(language, call_dir)
        except Exception as exc:
            violations.append(str(exc))
        try:
            self._validate_runtime(language)
        except Exception as exc:
            violations.append(str(exc))

        timeout_seconds = max(1, min(timeout_seconds, self.settings.max_wall_seconds))
        timed_out = False
        cancelled = False
        output_artifacts: list[ArtifactRef] = []
        if violations:
            status = "policy_denied"
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("\n".join(violations) + "\n", encoding="utf-8")
        else:
            command = self._limited_command(
                self._bwrap_command(
                    language,
                    script,
                    output_dir,
                    workspace_packages=workspace_packages,
                ),
                timeout_seconds,
            )
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                deadline = time.monotonic() + timeout_seconds + 2
                while process.poll() is None:
                    if self.cancel_event is not None and self.cancel_event.is_set():
                        cancelled = True
                        break
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                    time.sleep(0.05)
                if cancelled or timed_out:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=2)
                else:
                    process.wait()
                exit_code = process.returncode
            output_artifacts, output_violations = self._inspect_outputs(output_dir)
            violations.extend(output_violations)
            if cancelled:
                status = "cancelled"
            elif timed_out:
                status = "timed_out"
            elif exit_code == 0 and not violations:
                status = "succeeded"
            else:
                status = "policy_denied" if violations else "failed"

        if status != "succeeded" and output_artifacts:
            output_root = output_dir.resolve()
            rejected = [
                (Path(artifact.path).relative_to(output_root), artifact)
                for artifact in output_artifacts
            ]
            rejected_dir = call_dir / "rejected_output"
            output_dir.rename(rejected_dir)
            output_artifacts = [
                ArtifactRef(
                    path=str((rejected_dir / relative).resolve()),
                    sha256=artifact.sha256,
                    description="rejected sandbox output (not evidence)",
                )
                for relative, artifact in rejected
            ]

        stdout_path.chmod(0o600)
        stderr_path.chmod(0o600)
        artifacts = [
            _artifact(script, f"{language} analysis source"),
            _artifact(stdout_path, "captured standard output"),
            _artifact(stderr_path, "captured standard error"),
            *environment_artifacts,
            *output_artifacts,
        ]
        record = ComputationRecord(
            execution_id=execution_id,
            language=language,
            code_sha256=sha256_bytes(code_bytes),
            started_at=started_at,
            duration_seconds=round(time.monotonic() - started, 3),
            exit_code=exit_code,
            status=status,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            environment_locks=environment_locks,
            artifacts=artifacts,
        )
        self._records.append(record)
        calls_used = len(self._records)
        calls_remaining = max(0, self.settings.max_calls_per_attempt - calls_used)
        write_json(
            call_dir / "execution.json",
            {
                **record.model_dump(mode="json"),
                "violations": sorted(set(violations)),
                "limits": {
                    "wall_seconds": timeout_seconds,
                    "memory_bytes": self.settings.max_memory_bytes,
                    "processes": self.settings.max_processes,
                    "file_bytes": self.settings.max_file_bytes,
                    "total_output_bytes": self.settings.max_output_bytes,
                },
            },
        )
        return {
            "execution_id": execution_id,
            "language": language,
            "status": status,
            "exit_code": exit_code,
            "duration_seconds": record.duration_seconds,
            "environment_locks": environment_locks,
            "stdout": _read_bounded(stdout_path),
            "stderr": _read_bounded(stderr_path),
            "artifacts": [item.model_dump(mode="json") for item in artifacts],
            "output_previews": (
                _output_previews(output_artifacts) if status == "succeeded" else {}
            ),
            "violations": sorted(set(violations)),
            "calls_used": calls_used,
            "calls_remaining": calls_remaining,
            "stop_required": "analysis call budget exhausted" in violations,
            "workspace_path": "/workspace",
            "prior_outputs_path": "/prior",
            "attempt_history_path": "/history",
            "output_path": "/output",
        }

    def evidence(self) -> ComputationEvidence:
        successful = [
            record for record in self._records if record.status == "succeeded"
        ]
        artifacts = [
            artifact
            for record in successful
            for artifact in record.artifacts
            if artifact.description == "sandbox-generated analysis artifact"
        ]
        return ComputationEvidence(
            successful_calls=len(successful),
            records=list(self._records),
            artifacts=artifacts,
        )

    def close(self) -> None:
        """Release executor-scoped resources (none for in-process execution)."""


def build_analysis_tools(executor: AnalysisExecutor):
    def run_python_analysis(code: str, timeout_seconds: int = 120) -> dict:
        """Run Python scientific analysis in an offline sandbox.

        The assigned project is read-only at /workspace. Earlier calls are
        read-only at /prior/<execution-id>/output, and prior repair attempts at
        /history/attempt-N/<execution-id>/output. Write all tables,
        figures, and machine-readable results below /output. NumPy, pandas,
        SciPy, statsmodels, scikit-learn, and Matplotlib are available. This
        runtime uses Matplotlib 3.10+; Axes.boxplot uses tick_labels rather than
        the removed labels keyword.

        Args:
            code: Complete Python script to execute.
            timeout_seconds: Wall-time request, capped by controller policy.
        """

        return executor.execute("python", code, timeout_seconds)

    def run_r_analysis(code: str, timeout_seconds: int = 120) -> dict:
        """Run R scientific analysis in an offline sandbox.

        The assigned project is read-only at /workspace. Earlier calls are
        read-only at /prior/<execution-id>/output, and prior repair attempts at
        /history/attempt-N/<execution-id>/output. Write all tables,
        figures, and machine-readable results below /output. The curated R figure
        baseline includes publication rendering, tidy data/import, Excel,
        regression, survival, diagnostic-performance, survey, and imputation
        packages. It includes ggplot2, ragg, patchwork, ComplexHeatmap, dplyr,
        tidyr, openxlsx, broom, survival, survminer, pROC, survey, and mice. Verify
        version-sensitive requirements and use the isolated package installer only
        for missing, outdated, or genuinely specialist packages.

        Args:
            code: Complete R script to execute.
            timeout_seconds: Wall-time request, capped by controller policy.
        """

        return executor.execute("r", code, timeout_seconds)

    return [run_python_analysis, run_r_analysis]


def sandbox_preflight(settings: SandboxSettings, workspace: Path | None = None) -> dict:
    """Check runtime paths and execute fixed Python/R isolation probes."""

    paths = {
        "bwrap": settings.bwrap,
        "prlimit": settings.prlimit,
        "python": settings.python,
        "python_prefix": settings.python_prefix,
        "python_packages": settings.python_packages,
        "rscript": settings.rscript,
        "r_library": settings.r_library,
    }
    # A remote worker owns its runtime paths; the execution probe below is the
    # authoritative check. Local path existence matters only for in-process mode.
    missing = (
        []
        if settings.worker_url
        else [name for name, path in paths.items() if not path.exists()]
    )
    result = {
        "paths": {name: str(path) for name, path in paths.items()},
        "missing_required": missing,
        "network": "unshared",
        "workspace": "read-only",
        "output": "per-call writable directory",
    }
    if missing:
        result["probes"] = {}
        return result
    managed_base: Path | None = None
    temporary_root = None
    if settings.worker_url:
        data = Path(os.environ.get("SCIENTIFIC_AGENT_DATA_DIR", "/data")).resolve()
        managed_base = data / "workspaces" / str(uuid.uuid4())
        probe_workspace = managed_base / "files"
        probe_root = managed_base / "runs" / "preflight" / "computations" / "attempt-1"
        probe_workspace.mkdir(parents=True, mode=0o700)
        probe_root.mkdir(parents=True, mode=0o700)
        executor: AnalysisRunner = RemoteAnalysisExecutor(
            probe_workspace, probe_root, settings
        )
    else:
        temporary_root = tempfile.TemporaryDirectory(
            prefix="scientific-agent-preflight-"
        )
        executor = AnalysisExecutor(
            (workspace or Path.cwd()).resolve(),
            Path(temporary_root.name),
            settings,
        )
    try:
        python_probe = executor.execute(
            "python",
            (
                "import matplotlib,numpy,pandas,scipy,sklearn,statsmodels;"
                "open('/output/python-ok.txt', 'w').write('ok')"
            ),
            timeout_seconds=15,
        )
        r_packages = ",".join(
            f"'{package}'" for package in R_ANALYSIS_BASELINE_PACKAGES
        )
        r_version_checks = ";".join(
            f"stopifnot(packageVersion('{package}') >= '{version}')"
            for package, version in R_ANALYSIS_BASELINE_MINIMUM_VERSIONS.items()
        )
        r_probe = executor.execute(
            "r",
            (
                f"stopifnot(all(vapply(c({r_packages}), "
                "requireNamespace, logical(1), quietly=TRUE)));"
                f"{r_version_checks};"
                "writeLines('ok', '/output/r-ok.txt')"
            ),
            timeout_seconds=15,
        )
    finally:
        executor.close()
        if temporary_root is not None:
            temporary_root.cleanup()
        if managed_base is not None:
            shutil.rmtree(managed_base, ignore_errors=True)
    result["probes"] = {
        "python": python_probe["status"],
        "r": r_probe["status"],
    }
    return result


class AnalysisRunner(Protocol):
    def execute(
        self,
        language: Language,
        code: str,
        timeout_seconds: int = 120,
    ) -> dict: ...

    def evidence(self) -> ComputationEvidence: ...

    def close(self) -> None: ...


@dataclass
class RemoteAnalysisExecutor:
    """Typed client for the isolated, non-published container worker."""

    workspace: Path
    root: Path
    settings: SandboxSettings
    cancel_event: threading.Event | None = None
    _evidence: ComputationEvidence = field(default_factory=ComputationEvidence)

    def __post_init__(self) -> None:
        if not self.settings.worker_url or not self.settings.worker_token:
            raise RuntimeError("sandbox worker URL and token are both required")

    def execute(
        self,
        language: Language,
        code: str,
        timeout_seconds: int = 120,
    ) -> dict:
        timeout = max(1, min(timeout_seconds, self.settings.max_wall_seconds))
        request_id = str(uuid.uuid4())
        finished = threading.Event()

        def cancel_remote() -> None:
            if self.cancel_event is None:
                return
            while not finished.is_set():
                if not self.cancel_event.wait(timeout=0.1):
                    continue
                try:
                    response = httpx.post(
                        f"{self.settings.worker_url}/cancel",
                        headers={
                            "Authorization": f"Bearer {self.settings.worker_token}"
                        },
                        json={"request_id": request_id},
                        timeout=5,
                    )
                    if response.status_code in {200, 202}:
                        return
                except httpx.HTTPError:
                    pass
                finished.wait(timeout=0.2)

        watcher = threading.Thread(
            target=cancel_remote,
            name=f"sandbox-cancel-{request_id[:8]}",
            daemon=True,
        )
        watcher.start()
        try:
            response = httpx.post(
                f"{self.settings.worker_url}/execute",
                headers={"Authorization": f"Bearer {self.settings.worker_token}"},
                json={
                    "request_id": request_id,
                    "workspace": str(self.workspace),
                    "computation_root": str(self.root),
                    "language": language,
                    "code": code,
                    "timeout_seconds": timeout,
                    "max_calls_per_attempt": self.settings.max_calls_per_attempt,
                },
                timeout=timeout + 15,
            )
        finally:
            finished.set()
        if response.is_error:
            try:
                detail = response.json().get("detail", "worker request failed")
            except ValueError:
                detail = "worker returned a non-JSON error"
            raise RuntimeError(
                f"sandbox worker returned HTTP {response.status_code}: {detail}"
            )
        payload = response.json()
        self._evidence = ComputationEvidence.model_validate(payload["evidence"])
        return payload["result"]

    def evidence(self) -> ComputationEvidence:
        return self._evidence

    def close(self) -> None:
        response = httpx.post(
            f"{self.settings.worker_url}/release",
            headers={"Authorization": f"Bearer {self.settings.worker_token}"},
            json={
                "workspace": str(self.workspace),
                "computation_root": str(self.root),
            },
            timeout=10,
        )
        if response.is_error:
            try:
                detail = response.json().get("detail", "worker release failed")
            except ValueError:
                detail = "worker returned a non-JSON release error"
            raise RuntimeError(
                f"sandbox worker release returned HTTP {response.status_code}: {detail}"
            )


def create_analysis_executor(
    workspace: Path,
    root: Path,
    settings: SandboxSettings,
    *,
    cancel_event: threading.Event | None = None,
) -> AnalysisRunner:
    if settings.worker_url:
        return RemoteAnalysisExecutor(workspace, root, settings, cancel_event)
    return AnalysisExecutor(workspace, root, settings, cancel_event=cancel_event)
