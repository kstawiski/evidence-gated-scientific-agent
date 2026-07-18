#!/usr/bin/env python3
"""Run and score a deployed Evidence Bench case through its browser API."""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import math
import shlex
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def parse_env(path: Path) -> dict[str, str]:
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw or raw.lstrip().startswith("#"):
            continue
        name, value = raw.split("=", 1)
        parsed = shlex.split(value)
        if len(parsed) == 1:
            values[name] = parsed[0]
    return values


class Client:
    def __init__(
        self,
        base_url: str,
        username: str | None = None,
        password: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = {}
        if username and password:
            token = base64.b64encode(f"{username}:{password}".encode()).decode()
            self.headers = {"Authorization": f"Basic {token}"}

    def request(self, method: str, path: str, *, payload=None, body=None, headers=None):
        request_headers = {**self.headers, **(headers or {})}
        if payload is not None:
            body = json.dumps(payload).encode()
            request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
                return json.loads(data) if data else None
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"HTTP {exc.code}: {exc.read().decode(errors='replace')}"
            ) from exc

    def download_artifact(self, run_id: str, path: str) -> bytes:
        query = urllib.parse.urlencode({"path": path})
        request = urllib.request.Request(
            f"{self.base_url}/api/runs/{run_id}/artifacts?{query}",
            headers=self.headers,
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()


def multipart(filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = "evidence-bench-eval-boundary"
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="upload"; filename="{filename}"\r\n'
            "Content-Type: text/csv\r\n\r\n"
        ).encode()
        + content
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={boundary}"


def _artifact_jsons(client: Client, run_id: str, paths: set[str]) -> dict[str, object]:
    values = {}
    for path in paths:
        if not path.lower().endswith(".json") or "/output/" not in path:
            continue
        try:
            values[path] = json.loads(client.download_artifact(run_id, path))
        except (OSError, ValueError, RuntimeError):
            continue
    return values


def _find_json(values: dict[str, object], suffix: str) -> object | None:
    for path, value in values.items():
        if path.endswith(suffix):
            return value
    return None


def _metric(value: dict, *names: str) -> float:
    pending = [value]
    while pending:
        current = pending.pop(0)
        for name in names:
            candidate = current.get(name)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return float(candidate)
            if (
                isinstance(candidate, dict)
                and isinstance(candidate.get("value"), (int, float))
                and not isinstance(candidate["value"], bool)
            ):
                return float(candidate["value"])
            if (
                isinstance(candidate, list)
                and len(candidate) == 1
                and isinstance(candidate[0], (int, float))
                and not isinstance(candidate[0], bool)
            ):
                return float(candidate[0])
            if isinstance(candidate, dict):
                for estimate_name in ("value", "estimate"):
                    estimate = candidate.get(estimate_name)
                    if isinstance(estimate, (int, float)) and not isinstance(
                        estimate, bool
                    ):
                        return float(estimate)
                    if (
                        isinstance(estimate, list)
                        and len(estimate) == 1
                        and isinstance(estimate[0], (int, float))
                        and not isinstance(estimate[0], bool)
                    ):
                        return float(estimate[0])
        pending.extend(item for item in current.values() if isinstance(item, dict))
    return 1e9


def _language_result(
    computation: dict,
    generated: dict[str, object],
    language: str,
    reconciliation: dict | None = None,
) -> dict | None:
    if isinstance(reconciliation, dict):
        comparisons = reconciliation.get("comparisons")
        bound_digests = []
        for comparison in comparisons if isinstance(comparisons, list) else []:
            if not isinstance(comparison, dict):
                continue
            source = comparison.get(language)
            if isinstance(source, dict) and isinstance(
                source.get("artifact_sha256"), str
            ):
                bound_digests.append(source["artifact_sha256"].lower())
        bound_values = []
        for digest in sorted(set(bound_digests)):
            bound_paths = [
                artifact.get("path", "")
                for record in computation.get("records", [])
                if record.get("language") == language
                and record.get("status") == "succeeded"
                for artifact in record.get("artifacts", [])
                if artifact.get("description") == "sandbox-generated analysis artifact"
                and str(artifact.get("sha256", "")).lower() == digest
            ]
            if not bound_paths:
                return None
            digest_values = []
            for manifest_path, value in generated.items():
                if isinstance(value, dict) and any(
                    path.endswith(manifest_path) for path in bound_paths
                ):
                    digest_values.append(value)
            if not digest_values:
                return None
            bound_values.extend(digest_values)
        if bound_digests:
            return next(
                (
                    value
                    for value in bound_values
                    if _known_effect_matches_reference(value)
                ),
                None,
            )
        if isinstance(comparisons, list):
            return None

    artifact_paths = [
        artifact["path"]
        for record in computation.get("records", [])
        if record.get("language") == language and record.get("status") == "succeeded"
        for artifact in record.get("artifacts", [])
        if artifact.get("description") == "sandbox-generated analysis artifact"
    ]
    fallback = None
    for manifest_path, value in generated.items():
        if not isinstance(value, dict):
            continue
        if not any(path.endswith(manifest_path) for path in artifact_paths):
            continue
        if (
            _metric(
                value,
                "mean_difference",
                "mean_diff",
                "mean_difference_treatment_minus_control",
            )
            != 1e9
            and _metric(value, "hedges_g") != 1e9
        ):
            if _known_effect_matches_reference(value):
                return value
            fallback = fallback or value
    return fallback


def _known_effect_matches_reference(value: dict) -> bool:
    def metric(paths: tuple[str, ...], aliases: tuple[str, ...]) -> float:
        for path in paths:
            current: object = value
            for component in path.split("."):
                if isinstance(current, dict) and component in current:
                    current = current[component]
                    continue
                if (
                    isinstance(current, list)
                    and component.isdigit()
                    and int(component) < len(current)
                ):
                    current = current[int(component)]
                    continue
                else:
                    break
            else:
                if isinstance(current, (int, float)) and not isinstance(current, bool):
                    return float(current)

        # A fallback alias is accepted only when every occurrence agrees. This
        # supports alternate schemas without letting a shallow reference/target
        # echo mask a contradictory value in the actual result object.
        observed: list[float] = []
        pending: list[object] = [value]
        while pending:
            current = pending.pop(0)
            if isinstance(current, dict):
                for key, candidate in current.items():
                    if (
                        key in aliases
                        and isinstance(candidate, (int, float))
                        and not isinstance(candidate, bool)
                    ):
                        observed.append(float(candidate))
                    elif isinstance(candidate, (dict, list)):
                        pending.append(candidate)
            elif isinstance(current, list):
                pending.extend(current)
        if not observed or any(
            not math.isclose(candidate, observed[0], rel_tol=1e-12, abs_tol=1e-12)
            for candidate in observed[1:]
        ):
            return 1e9
        return observed[0]

    expected = [
        (
            (
                "study_design.n_treatment",
                "sample_sizes.n_treatment",
                "groups.treatment.n",
                "group_statistics.treatment.n",
                "group_summaries.treatment.n",
            ),
            ("n_treatment", "treatment_n"),
            20.0,
        ),
        (
            (
                "study_design.n_control",
                "sample_sizes.n_control",
                "groups.control.n",
                "group_statistics.control.n",
                "group_summaries.control.n",
            ),
            ("n_control", "control_n"),
            20.0,
        ),
        (
            (
                "group_summaries.treatment.mean_change",
                "group_summaries.treatment.change_mean",
                "group_descriptives.treatment.mean_change",
                "descriptive_statistics.treatment.mean_change",
                "groups.treatment.mean_change",
                "group_statistics.treatment.mean_change",
            ),
            ("treatment_mean_change", "mean_change_treatment"),
            5.0,
        ),
        (
            (
                "group_summaries.control.mean_change",
                "group_summaries.control.change_mean",
                "group_descriptives.control.mean_change",
                "descriptive_statistics.control.mean_change",
                "groups.control.mean_change",
                "group_statistics.control.mean_change",
            ),
            ("control_mean_change", "mean_change_control"),
            0.0,
        ),
        (
            ("primary.point_estimate",),
            ("mean_difference", "mean_difference_treatment_minus_control"),
            5.0,
        ),
        (
            (
                "primary.welch_t_statistic",
                "primary.welch_t_test.t_statistic",
                "welch_t_test.t_statistic",
                "welch_t_test.statistic",
            ),
            ("welch_t_statistic", "t_statistic"),
            10.897247358851683,
        ),
        (
            (
                "primary.degrees_of_freedom",
                "primary.welch_t_test.degrees_of_freedom",
                "welch_t_test.degrees_of_freedom",
            ),
            ("degrees_of_freedom", "welch_df", "df_welch"),
            38.0,
        ),
        (
            (
                "primary.ci_lower",
                "primary.ci_95_lower",
                "primary.ci_lower_95",
                "primary.ci_95_percent.lower",
                "primary.confidence_interval_95.0",
            ),
            ("ci_95_lower", "ci_lower_95"),
            4.071144254485707,
        ),
        (
            (
                "primary.ci_upper",
                "primary.ci_95_upper",
                "primary.ci_upper_95",
                "primary.ci_95_percent.upper",
                "primary.confidence_interval_95.1",
            ),
            ("ci_95_upper", "ci_upper_95"),
            5.928855745514293,
        ),
        (
            (
                "primary.pooled_sd",
                "descriptive_statistics.treatment.sd_change",
                "effect_size.pooled_sd",
            ),
            ("pooled_sd",),
            1.4509525002200232,
        ),
        (
            (
                "primary.hedges_g_correction_factor_J",
                "primary.hedges_correction_J",
                "primary.hedges_g.j_correction_factor",
                "effect_size.hedges_g_correction_factor_J",
                "effect_size.hedges_correction_J",
                "effect_size.j_correction_factor",
            ),
            (
                "hedges_g_correction_factor_J",
                "hedges_g_correction_J",
                "hedges_correction_J",
                "j_correction",
                "J_correction",
            ),
            0.9801324503311258,
        ),
        (
            (
                "primary.hedges_g",
                "primary.hedges_g.value",
                "effect_size.hedges_g",
            ),
            ("hedges_g",),
            3.3775483697174717,
        ),
    ]
    if not all(
        math.isclose(metric(paths, aliases), target, rel_tol=1e-9, abs_tol=1e-6)
        for paths, aliases, target in expected
    ):
        return False
    return math.isclose(
        metric(
            (
                "primary.p_value",
                "primary.welch_t_test.p_value_two_sided",
                "welch_t_test.p_value",
                "welch_t_test.p_value_two_sided",
            ),
            ("p_value", "p_value_two_sided"),
        ),
        2.971749478841818e-13,
        rel_tol=1e-6,
        abs_tol=1e-18,
    )


def _reconciliation_delta(value: dict, *names: str) -> float:
    """Read a metric delta from supported reconciliation artifact shapes."""

    for collection_name in ("checks", "metrics", "differences", "primary_metrics"):
        checks = value.get(collection_name)
        if not isinstance(checks, dict):
            continue
        for name in names:
            check = checks.get(name)
            if isinstance(check, dict):
                delta = check.get("absolute_delta", check.get("absolute_difference"))
                if isinstance(delta, (int, float)) and not isinstance(delta, bool):
                    return float(delta)
    comparisons = value.get("comparisons")
    if isinstance(comparisons, list):
        requested = set(names)
        for check in comparisons:
            if not isinstance(check, dict) or check.get("metric") not in requested:
                continue
            delta = check.get("absolute_delta", check.get("absolute_difference"))
            if isinstance(delta, (int, float)) and not isinstance(delta, bool):
                return float(delta)
    for name in names:
        direct = value.get(f"abs_diff_{name}")
        if isinstance(direct, (int, float)) and not isinstance(direct, bool):
            return float(direct)
        nested = value.get(name)
        if isinstance(nested, dict):
            delta = nested.get("absolute_delta", nested.get("absolute_difference"))
            if isinstance(delta, (int, float)) and not isinstance(delta, bool):
                return float(delta)
    return 1e9


def _latest_reconciliation(
    computation: dict,
    generated: dict[str, object],
) -> dict | None:
    artifact_paths = [
        artifact.get("path", "")
        for record in computation.get("records", [])
        if record.get("status") == "succeeded"
        for artifact in record.get("artifacts", [])
        if "reconcil" in artifact.get("path", "").lower()
    ]
    for artifact_path in reversed(artifact_paths):
        for manifest_path, value in generated.items():
            if artifact_path.endswith(manifest_path) and isinstance(value, dict):
                return value
    return None


def score(
    case_name: str,
    case: dict,
    source: Path | None,
    detail: dict,
    client: Client,
    live_observation: dict | None = None,
) -> dict:
    checks = {}
    run_result = detail.get("result") or {}
    checks["terminal_status"] = detail["status"] not in {
        "queued",
        "running",
        "cancel_requested",
        "failed",
        "interrupted",
        "cancelled",
    }
    if not checks["terminal_status"]:
        if live_observation is not None:
            checks["live_events_observed"] = bool(live_observation["event_ids"])
            checks["qwen_output_streamed"] = "Qwen" in live_observation["stream_actors"]
            checks["gemma_output_streamed"] = (
                "Gemma" in live_observation["stream_actors"]
            )
        return {
            "checks": checks,
            "passed": False,
            "score": sum(checks.values()),
            "total": len(checks),
        }
    checks["deterministic_validation"] = bool(
        (run_result.get("deterministic_validation") or {}).get("passed")
    )
    review = run_result.get("scientific_review") or {}
    checks["independent_review"] = review.get("verdict") in {
        "pass",
        "pass_with_nonblocking_comments",
    }
    manifest_paths = {item["path"] for item in detail.get("artifacts", [])}
    checks["core_provenance"] = {
        "environment.json",
        "input_manifest.json",
        "protocol.json",
        "report.md",
    }.issubset(manifest_paths)
    computation = run_result.get("computation_evidence") or {}
    report = detail.get("report") or run_result.get("report") or {}
    checks["article_sections"] = bool(
        report.get("executive_summary")
        and report.get("introduction")
        and report.get("methods")
        and report.get("results")
        and report.get("discussion")
        and report.get("conclusions")
    )
    if live_observation is not None:
        checks["live_events_observed"] = bool(live_observation["event_ids"])
        checks["live_artifact_access"] = bool(
            live_observation["downloaded_while_active"]
        )
        checks["qwen_output_streamed"] = "Qwen" in live_observation["stream_actors"]
        checks["gemma_output_streamed"] = "Gemma" in live_observation["stream_actors"]
    if case["enable_code"]:
        languages = {
            record["language"]
            for record in computation.get("records", [])
            if record["status"] == "succeeded"
            and any(
                artifact.get("description") == "sandbox-generated analysis artifact"
                for artifact in record.get("artifacts", [])
            )
        }
        checks["python_and_r_succeeded"] = {"python", "r"}.issubset(languages)
        artifact_paths = {item["path"] for item in computation.get("artifacts", [])}
        sources = {
            item["source_id"]: item.get("artifact_path")
            for item in report.get("sources", [])
        }
        computed_claims = [
            claim
            for claim in report.get("claims", [])
            if claim["claim_type"] == "computed"
        ]
        checks["computed_claims_traceable"] = bool(computed_claims) and all(
            any(
                sources.get(source_id) in artifact_paths
                for source_id in claim["evidence_refs"]
            )
            for claim in computed_claims
        )
        generated_json = _artifact_jsons(client, detail["id"], manifest_paths)
        ancestor_id = detail.get("parent_run_id")
        seen_run_ids = {detail["id"]}
        while ancestor_id and ancestor_id not in seen_run_ids:
            seen_run_ids.add(ancestor_id)
            try:
                ancestor = client.request("GET", f"/api/runs/{ancestor_id}")
                ancestor_paths = {
                    item["path"] for item in ancestor.get("artifacts", [])
                }
                generated_json.update(
                    _artifact_jsons(client, ancestor_id, ancestor_paths)
                )
                ancestor_id = ancestor.get("parent_run_id")
            except (KeyError, OSError, RuntimeError):
                break
        generated_paths = {item["path"] for item in computation.get("artifacts", [])}
        if case_name == "known-effect":
            reconciliation = _latest_reconciliation(computation, generated_json)
            python_result = _language_result(
                computation, generated_json, "python", reconciliation
            )
            r_result = _language_result(
                computation, generated_json, "r", reconciliation
            )
            checks["planted_effect_recovered"] = bool(
                isinstance(python_result, dict)
                and isinstance(r_result, dict)
                and _known_effect_matches_reference(python_result)
                and _known_effect_matches_reference(r_result)
            )
            checks["effect_size_reconciled"] = bool(
                isinstance(python_result, dict)
                and isinstance(r_result, dict)
                and abs(
                    _metric(python_result, "hedges_g") - _metric(r_result, "hedges_g")
                )
                <= 1e-6
            )
            checks["cross_language_reconciled"] = bool(
                isinstance(reconciliation, dict)
                and _reconciliation_delta(
                    reconciliation,
                    "mean_diff",
                    "mean_difference",
                    "primary.point_estimate",
                )
                <= 1e-6
                and _reconciliation_delta(
                    reconciliation,
                    "t_stat",
                    "t_statistic",
                    "welch_t_statistic",
                    "primary.t_statistic",
                )
                <= 1e-6
                and _reconciliation_delta(
                    reconciliation, "hedges_g", "effect_size.hedges_g"
                )
                <= 1e-6
                and reconciliation.get("reconciliation_passed") is True
            )
            displays = (detail.get("display_manifest") or {}).get("displays", [])
            checks["figure_generated"] = any(
                path.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                for path in generated_paths
            )
            checks["figure_and_table_registered"] = any(
                item.get("kind") == "figure" for item in displays
            ) and any(item.get("kind") == "table" for item in displays)
            display_audit = (
                json.loads(
                    client.download_artifact(detail["id"], "gemma_display_audit.json")
                )
                if "gemma_display_audit.json" in manifest_paths
                else {}
            )
            checks["gemma_multimodal_display_audit"] = bool(
                display_audit.get("review_mode")
                == "raster_with_ocr_geometry_and_table_previews"
                and display_audit.get("review_source") == "gemma_multimodal_critic"
                and display_audit.get("critic_model")
                and display_audit.get("visual_critic") == "Gemma"
                and display_audit.get("qwen_image_inputs") == 0
                and display_audit.get("figure_text_inputs")
                and all(
                    item.get("ocr_available") is True
                    and item.get("ocr_character_count", 0) > 0
                    and item.get("ocr_text_sha256")
                    for item in display_audit["figure_text_inputs"]
                )
                and display_audit.get("table_previews")
                and display_audit.get("verdict")
                in {"pass", "pass_with_nonblocking_comments"}
            )
        elif case_name == "corrupted-input":
            evidence_text = json.dumps(
                {"report": report, "generated": generated_json},
                sort_keys=True,
            ).lower()
            checks["seeded_corruptions_detected"] = all(
                token in evidence_text for token in ("c10", "t05", "t08")
            )
            checks["decision_readiness_withheld"] = (
                any(
                    term in evidence_text
                    for term in ("not decision-ready", "not decision ready")
                )
                and "inconclusive" in evidence_text
                and "sensitivity" in evidence_text
            )
    else:
        retrieval = run_result.get("retrieval_evidence") or {}
        source_urls = {s.get("url") for s in report.get("sources", []) if s.get("url")}
        checks["retrieval_used"] = retrieval.get("successful_calls", 0) > 0
        checks["source_urls_observed"] = bool(source_urls) and source_urls.issubset(
            set(retrieval.get("urls", []))
        )
        if case_name == "retrieval-grounding":
            report_text = json.dumps(report, sort_keys=True).lower()
            checks["official_scipy_source"] = any(
                "docs.scipy.org" in url for url in source_urls
            )
            checks["official_r_source"] = any(
                domain in url
                for url in source_urls
                for domain in ("stat.ethz.ch", "r-project.org")
            )
            checks["welch_api_switches_reported"] = (
                "equal_var" in report_text and "var.equal" in report_text
            )
        elif case_name == "pubmed-fulltext":
            references = (detail.get("reference_manifest") or {}).get("references", [])
            article = next(
                (item for item in references if str(item.get("pmid")) == "42158852"),
                None,
            )
            checks["pubmed_identifiers_verified"] = bool(
                article
                and article.get("pmcid") == "PMC13180577"
                and str(article.get("doi", "")).lower() == "10.3389/fonc.2025.1657746"
            )
            markdown = article.get("markdown") if article else None
            pdf = article.get("pdf") if article else None
            markdown_bytes = (
                client.download_artifact(detail["id"], markdown["path"])
                if markdown and markdown.get("path")
                else b""
            )
            pdf_bytes = (
                client.download_artifact(detail["id"], pdf["path"])
                if pdf and pdf.get("path")
                else b""
            )
            checks["local_article_markdown_stored"] = bool(
                markdown_bytes
                and hashlib.sha256(markdown_bytes).hexdigest() == markdown.get("sha256")
                and b"42158852" in markdown_bytes
                and b"mutant kras" in markdown_bytes.lower()
            )
            checks["local_article_pdf_stored"] = bool(
                pdf_bytes
                and pdf_bytes.startswith(b"%PDF-")
                and hashlib.sha256(pdf_bytes).hexdigest() == pdf.get("sha256")
            )
            report_text = json.dumps(report, sort_keys=True).lower()
            checks["article_interpretation_grounded"] = (
                all(
                    phrase in report_text
                    for phrase in (
                        "45",
                        "11",
                        "7",
                        "2.57",
                        "0.94",
                        "7.04",
                        "3.13",
                        "1.18",
                        "8.29",
                        "prognostic",
                    )
                )
                and any(phrase in report_text for phrase in ("not causal", "causality"))
                and any(
                    phrase in report_text
                    for phrase in ("small cohort", "small sample", "limited sample")
                )
            )
            article_source_id = article.get("source_id") if article else None
            literature_claims = [
                claim
                for claim in report.get("claims", [])
                if claim.get("claim_type")
                in {"literature_supported", "inference", "observed"}
            ]
            checks["literature_claims_link_local_source"] = bool(
                article_source_id
                and literature_claims
                and all(
                    article_source_id in claim.get("evidence_refs", [])
                    for claim in literature_claims
                )
            )
            try:
                rendered = (
                    client.download_artifact(detail["id"], "report.md").decode(
                        "utf-8", errors="replace"
                    )
                    if "report.md" in manifest_paths
                    else ""
                )
            except (OSError, RuntimeError, urllib.error.HTTPError):
                rendered = ""
            checks["report_links_local_article"] = bool(
                article
                and markdown
                and pdf
                and markdown.get("path") in rendered
                and pdf.get("path") in rendered
            )
    if source is not None:
        checks["input_unchanged"] = (
            hashlib.sha256(source.read_bytes()).hexdigest() == case["source_sha256"]
        )
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "score": sum(checks.values()),
        "total": len(checks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument(
        "--cases", type=Path, default=Path(__file__).parent / "cases" / "cases.json"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-minutes", type=int, default=45)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    case = cases[args.case]
    env = parse_env(args.env_file)
    client = Client(
        env["SCIENTIFIC_AGENT_PUBLIC_URL"],
        env.get("WEB_USERNAME"),
        env.get("WEB_PASSWORD"),
    )
    workspace = client.request(
        "POST", "/api/workspaces", payload={"name": f"EVAL — {case['name']}"}
    )
    source = None
    if case.get("file"):
        source = args.cases.parent / case["file"]
        content = source.read_bytes()
        case["source_sha256"] = hashlib.sha256(content).hexdigest()
        body, content_type = multipart(source.name, content)
        client.request(
            "POST",
            f"/api/workspaces/{workspace['id']}/files",
            body=body,
            headers={"Content-Type": content_type},
        )
    run = client.request(
        "POST",
        f"/api/workspaces/{workspace['id']}/runs",
        payload={
            "objective": case["objective"],
            "enable_code": case["enable_code"],
            "mcp_servers": case["mcp_servers"],
        },
    )
    deadline = time.monotonic() + args.timeout_minutes * 60
    last_phase = None
    event_cursor = 0
    live_observation = {
        "event_ids": [],
        "stream_actors": set(),
        "downloaded_while_active": set(),
    }
    while time.monotonic() < deadline:
        result = client.request("GET", f"/api/runs/{run['id']}")
        events = client.request(
            "GET", f"/api/runs/{run['id']}/events?after_id={event_cursor}"
        )
        for event in events:
            event_cursor = max(event_cursor, event["id"])
            live_observation["event_ids"].append(event["id"])
            if event["event_type"] == "model_output_stream":
                live_observation["stream_actors"].add(event["actor"])
            artifact_path = event.get("artifact_path")
            if artifact_path and result["status"] in {
                "queued",
                "running",
                "cancel_requested",
            }:
                try:
                    if client.download_artifact(run["id"], artifact_path):
                        live_observation["downloaded_while_active"].add(artifact_path)
                except (OSError, RuntimeError, http.client.HTTPException):
                    pass
        if result["phase"] != last_phase:
            print(f"phase={result['phase']} status={result['status']}", flush=True)
            last_phase = result["phase"]
        if result["status"] not in {"queued", "running", "cancel_requested"}:
            break
        time.sleep(1)
    else:
        raise SystemExit("evaluation timed out")
    evaluation = score(
        args.case,
        case,
        source,
        result,
        client,
        live_observation,
    )
    payload = {
        "case": args.case,
        "workspace_id": workspace["id"],
        "run_id": run["id"],
        "evaluation": evaluation,
        "live_observation": {
            "event_ids": live_observation["event_ids"],
            "stream_actors": sorted(live_observation["stream_actors"]),
            "downloaded_while_active": sorted(
                live_observation["downloaded_while_active"]
            ),
        },
        "result": result,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {"case": args.case, "run_id": run["id"], **evaluation}, sort_keys=True
        )
    )
    raise SystemExit(0 if evaluation["passed"] else 2)


if __name__ == "__main__":
    main()
