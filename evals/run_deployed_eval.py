#!/usr/bin/env python3
"""Run and score a deployed Evidence Bench case through its browser API."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
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
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
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
            raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc

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
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="upload"; filename="{filename}"\r\n'
        "Content-Type: text/csv\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
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
            if isinstance(candidate, dict):
                for estimate_name in ("value", "estimate"):
                    estimate = candidate.get(estimate_name)
                    if isinstance(estimate, (int, float)) and not isinstance(
                        estimate, bool
                    ):
                        return float(estimate)
        pending.extend(item for item in current.values() if isinstance(item, dict))
    return 1e9


def _language_result(
    computation: dict,
    generated: dict[str, object],
    language: str,
) -> dict | None:
    artifact_paths = [
        artifact["path"]
        for record in computation.get("records", [])
        if record.get("language") == language and record.get("status") == "succeeded"
        for artifact in record.get("artifacts", [])
        if artifact.get("description") == "sandbox-generated analysis artifact"
    ]
    for manifest_path, value in generated.items():
        if not isinstance(value, dict):
            continue
        if not any(path.endswith(manifest_path) for path in artifact_paths):
            continue
        if _metric(value, "mean_difference", "mean_diff") != 1e9 and _metric(
            value, "hedges_g"
        ) != 1e9:
            return value
    return None


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


def score(case_name: str, case: dict, source: Path | None, detail: dict, client: Client) -> dict:
    checks = {}
    run_result = detail.get("result") or {}
    checks["terminal_status"] = detail["status"] not in {"queued", "running", "failed", "interrupted"}
    checks["deterministic_validation"] = bool(run_result.get("deterministic_validation", {}).get("passed"))
    review = run_result.get("scientific_review") or {}
    checks["independent_review"] = review.get("verdict") in {"pass", "pass_with_nonblocking_comments"}
    manifest_paths = {item["path"] for item in detail.get("artifacts", [])}
    checks["core_provenance"] = {
        "environment.json", "input_manifest.json", "protocol.json", "report.md"
    }.issubset(manifest_paths)
    computation = run_result.get("computation_evidence") or {}
    report = detail.get("report") or run_result.get("report") or {}
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
            any(sources.get(source_id) in artifact_paths for source_id in claim["evidence_refs"])
            for claim in computed_claims
        )
        generated_json = _artifact_jsons(client, detail["id"], manifest_paths)
        generated_paths = {
            item["path"] for item in computation.get("artifacts", [])
        }
        if case_name == "known-effect":
            python_result = _language_result(computation, generated_json, "python")
            r_result = _language_result(computation, generated_json, "r")
            reconciliation = next(
                (
                    value
                    for path, value in generated_json.items()
                    if "reconcil" in path.lower() and isinstance(value, dict)
                ),
                None,
            )
            checks["planted_effect_recovered"] = bool(
                isinstance(python_result, dict)
                and isinstance(r_result, dict)
                and abs(_metric(python_result, "mean_difference", "mean_diff") - 5.0)
                <= 1e-6
                and abs(_metric(r_result, "mean_difference", "mean_diff") - 5.0)
                <= 1e-6
            )
            checks["effect_size_reconciled"] = bool(
                isinstance(python_result, dict)
                and isinstance(r_result, dict)
                and abs(
                    _metric(python_result, "hedges_g")
                    - _metric(r_result, "hedges_g")
                )
                <= 1e-6
            )
            checks["cross_language_reconciled"] = bool(
                isinstance(reconciliation, dict)
                and _reconciliation_delta(
                    reconciliation, "mean_diff", "mean_difference"
                ) <= 1e-6
                and _reconciliation_delta(
                    reconciliation, "t_stat", "t_statistic", "welch_t_statistic"
                ) <= 1e-6
                and (
                    reconciliation.get("all_pass") is True
                    or reconciliation.get("reconciliation_passed") is True
                )
            )
            checks["figure_generated"] = any(
                path.lower().endswith((".png", ".svg", ".pdf"))
                for path in generated_paths
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
                any(term in evidence_text for term in ("not decision-ready", "not decision ready"))
                and "inconclusive" in evidence_text
                and "sensitivity" in evidence_text
            )
    else:
        retrieval = run_result.get("retrieval_evidence") or {}
        source_urls = {s.get("url") for s in report.get("sources", []) if s.get("url")}
        checks["retrieval_used"] = retrieval.get("successful_calls", 0) > 0
        checks["source_urls_observed"] = bool(source_urls) and source_urls.issubset(set(retrieval.get("urls", [])))
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
    if source is not None:
        checks["input_unchanged"] = hashlib.sha256(source.read_bytes()).hexdigest() == case["source_sha256"]
    return {"checks": checks, "passed": all(checks.values()), "score": sum(checks.values()), "total": len(checks)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path(__file__).parent / "cases" / "cases.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-minutes", type=int, default=45)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    case = cases[args.case]
    env = parse_env(args.env_file)
    client = Client(env["SCIENTIFIC_AGENT_PUBLIC_URL"], env["WEB_USERNAME"], env["WEB_PASSWORD"])
    workspace = client.request("POST", "/api/workspaces", payload={"name": f"EVAL — {case['name']}"})
    source = None
    if case.get("file"):
        source = args.cases.parent / case["file"]
        content = source.read_bytes()
        case["source_sha256"] = hashlib.sha256(content).hexdigest()
        body, content_type = multipart(source.name, content)
        client.request(
            "POST", f"/api/workspaces/{workspace['id']}/files", body=body,
            headers={"Content-Type": content_type},
        )
    run = client.request(
        "POST", f"/api/workspaces/{workspace['id']}/runs",
        payload={
            "objective": case["objective"],
            "enable_code": case["enable_code"],
            "mcp_servers": case["mcp_servers"],
        },
    )
    deadline = time.monotonic() + args.timeout_minutes * 60
    last_phase = None
    while time.monotonic() < deadline:
        result = client.request("GET", f"/api/runs/{run['id']}")
        if result["phase"] != last_phase:
            print(f"phase={result['phase']} status={result['status']}", flush=True)
            last_phase = result["phase"]
        if result["status"] not in {"queued", "running"}:
            break
        time.sleep(3)
    else:
        raise SystemExit("evaluation timed out")
    evaluation = score(args.case, case, source, result, client)
    payload = {"case": args.case, "workspace_id": workspace["id"], "run_id": run["id"], "evaluation": evaluation, "result": result}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"case": args.case, "run_id": run["id"], **evaluation}, sort_keys=True))
    raise SystemExit(0 if evaluation["passed"] else 2)


if __name__ == "__main__":
    main()
