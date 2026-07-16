import os
import threading
import uuid
from pathlib import Path

import pytest

from scientific_agent.config import SandboxSettings
from scientific_agent.sandbox_worker import (
    ExecuteRequest,
    FigureOcrRequest,
    PdfExtractRequest,
    ReleaseRequest,
    WorkerState,
    _worker_authorized,
)
from scientific_agent.schemas import ComputationEvidence


def _state(tmp_path: Path) -> tuple[WorkerState, Path, Path]:
    workspace_id = str(uuid.uuid4())
    workspace = tmp_path / "workspaces" / workspace_id / "files"
    root = tmp_path / "workspaces" / workspace_id / "runs" / "run-1" / "computations"
    workspace.mkdir(parents=True)
    root.mkdir(parents=True)
    state = WorkerState(tmp_path, "x" * 32, SandboxSettings())
    return state, workspace, root


def test_worker_auth_rejects_non_ascii_header_without_exception():
    assert _worker_authorized("Bearer token", "token")
    assert not _worker_authorized("Bearer tokén", "token")


def test_worker_accepts_only_matching_confined_workspace_and_run_paths(tmp_path):
    state, workspace, root = _state(tmp_path)
    request = ExecuteRequest(
        request_id=str(uuid.uuid4()),
        workspace=str(workspace),
        computation_root=str(root),
        language="python",
        code="print(1)",
        timeout_seconds=10,
    )
    assert state.confined_paths(request) == (workspace.resolve(), root.resolve())

    request.computation_root = str(tmp_path / "outside")
    with pytest.raises(ValueError):
        state.confined_paths(request)


def test_worker_uses_caller_attempt_budget_and_rejects_budget_changes(
    tmp_path, monkeypatch
):
    state, workspace, root = _state(tmp_path)
    observed = []

    def execute(self, language, code, timeout_seconds=120):
        del language, code, timeout_seconds
        observed.append(self.settings.max_calls_per_attempt)
        return {"status": "succeeded"}

    monkeypatch.setattr(
        "scientific_agent.sandbox_worker.AnalysisExecutor.execute", execute
    )
    monkeypatch.setattr(
        "scientific_agent.sandbox_worker.AnalysisExecutor.evidence",
        lambda self: ComputationEvidence(),
    )
    request = ExecuteRequest(
        request_id=str(uuid.uuid4()),
        workspace=str(workspace),
        computation_root=str(root),
        language="python",
        code="print(1)",
        timeout_seconds=10,
        max_calls_per_attempt=8,
    )

    state.execute(request)

    assert observed == [8]
    changed = request.model_copy(
        update={
            "request_id": str(uuid.uuid4()),
            "max_calls_per_attempt": 9,
        }
    )
    with pytest.raises(ValueError, match="budget cannot change"):
        state.execute(changed)


def test_worker_uses_root_owned_staging_for_history_bind(tmp_path, monkeypatch):
    state, workspace, root = _state(tmp_path)
    observed = {}

    def execute(self, language, code, timeout_seconds=120):
        del language, code, timeout_seconds
        observed["history_dir"] = self.history_dir
        observed["workspace"] = self.workspace
        return {"status": "succeeded"}

    monkeypatch.setattr(
        "scientific_agent.sandbox_worker.AnalysisExecutor.execute", execute
    )
    monkeypatch.setattr(
        "scientific_agent.sandbox_worker.AnalysisExecutor.evidence",
        lambda self: ComputationEvidence(),
    )
    request = ExecuteRequest(
        request_id=str(uuid.uuid4()),
        workspace=str(workspace),
        computation_root=str(root),
        language="python",
        code="print(1)",
        timeout_seconds=10,
    )

    state.execute(request)

    assert observed["history_dir"] == observed["workspace"].parent / "history"
    assert observed["history_dir"] != root.parent.resolve()
    assert observed["history_dir"].parent.name.startswith("evidence-worker-")


def test_worker_stages_prior_attempts_at_expected_history_paths(tmp_path, monkeypatch):
    state, workspace, base_root = _state(tmp_path)
    history = base_root.parent / "computations"
    current = history / "attempt-1"
    prior = history / "attempt-0" / "exec-001" / "output"
    prior.mkdir(parents=True)
    (prior / "result.json").write_text('{"estimate": 2.5}', encoding="utf-8")
    current.mkdir(parents=True)
    observed = {}

    def execute(self, language, code, timeout_seconds=120):
        del language, code, timeout_seconds
        prior_result = (
            self.history_dir / "attempt-0" / "exec-001" / "output" / "result.json"
        )
        observed["prior"] = prior_result.read_text(encoding="utf-8")
        observed["current_root"] = self.root.name
        return {"status": "succeeded"}

    monkeypatch.setattr(
        "scientific_agent.sandbox_worker.AnalysisExecutor.execute", execute
    )
    monkeypatch.setattr(
        "scientific_agent.sandbox_worker.AnalysisExecutor.evidence",
        lambda self: ComputationEvidence(),
    )
    request = ExecuteRequest(
        request_id=str(uuid.uuid4()),
        workspace=str(workspace),
        computation_root=str(current),
        language="python",
        code="print(1)",
        timeout_seconds=10,
    )

    state.execute(request)

    assert observed == {"prior": '{"estimate": 2.5}', "current_root": "attempt-1"}


def test_worker_call_budget_is_capped_by_worker_configuration(tmp_path, monkeypatch):
    state, workspace, root = _state(tmp_path)
    observed = []

    def execute(self, language, code, timeout_seconds=120):
        del language, code, timeout_seconds
        observed.append(self.settings.max_calls_per_attempt)
        return {"status": "succeeded"}

    monkeypatch.setattr(
        "scientific_agent.sandbox_worker.AnalysisExecutor.execute", execute
    )
    monkeypatch.setattr(
        "scientific_agent.sandbox_worker.AnalysisExecutor.evidence",
        lambda self: ComputationEvidence(),
    )
    request = ExecuteRequest(
        request_id=str(uuid.uuid4()),
        workspace=str(workspace),
        computation_root=str(root),
        language="python",
        code="print(1)",
        timeout_seconds=10,
        max_calls_per_attempt=80,
    )

    state.execute(request)

    assert observed == [state.settings.max_calls_per_attempt]


def test_worker_omitted_caller_budget_uses_worker_configuration(tmp_path, monkeypatch):
    state, workspace, root = _state(tmp_path)
    state.settings = SandboxSettings(max_calls_per_attempt=40)
    observed = []

    def execute(self, language, code, timeout_seconds=120):
        del language, code, timeout_seconds
        observed.append(self.settings.max_calls_per_attempt)
        return {"status": "succeeded"}

    monkeypatch.setattr(
        "scientific_agent.sandbox_worker.AnalysisExecutor.execute", execute
    )
    monkeypatch.setattr(
        "scientific_agent.sandbox_worker.AnalysisExecutor.evidence",
        lambda self: ComputationEvidence(),
    )
    request = ExecuteRequest(
        request_id=str(uuid.uuid4()),
        workspace=str(workspace),
        computation_root=str(root),
        language="python",
        code="print(1)",
        timeout_seconds=10,
    )

    state.execute(request)

    assert observed == [40]


def test_worker_rejects_a_run_path_belonging_to_another_workspace(tmp_path):
    state, workspace, _ = _state(tmp_path)
    other = tmp_path / "workspaces" / str(uuid.uuid4()) / "runs" / "run-2"
    other.mkdir(parents=True)
    request = ExecuteRequest(
        request_id=str(uuid.uuid4()),
        workspace=str(workspace),
        computation_root=str(other),
        language="r",
        code="print(1)",
        timeout_seconds=10,
    )
    with pytest.raises(ValueError, match="invalid workspace"):
        state.confined_paths(request)


def test_worker_stages_uploaded_files_after_literature_acquisition(tmp_path):
    state, workspace, _ = _state(tmp_path)
    uploaded = workspace / "known_effect.csv"
    uploaded.write_text("group,outcome\ncontrol,1\n", encoding="utf-8")
    reference = workspace / "references" / "markdown" / "article.md"
    reference.parent.mkdir(parents=True)
    reference.write_text("article", encoding="utf-8")
    staged = tmp_path / "staged"
    staged.mkdir()

    state._stage_workspace_inputs(workspace, staged)

    assert (staged / uploaded.name).read_bytes() == uploaded.read_bytes()
    assert not (staged / "references").exists()


def test_worker_rejects_symlinked_workspace_input(tmp_path):
    state, workspace, _ = _state(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_text("secret", encoding="utf-8")
    (workspace / "linked.csv").symlink_to(outside)
    staged = tmp_path / "staged"
    staged.mkdir()

    with pytest.raises(ValueError, match="symlinks"):
        state._stage_workspace_inputs(workspace, staged)


def test_worker_rejects_symlinked_workspace_directory(tmp_path):
    state, workspace, _ = _state(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "linked-directory").symlink_to(outside, target_is_directory=True)
    staged = tmp_path / "staged"
    staged.mkdir()

    with pytest.raises(ValueError, match="symlinks"):
        state._stage_workspace_inputs(workspace, staged)


def test_worker_rejects_special_workspace_file(tmp_path):
    state, workspace, _ = _state(tmp_path)
    os.mkfifo(workspace / "named-pipe")
    staged = tmp_path / "staged"
    staged.mkdir()

    with pytest.raises(ValueError, match="regular files"):
        state._stage_workspace_inputs(workspace, staged)


def test_handoff_tolerates_root_squashed_nfs_chown(tmp_path, monkeypatch):
    state, _, root = _state(tmp_path)
    artifact = root / "artifact.txt"
    artifact.write_text("result", encoding="utf-8")

    def denied(*args, **kwargs):
        del args, kwargs
        raise PermissionError("root squash")

    monkeypatch.setattr("scientific_agent.sandbox_worker.os.chown", denied)
    state._handoff_ownership(root)


def test_worker_cancel_sets_only_the_matching_request_event(tmp_path):
    state, _, _ = _state(tmp_path)
    request_id = str(uuid.uuid4())
    event = __import__("threading").Event()
    state.cancellation_events[request_id] = event

    assert state.cancel(request_id)
    assert event.is_set()
    assert not state.cancel(str(uuid.uuid4()))


def test_worker_release_removes_executor_state_and_staging_tree(tmp_path):
    state, workspace, root = _state(tmp_path)
    staging_root = tmp_path / "worker-staging" / "computations"
    staging_root.mkdir(parents=True)
    (staging_root / "retained.txt").write_text("temporary", encoding="utf-8")
    key = str(root.resolve())
    state.executors[key] = object()  # type: ignore[assignment]
    state.staging_roots[key] = staging_root
    state.executor_locks[key] = threading.Lock()
    request = ReleaseRequest(workspace=str(workspace), computation_root=str(root))

    assert state.release(request)
    assert key not in state.executors
    assert key not in state.staging_roots
    assert key not in state.executor_locks
    assert not staging_root.parent.exists()
    assert root.is_dir()
    assert not state.release(request)


def _pdf_request(path: Path) -> PdfExtractRequest:
    return PdfExtractRequest(request_id=str(uuid.uuid4()), pdf_path=str(path))


def _ocr_request(path: Path) -> FigureOcrRequest:
    return FigureOcrRequest(request_id=str(uuid.uuid4()), figure_path=str(path))


def test_worker_accepts_only_generated_run_figure(tmp_path):
    state, _, root = _state(tmp_path)
    figure = root / "attempt-0" / "exec-001" / "output" / "figures" / "effect.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"\x89PNG\r\n\x1a\nfixture")

    assert state.confined_figure_path(_ocr_request(figure)) == figure.resolve()

    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    with pytest.raises(ValueError, match="figure path"):
        state.confined_figure_path(_ocr_request(outside))


def test_worker_rejects_generated_figure_symlink(tmp_path):
    state, _, root = _state(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    figure = root / "attempt-0" / "exec-001" / "output" / "figures" / "effect.png"
    figure.parent.mkdir(parents=True)
    figure.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        state.confined_figure_path(_ocr_request(figure))


def test_worker_accepts_only_direct_uuid_workspace_reference_pdf(tmp_path):
    state, workspace, _ = _state(tmp_path)
    pdf = workspace / "references" / "pdfs" / "article.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.7\nfixture")

    assert state.confined_pdf_path(_pdf_request(pdf)) == pdf.resolve()


@pytest.mark.parametrize(
    "relative",
    [
        "article.pdf",
        "references/article.pdf",
        "references/pdfs/nested/article.pdf",
        "references/pdfs/article.txt",
    ],
)
def test_worker_rejects_pdf_outside_exact_reference_directory(tmp_path, relative):
    state, workspace, _ = _state(tmp_path)
    candidate = workspace / relative
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"%PDF-1.7\nfixture")

    with pytest.raises(ValueError, match="workspace PDF path"):
        state.confined_pdf_path(_pdf_request(candidate))


def test_worker_rejects_pdf_symlink_and_outside_path(tmp_path):
    state, workspace, _ = _state(tmp_path)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-1.7\nfixture")
    link = workspace / "references" / "pdfs" / "article.pdf"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        state.confined_pdf_path(_pdf_request(link))
    with pytest.raises(ValueError, match="workspace PDF path|data directory"):
        state.confined_pdf_path(_pdf_request(outside))


def test_pdf_parser_command_has_one_input_one_output_and_no_network(
    tmp_path, monkeypatch
):
    state, workspace, _ = _state(tmp_path)
    pdf = workspace / "references" / "pdfs" / "article.pdf"
    output = tmp_path / "parser-output"
    pdf.parent.mkdir(parents=True)
    output.mkdir()
    pdf.write_bytes(b"%PDF-1.7\nfixture")

    original_exists = Path.exists

    def runtime_exists(path: Path) -> bool:
        if path in {Path("/usr/bin/bwrap"), Path("/usr/bin/pdftotext")}:
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", runtime_exists)

    command = state._pdf_bwrap_command(pdf.resolve(), output.resolve())
    sandbox = command[command.index("--") + 1 :]

    assert "--unshare-all" in sandbox
    assert [str(pdf.resolve()), "/input/article.pdf"] == sandbox[
        sandbox.index(str(pdf.resolve())) : sandbox.index(str(pdf.resolve())) + 2
    ]
    assert [str(output.resolve()), "/output"] == sandbox[
        sandbox.index(str(output.resolve())) : sandbox.index(str(output.resolve())) + 2
    ]
    assert "--dev" not in sandbox
    assert sandbox.count("--dev-bind") == 1
    assert "/dev/null" in sandbox
    assert str(state.data_dir.resolve()) not in sandbox
    assert ["--ro-bind", "/usr", "/usr"] not in [
        sandbox[index : index + 3] for index in range(max(0, len(sandbox) - 2))
    ]


def test_pdf_extraction_stages_private_workspace_input(tmp_path, monkeypatch):
    state, workspace, _ = _state(tmp_path)
    pdf = workspace / "references" / "pdfs" / "article.pdf"
    pdf.parent.mkdir(parents=True)
    payload = b"%PDF-1.7\nprivate-workspace-fixture"
    pdf.write_bytes(payload)
    observed = {}

    def command(parser_input, output_dir):
        observed["path"] = parser_input
        observed["bytes"] = parser_input.read_bytes()
        observed["mode"] = parser_input.stat().st_mode & 0o777
        assert parser_input != pdf.resolve()
        (output_dir / "article.txt").write_text(
            "Controller-staged article text", encoding="utf-8"
        )
        return ["/bin/true"]

    monkeypatch.setattr(state, "_pdf_bwrap_command", command)

    result = state.extract_pdf_text(_pdf_request(pdf))

    assert observed["bytes"] == payload
    assert observed["mode"] == 0o600
    assert result["text"] == "Controller-staged article text"
    assert observed["path"].name == "input.pdf"
