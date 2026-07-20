import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from http.server import ThreadingHTTPServer

import httpx
import pytest

from scientific_agent.config import EnvironmentSettings
from scientific_agent.environment import (
    EnvironmentManager,
    cleanup_workspace_environment,
    validate_python_packages,
    validate_r_packages,
)
from scientific_agent.environment_worker import (
    EnvironmentWorkerState,
    InstallRequest,
    _handler,
)


@contextmanager
def _package_server(state):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_registry_requirements_allow_versions_but_deny_urls_flags_and_bad_r_names():
    assert validate_python_packages(["polars>=1.0", "scanpy[leiden]"], 4) == [
        "polars>=1.0",
        "scanpy[leiden]",
    ]
    for value in (
        "--index-url=https://example.test",
        "pkg @ https://example.test/pkg.whl",
    ):
        with pytest.raises(ValueError):
            validate_python_packages([value], 4)
    assert validate_r_packages(["DESeq2", "data.table"], 4) == [
        "DESeq2",
        "data.table",
    ]
    with pytest.raises(ValueError):
        validate_r_packages(["github/user/repo"], 4)


def test_python_inventory_must_satisfy_requested_version():
    inventory = [{"name": "polars", "version": "1.0.0"}]
    assert (
        EnvironmentWorkerState._missing_requested("python", ["polars>=1"], inventory)
        == []
    )
    assert EnvironmentWorkerState._missing_requested(
        "python", ["polars>=2"], inventory
    ) == ["polars>=2"]


def test_worker_rejects_escaping_symlinks_and_special_files(tmp_path):
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "safe.txt").write_text("safe", encoding="utf-8")
    (packages / "escape").symlink_to("/etc/passwd")
    os.mkfifo(packages / "pipe")

    assert sorted(EnvironmentWorkerState._unsafe_entries(packages)) == [
        "escape",
        "pipe",
    ]


def test_package_install_subprocess_receives_only_configured_proxy(
    tmp_path, monkeypatch
):
    state = EnvironmentWorkerState(tmp_path / "environments", "x" * 32)
    request = InstallRequest(
        request_id=str(uuid.uuid4()),
        workspace_id=str(uuid.uuid4()),
        language="python",
        repository="pypi",
        packages=["polars"],
        timeout_seconds=30,
    )
    monkeypatch.setenv(
        "SCIENTIFIC_AGENT_PACKAGE_PROXY_URL", "http://browser-egress:3128"
    )

    command, outer_environment = state._command(
        request,
        ["polars"],
        tmp_path / "packages",
        tmp_path / "temporary",
    )
    command_text = " ".join(command)

    assert outer_environment == {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert f"--setenv {name} http://browser-egress:3128" in command_text
    assert "--setenv NO_PROXY  " in command_text


def test_r_package_installs_are_serial_and_failed_staging_is_not_reported_active(
    tmp_path, monkeypatch
):
    state = EnvironmentWorkerState(tmp_path / "environments", "x" * 32)
    workspace_id = str(uuid.uuid4())
    request = InstallRequest(
        request_id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        language="r",
        repository="cran",
        packages=["needed", "missing"],
        timeout_seconds=30,
    )
    command, _ = state._command(
        request,
        request.packages,
        tmp_path / "packages",
        tmp_path / "temporary",
    )
    assert "Ncpus=1" in " ".join(command)

    monkeypatch.setattr(
        state,
        "_command",
        lambda request, packages, package_dir, temporary: (
            ["/bin/sh", "-c", "true"],
            {"PATH": "/usr/bin:/bin"},
        ),
    )
    monkeypatch.setattr(
        state,
        "_inventory",
        lambda language, path: [{"name": "needed", "version": "1.0.0"}],
    )
    result = state.install(request)

    assert result["status"] == "failed"
    assert result["installed"] == []
    assert "No packages were activated" in result["stderr"]
    assert not (state.workspace_root(workspace_id, create=False) / "r").exists()


def test_package_proxy_rejects_credentials(tmp_path, monkeypatch):
    state = EnvironmentWorkerState(tmp_path / "environments", "x" * 32)
    request = InstallRequest(
        request_id=str(uuid.uuid4()),
        workspace_id=str(uuid.uuid4()),
        language="python",
        repository="pypi",
        packages=["polars"],
        timeout_seconds=30,
    )
    monkeypatch.setenv(
        "SCIENTIFIC_AGENT_PACKAGE_PROXY_URL", "http://user:secret@proxy:3128"
    )

    with pytest.raises(ValueError, match="unauthenticated HTTP origin"):
        state._command(
            request,
            ["polars"],
            tmp_path / "packages",
            tmp_path / "temporary",
        )


def test_worker_confines_uuid_workspace_and_transactionally_commits(
    tmp_path, monkeypatch
):
    state = EnvironmentWorkerState(tmp_path / "environments", "x" * 32)
    workspace_id = str(uuid.uuid4())

    def command(request, packages, staging, temporary):
        del request, packages, temporary
        return ["/bin/sh", "-c", f"touch {staging / 'installed'}"], {
            "PATH": "/usr/bin:/bin"
        }

    monkeypatch.setattr(state, "_command", command)
    monkeypatch.setattr(
        state,
        "_inventory",
        lambda language, path: [{"name": "polars", "version": "1.0.0"}],
    )
    result = state.install(
        InstallRequest(
            request_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            language="python",
            repository="pypi",
            packages=["polars"],
            timeout_seconds=30,
        )
    )
    assert result["status"] == "succeeded"
    generation = tmp_path / "environments" / workspace_id / "python"
    assert generation.is_symlink()
    assert (generation / "packages" / "installed").is_file()
    assert (generation / "lock.json").is_file()
    lock = __import__("json").loads((generation / "lock.json").read_text())
    assert lock["package_tree_sha256"]
    assert lock["package_tree_bytes"] == 0
    assert lock["package_tree_entries"] == 1

    with pytest.raises(ValueError, match="invalid workspace ID"):
        state.workspace_root("../escape")


def test_worker_reuses_satisfying_locked_generation_without_copy_or_quota_scan(
    tmp_path, monkeypatch
):
    state = EnvironmentWorkerState(tmp_path / "environments", "x" * 32)
    workspace_id = str(uuid.uuid4())

    def command(request, packages, staging, temporary):
        del request, packages, temporary
        return ["/bin/sh", "-c", f"touch {staging / 'installed'}"], {
            "PATH": "/usr/bin:/bin"
        }

    monkeypatch.setattr(state, "_command", command)
    monkeypatch.setattr(
        state,
        "_inventory",
        lambda language, path: [{"name": "polars", "version": "1.0.0"}],
    )
    first = state.install(
        InstallRequest(
            request_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            language="python",
            repository="pypi",
            packages=["polars"],
            timeout_seconds=30,
        )
    )

    def must_not_repeat_work(*args, **kwargs):
        raise AssertionError("satisfied generation must be reused without scanning")

    monkeypatch.setattr(state, "_command", must_not_repeat_work)
    monkeypatch.setattr(state, "_cumulative_quota_failure", must_not_repeat_work)
    second = state.install(
        InstallRequest(
            request_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            language="python",
            repository="pypi",
            packages=["polars>=0.5"],
            timeout_seconds=30,
        )
    )

    generations = (tmp_path / "environments" / workspace_id / ".generations").iterdir()
    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert second["reused"] is True
    assert second["generation"] == first["generation"]
    assert [path.name for path in generations] == [first["generation"]]


def test_worker_cancels_during_cumulative_quota_scan(tmp_path, monkeypatch):
    state = EnvironmentWorkerState(tmp_path / "environments", "x" * 32)
    request_id = str(uuid.uuid4())
    started = threading.Event()

    def cancellable_usage(root, cancellation=None, **kwargs):
        del root, kwargs
        assert cancellation is not None
        started.set()
        while not cancellation.wait(0.01):
            pass
        raise InterruptedError("cancelled during quota scan")

    monkeypatch.setattr(state, "_directory_usage", cancellable_usage)
    results = []
    thread = threading.Thread(
        target=lambda: results.append(
            state.install(
                InstallRequest(
                    request_id=request_id,
                    workspace_id=str(uuid.uuid4()),
                    language="python",
                    repository="pypi",
                    packages=["polars"],
                    timeout_seconds=30,
                )
            )
        )
    )
    thread.start()
    assert started.wait(1)
    assert state.cancel(request_id) is True
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert results == [
        {
            "status": "cancelled",
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "installed": [],
        }
    ]


def test_worker_denies_update_before_copy_when_workspace_quota_is_insufficient(
    tmp_path, monkeypatch
):
    state = EnvironmentWorkerState(tmp_path / "environments", "x" * 32)
    workspace_id = str(uuid.uuid4())
    root = state.workspace_root(workspace_id)
    generations = state._generation_root(root)
    previous = generations / "python-existing"
    packages = previous / "packages"
    packages.mkdir(parents=True)
    (packages / "payload.bin").write_bytes(b"12345")
    (previous / "lock.json").write_text("{}\n", encoding="utf-8")
    (root / "python").symlink_to(".generations/python-existing")
    state.max_workspace_bytes = state._directory_bytes(root) + 4

    def must_not_install(*args, **kwargs):
        raise AssertionError("quota denial must happen before copying or installing")

    monkeypatch.setattr(state, "_command", must_not_install)
    result = state.install(
        InstallRequest(
            request_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            language="python",
            repository="pypi",
            packages=["polars"],
            timeout_seconds=30,
        )
    )

    assert result["status"] == "failed"
    assert "Workspace package-generation quota" in result["stderr"]
    assert [path.name for path in generations.iterdir()] == ["python-existing"]


def test_worker_removes_staging_generation_when_post_install_quota_fails(
    tmp_path, monkeypatch
):
    state = EnvironmentWorkerState(
        tmp_path / "environments",
        "x" * 32,
        max_environment_bytes=1024,
        max_workspace_bytes=4,
        max_total_bytes=1024,
    )
    workspace_id = str(uuid.uuid4())

    def command(request, packages, package_dir, temporary):
        del request, packages, temporary
        return ["/bin/sh", "-c", f"printf 123 > {package_dir / 'payload.bin'}"], {
            "PATH": "/usr/bin:/bin"
        }

    monkeypatch.setattr(state, "_command", command)
    monkeypatch.setattr(
        state,
        "_inventory",
        lambda language, path: [{"name": "polars", "version": "1.0.0"}],
    )
    result = state.install(
        InstallRequest(
            request_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            language="python",
            repository="pypi",
            packages=["polars"],
            timeout_seconds=30,
        )
    )

    root = state.workspace_root(workspace_id, create=False)
    assert result["status"] == "failed"
    assert "Workspace package-generation quota" in result["stderr"]
    assert not (root / "python").exists()
    assert list((root / ".generations").iterdir()) == []


def test_worker_removes_staging_generation_when_single_generation_is_too_large(
    tmp_path, monkeypatch
):
    state = EnvironmentWorkerState(
        tmp_path / "environments",
        "x" * 32,
        max_environment_bytes=2,
        max_workspace_bytes=1024,
        max_total_bytes=1024,
    )
    workspace_id = str(uuid.uuid4())

    def command(request, packages, package_dir, temporary):
        del request, packages, temporary
        return ["/bin/sh", "-c", f"printf 123 > {package_dir / 'payload.bin'}"], {
            "PATH": "/usr/bin:/bin"
        }

    monkeypatch.setattr(state, "_command", command)
    monkeypatch.setattr(
        state,
        "_inventory",
        lambda language, path: [{"name": "polars", "version": "1.0.0"}],
    )
    result = state.install(
        InstallRequest(
            request_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            language="python",
            repository="pypi",
            packages=["polars"],
            timeout_seconds=30,
        )
    )

    root = state.workspace_root(workspace_id, create=False)
    assert result["status"] == "failed"
    assert "per-generation quota" in result["stderr"]
    assert "active allowance is 2 bytes" in result["stderr"]
    assert not (root / "python").exists()
    assert list((root / ".generations").iterdir()) == []


@pytest.mark.parametrize(
    ("quota_overrides", "expected_scope"),
    [
        ({"max_environment_bytes": 4096}, "per-generation"),
        ({"max_workspace_bytes": 4096}, "workspace"),
        ({"max_total_bytes": 4096}, "global"),
    ],
)
def test_active_quota_monitor_kills_long_running_writer_and_removes_staging(
    tmp_path, monkeypatch, quota_overrides, expected_scope
):
    limits = {
        "max_environment_bytes": 1024 * 1024,
        "max_workspace_bytes": 1024 * 1024,
        "max_total_bytes": 1024 * 1024,
        **quota_overrides,
    }
    state = EnvironmentWorkerState(
        tmp_path / "environments",
        "x" * 32,
        **limits,
    )
    workspace_id = str(uuid.uuid4())
    writer_completed = tmp_path / f"{expected_scope}-writer-completed"
    if expected_scope == "workspace":
        retained = state.workspace_root(workspace_id)
        (retained / "retained.bin").write_bytes(b"r" * 3072)
    elif expected_scope == "global":
        retained = state.workspace_root(str(uuid.uuid4()))
        (retained / "retained.bin").write_bytes(b"r" * 3072)

    def command(request, packages, package_dir, temporary):
        del request, packages, temporary
        script = (
            "import pathlib,time; "
            f"pathlib.Path({str(package_dir / 'payload.bin')!r}).write_bytes(b'x' * 65536); "
            "time.sleep(30); "
            f"pathlib.Path({str(writer_completed)!r}).touch()"
        )
        return [sys.executable, "-c", script], {"PATH": "/usr/bin:/bin"}

    monkeypatch.setattr(state, "_command", command)
    started_at = time.monotonic()
    result = state.install(
        InstallRequest(
            request_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            language="python",
            repository="pypi",
            packages=["polars"],
            timeout_seconds=30,
        )
    )

    root = state.workspace_root(workspace_id, create=False)
    assert time.monotonic() - started_at < 5
    assert result["status"] == "failed"
    assert (
        f"Active package installation exceeded {expected_scope} quota"
        in result["stderr"]
    )
    assert not writer_completed.exists()
    assert not (root / "python").exists()
    assert list((root / ".generations").iterdir()) == []


@pytest.mark.parametrize(
    ("quota_overrides", "expected_scope"),
    [
        ({"max_environment_entries": 2}, "per-generation"),
        ({"max_workspace_entries": 5}, "workspace"),
        ({"max_total_entries": 6}, "global"),
    ],
)
def test_active_entry_quota_kills_empty_file_flood_and_removes_staging(
    tmp_path, monkeypatch, quota_overrides, expected_scope
):
    limits = {
        "max_environment_entries": 10_000,
        "max_workspace_entries": 10_000,
        "max_total_entries": 10_000,
        **quota_overrides,
    }
    state = EnvironmentWorkerState(
        tmp_path / "environments",
        "x" * 32,
        **limits,
    )
    workspace_id = str(uuid.uuid4())
    writer_completed = tmp_path / f"{expected_scope}-entry-writer-completed"

    def command(request, packages, package_dir, temporary):
        del request, packages, temporary
        script = (
            "import pathlib,time; "
            f"root=pathlib.Path({str(package_dir)!r}); "
            "[(root / f'empty-{index}').touch() for index in range(64)]; "
            "time.sleep(30); "
            f"pathlib.Path({str(writer_completed)!r}).touch()"
        )
        return [sys.executable, "-c", script], {"PATH": "/usr/bin:/bin"}

    monkeypatch.setattr(state, "_command", command)
    started_at = time.monotonic()
    result = state.install(
        InstallRequest(
            request_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            language="python",
            repository="pypi",
            packages=["polars"],
            timeout_seconds=30,
        )
    )

    root = state.workspace_root(workspace_id, create=False)
    assert time.monotonic() - started_at < 5
    assert result["status"] == "failed"
    assert (
        f"Active package installation exceeded {expected_scope} entry quota"
        in result["stderr"]
    )
    assert not writer_completed.exists()
    assert not (root / "python").exists()
    assert list((root / ".generations").iterdir()) == []


@pytest.mark.parametrize(
    "create_entries",
    [
        "[(root / f'empty-{index}').touch() for index in range(3)]",
        "[(root / f'directory-{index}').mkdir() for index in range(3)]",
    ],
    ids=["regular-files", "directories"],
)
def test_entry_quota_is_rechecked_before_inventory_after_fast_install(
    tmp_path, monkeypatch, create_entries
):
    state = EnvironmentWorkerState(
        tmp_path / "environments",
        "x" * 32,
        max_environment_entries=2,
    )
    workspace_id = str(uuid.uuid4())

    def command(request, packages, package_dir, temporary):
        del request, packages, temporary
        script = (
            "import pathlib; "
            f"root=pathlib.Path({str(package_dir)!r}); " + create_entries
        )
        return [sys.executable, "-c", script], {"PATH": "/usr/bin:/bin"}

    monkeypatch.setattr(state, "_command", command)

    def must_not_inventory(*args, **kwargs):
        raise AssertionError("entry quota must be checked before inventory")

    monkeypatch.setattr(state, "_inventory", must_not_inventory)
    result = state.install(
        InstallRequest(
            request_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            language="python",
            repository="pypi",
            packages=["polars"],
            timeout_seconds=30,
        )
    )

    root = state.workspace_root(workspace_id, create=False)
    assert result["status"] == "failed"
    assert "per-generation entry quota" in result["stderr"]
    assert not (root / "python").exists()
    assert list((root / ".generations").iterdir()) == []


def test_global_entry_quota_denies_new_workspace_before_creating_metadata(
    tmp_path, monkeypatch
):
    state = EnvironmentWorkerState(
        tmp_path / "environments",
        "x" * 32,
        max_total_entries=1,
    )
    workspace_id = str(uuid.uuid4())

    def must_not_install(*args, **kwargs):
        raise AssertionError("quota denial must happen before creating metadata")

    monkeypatch.setattr(state, "_command", must_not_install)
    result = state.install(
        InstallRequest(
            request_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            language="python",
            repository="pypi",
            packages=["polars"],
            timeout_seconds=30,
        )
    )

    assert result["status"] == "failed"
    assert "Global package-entry quota" in result["stderr"]
    assert not (state.environments_dir / workspace_id).exists()


def test_global_quota_includes_lock_file_and_preserves_other_workspace(
    tmp_path, monkeypatch
):
    state = EnvironmentWorkerState(tmp_path / "environments", "x" * 32)
    retained_id = str(uuid.uuid4())
    target_id = str(uuid.uuid4())
    retained = state.workspace_root(retained_id)
    (retained / "keep.bin").write_bytes(b"retained")
    target = state.workspace_root(target_id)
    state.max_total_bytes = state._directory_bytes(state.environments_dir) + 3

    def command(request, packages, package_dir, temporary):
        del request, packages, temporary
        return ["/bin/sh", "-c", f"printf 123 > {package_dir / 'payload.bin'}"], {
            "PATH": "/usr/bin:/bin"
        }

    monkeypatch.setattr(state, "_command", command)
    monkeypatch.setattr(
        state,
        "_inventory",
        lambda language, path: [{"name": "polars", "version": "1.0.0"}],
    )
    result = state.install(
        InstallRequest(
            request_id=str(uuid.uuid4()),
            workspace_id=target_id,
            language="python",
            repository="pypi",
            packages=["polars"],
            timeout_seconds=30,
        )
    )

    assert result["status"] == "failed"
    assert "Global package-environment quota" in result["stderr"]
    assert list((target / ".generations").iterdir()) == []
    assert (retained / "keep.bin").read_bytes() == b"retained"


def test_quota_admission_serializes_concurrent_workspace_installs(
    tmp_path, monkeypatch
):
    state = EnvironmentWorkerState(tmp_path / "environments", "x" * 32)
    active = 0
    peak_active = 0
    activity_lock = threading.Lock()

    def command(request, packages, package_dir, temporary):
        del request, packages, temporary
        script = (
            "import pathlib,time; "
            f"pathlib.Path({str(package_dir / 'payload.bin')!r}).write_bytes(b'x'); "
            "time.sleep(0.15)"
        )
        return [sys.executable, "-c", script], {"PATH": "/usr/bin:/bin"}

    original_command = command

    def observed_command(*args, **kwargs):
        nonlocal active, peak_active
        with activity_lock:
            active += 1
            peak_active = max(peak_active, active)
        command_value = original_command(*args, **kwargs)
        # The command call itself is inside the serialized quota section. The
        # process sleep keeps the first install there while the second arrives.
        time.sleep(0.1)
        with activity_lock:
            active -= 1
        return command_value

    monkeypatch.setattr(state, "_command", observed_command)
    monkeypatch.setattr(
        state,
        "_inventory",
        lambda language, path: [{"name": "polars", "version": "1.0.0"}],
    )
    results = []

    def install(workspace_id):
        results.append(
            state.install(
                InstallRequest(
                    request_id=str(uuid.uuid4()),
                    workspace_id=workspace_id,
                    language="python",
                    repository="pypi",
                    packages=["polars"],
                    timeout_seconds=30,
                )
            )
        )

    threads = [
        threading.Thread(target=install, args=(str(uuid.uuid4()),)) for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert [result["status"] for result in results] == ["succeeded", "succeeded"]
    assert peak_active == 1


def test_cleanup_rejects_workspace_symlink_without_deleting_sibling(tmp_path):
    state = EnvironmentWorkerState(tmp_path / "environments", "x" * 32)
    target_id = str(uuid.uuid4())
    retained_id = str(uuid.uuid4())
    retained = state.workspace_root(retained_id)
    (retained / "keep.txt").write_text("keep", encoding="utf-8")
    (state.environments_dir / target_id).symlink_to(retained)

    with pytest.raises(ValueError, match="invalid workspace environment path"):
        state.cleanup(target_id)

    assert (retained / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_authenticated_cleanup_endpoint_removes_only_requested_workspace(tmp_path):
    token = "x" * 32
    state = EnvironmentWorkerState(tmp_path / "environments", token)
    target_id = str(uuid.uuid4())
    retained_id = str(uuid.uuid4())
    target = state.workspace_root(target_id)
    retained = state.workspace_root(retained_id)
    (target / ".generations" / "python-a").mkdir(parents=True)
    (target / ".generations" / "r-a").mkdir()
    (target / ".generations" / "python-a" / "package.bin").write_bytes(b"python")
    (target / ".generations" / "r-a" / "package.bin").write_bytes(b"r")
    (retained / "keep.txt").write_text("keep", encoding="utf-8")

    with _package_server(state) as worker_url:
        denied = httpx.post(
            f"{worker_url}/cleanup",
            json={"workspace_id": target_id},
            timeout=5,
        )
        assert denied.status_code == 401
        assert target.is_dir()

        deleted = httpx.post(
            f"{worker_url}/cleanup",
            headers={"Authorization": f"Bearer {token}"},
            json={"workspace_id": target_id},
            timeout=5,
        )

    assert deleted.status_code == 200
    assert deleted.json() == {
        "status": "deleted",
        "workspace_id": target_id,
        "removed_bytes": 7,
    }
    assert not target.exists()
    assert (retained / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_package_worker_health_is_constant_time_and_does_not_scan_storage(
    tmp_path, monkeypatch
):
    state = EnvironmentWorkerState(tmp_path / "environments", "x" * 32)

    def must_not_scan(*args, **kwargs):
        raise AssertionError("health endpoint must not scan package storage")

    monkeypatch.setattr(state, "_directory_usage", must_not_scan)
    with _package_server(state) as worker_url:
        response = httpx.get(f"{worker_url}/healthz", timeout=5)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cleanup_client_fails_closed_on_transport_and_invalid_worker_response(
    monkeypatch,
):
    workspace_id = str(uuid.uuid4())
    settings = replace(
        EnvironmentSettings(),
        worker_url="http://packages:8091",
        worker_token="x" * 32,
    )

    def disconnected(*args, **kwargs):
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr("scientific_agent.environment.httpx.post", disconnected)
    with pytest.raises(RuntimeError, match="cleanup request failed"):
        cleanup_workspace_environment(settings, workspace_id)

    class InvalidResponse:
        is_error = False

        @staticmethod
        def json():
            return {"status": "deleted", "workspace_id": "wrong", "removed_bytes": 0}

    monkeypatch.setattr(
        "scientific_agent.environment.httpx.post",
        lambda *args, **kwargs: InvalidResponse(),
    )
    with pytest.raises(RuntimeError, match="invalid cleanup response"):
        cleanup_workspace_environment(settings, workspace_id)


def test_manager_records_exact_worker_inventory(tmp_path, monkeypatch):
    workspace_id = str(uuid.uuid4())
    workspace = tmp_path / "data" / "workspaces" / workspace_id / "files"
    workspace.mkdir(parents=True)
    evidence = tmp_path / "run" / "package_installations.jsonl"

    class Response:
        is_error = False

        @staticmethod
        def json():
            return {
                "status": "succeeded",
                "installed": [{"name": "polars", "version": "1.0.0"}],
            }

    monkeypatch.setattr(
        "scientific_agent.environment.httpx.post", lambda *a, **k: Response()
    )
    settings = replace(
        EnvironmentSettings(), worker_url="http://packages:8091", worker_token="x" * 32
    )
    result = EnvironmentManager(workspace, settings, evidence).install(
        "python", ["polars"], "pypi"
    )
    assert result["installed"] == [{"name": "polars", "version": "1.0.0"}]
    assert '"version":"1.0.0"' in evidence.read_text(encoding="utf-8")


def test_package_worker_cancels_process_group_without_committing_generation(
    tmp_path, monkeypatch
):
    state = EnvironmentWorkerState(tmp_path / "environments", "x" * 32)
    request_id = str(uuid.uuid4())
    workspace_id = str(uuid.uuid4())

    def command(request, packages, package_dir, temporary):
        del request, packages, temporary
        return [
            "/bin/sh",
            "-c",
            f"touch {package_dir / 'started'}; sleep 30; touch {package_dir / 'late'}",
        ], {"PATH": "/usr/bin:/bin"}

    monkeypatch.setattr(state, "_command", command)
    result = {}
    thread = threading.Thread(
        target=lambda: result.update(
            state.install(
                InstallRequest(
                    request_id=request_id,
                    workspace_id=workspace_id,
                    language="python",
                    repository="pypi",
                    packages=["polars"],
                    timeout_seconds=30,
                )
            )
        )
    )
    thread.start()
    started = tmp_path / "environments" / workspace_id / ".generations"
    for _ in range(200):
        if started.exists() and any(started.rglob("started")):
            break
        time.sleep(0.01)
    else:
        pytest.fail("package subprocess did not start")

    before = time.monotonic()
    assert state.cancel(request_id)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert time.monotonic() - before < 5
    assert result["status"] == "cancelled"
    assert not (tmp_path / "environments" / workspace_id / "python").exists()
    assert request_id not in state.cancellation_events


def test_environment_manager_forwards_cancellation_to_matching_request(
    tmp_path, monkeypatch
):
    workspace_id = str(uuid.uuid4())
    workspace = tmp_path / "data" / "workspaces" / workspace_id / "files"
    workspace.mkdir(parents=True)
    cancellation = threading.Event()
    install_started = threading.Event()
    cancel_received = threading.Event()
    request_ids = {}

    class Response:
        is_error = False
        status_code = 200

        def __init__(self, payload, status_code=200):
            self.payload = payload
            self.status_code = status_code

        def json(self):
            return self.payload

    def post(url, **kwargs):
        request_id = kwargs["json"]["request_id"]
        if url.endswith("/cancel"):
            request_ids["cancel"] = request_id
            cancel_received.set()
            return Response({"accepted": True}, 202)
        request_ids["install"] = request_id
        install_started.set()
        assert cancel_received.wait(timeout=5)
        return Response({"status": "cancelled", "installed": []})

    monkeypatch.setattr("scientific_agent.environment.httpx.post", post)
    settings = replace(
        EnvironmentSettings(), worker_url="http://packages:8091", worker_token="x" * 32
    )
    manager = EnvironmentManager(
        workspace,
        settings,
        tmp_path / "run" / "package_installations.jsonl",
        cancel_event=cancellation,
    )
    result = {}
    thread = threading.Thread(
        target=lambda: result.update(manager.install("python", ["polars"], "pypi"))
    )
    thread.start()
    assert install_started.wait(timeout=5)
    cancellation.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result["status"] == "cancelled"
    assert request_ids["install"] == request_ids["cancel"]
