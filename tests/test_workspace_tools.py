from scientific_agent.provenance import sha256_file
from scientific_agent.schemas import ArtifactRef
from scientific_agent.workspace_tools import MAX_READ_BYTES, build_workspace_tools


def test_workspace_tools_read_and_block_escape(tmp_path):
    (tmp_path / "inside.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    tools = {tool.__name__: tool for tool in build_workspace_tools(tmp_path)}
    assert tools["read_text_file"]("inside.txt")["content"] == "alpha\nbeta\n"
    assert tools["read_text_file"]("/workspace/inside.txt")["content"] == (
        "alpha\nbeta\n"
    )
    assert "error" in tools["read_text_file"]("../outside.txt")
    assert "error" in tools["read_text_file"]("/etc/passwd")
    assert tools["search_workspace"]("beta")["matches"]


def test_workspace_search_does_not_follow_external_symlink(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("TOPSECRET marker\n", encoding="utf-8")
    (tmp_path / "leak.txt").symlink_to(outside)
    tools = {tool.__name__: tool for tool in build_workspace_tools(tmp_path)}

    assert "error" in tools["read_text_file"]("/workspace/leak.txt")
    result = tools["search_workspace"]("TOPSECRET", "/workspace")
    assert result == {"matches": [], "truncated": False}


def test_read_text_file_accepts_only_exact_registered_generated_artifacts(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    managed = tmp_path / "computations" / "attempt-0" / "exec-001" / "output"
    managed.mkdir(parents=True)
    result_path = managed / "result.json"
    result_path.write_text('{"estimate": 5}\n', encoding="utf-8")
    neighbor = managed / "neighbor.txt"
    neighbor.write_text("not registered", encoding="utf-8")
    registered = [
        ArtifactRef(
            path=str(result_path.resolve()),
            sha256=sha256_file(result_path),
            description="sandbox-generated analysis artifact",
        )
    ]
    tools = {
        tool.__name__: tool
        for tool in build_workspace_tools(
            workspace, registered_artifacts=lambda: tuple(registered)
        )
    }

    observed = tools["read_text_file"](str(result_path.resolve()))
    assert observed["content"] == '{"estimate": 5}\n'
    assert observed["source"] == "registered_computation_artifact"
    assert "error" in tools["read_text_file"](str(neighbor.resolve()))
    assert "error" in tools["read_text_file"]("/etc/passwd")
    assert "error" in tools["list_workspace"](str(managed.resolve()))
    assert "error" in tools["search_workspace"]("estimate", str(managed.resolve()))

    result_path.write_text('{"estimate": 500}\n', encoding="utf-8")
    assert (
        "hash mismatch" in tools["read_text_file"](str(result_path.resolve()))["error"]
    )


def test_registered_artifact_reader_is_dynamic_and_rejects_unsafe_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    text = managed / "table.csv"
    text.write_text("group,value\nA,1\n", encoding="utf-8")
    image = managed / "figure.png"
    image.write_bytes(b"not really an image")
    rejected = managed / "rejected.txt"
    rejected.write_text("failed partial output", encoding="utf-8")
    symlink = managed / "linked.txt"
    symlink.symlink_to(text)
    oversized = managed / "large.txt"
    oversized.write_bytes(b"x" * (MAX_READ_BYTES + 1))
    registered = []
    tools = {
        tool.__name__: tool
        for tool in build_workspace_tools(
            workspace, registered_artifacts=lambda: tuple(registered)
        )
    }

    assert "error" in tools["read_text_file"](str(text.resolve()))
    registered.append(
        ArtifactRef(
            path=str(text.resolve()),
            sha256=sha256_file(text),
            description="sandbox-generated analysis artifact",
        )
    )
    assert tools["read_text_file"](str(text.resolve()))["content"].startswith(
        "group,value"
    )
    registered.append(
        ArtifactRef(
            path=str(rejected.resolve()),
            sha256=sha256_file(rejected),
            description="rejected sandbox output (not evidence)",
        )
    )
    assert "error" in tools["read_text_file"](str(rejected.resolve()))

    for unsafe in (image, symlink, oversized):
        registered.append(
            ArtifactRef(
                path=str(unsafe),
                sha256=sha256_file(unsafe.resolve()),
                description="sandbox-generated analysis artifact",
            )
        )
        assert "error" in tools["read_text_file"](str(unsafe))
