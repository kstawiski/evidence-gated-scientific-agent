from scientific_agent.workspace_tools import build_workspace_tools


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
