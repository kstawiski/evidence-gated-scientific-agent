from scientific_agent.workspace_tools import build_workspace_tools


def test_workspace_tools_read_and_block_escape(tmp_path):
    (tmp_path / "inside.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    tools = {tool.__name__: tool for tool in build_workspace_tools(tmp_path)}
    assert tools["read_text_file"]("inside.txt")["content"] == "alpha\nbeta\n"
    assert "error" in tools["read_text_file"]("../outside.txt")
    assert tools["search_workspace"]("beta")["matches"]
