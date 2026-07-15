from scientific_agent.visibility import VisibleTextFilter, strip_reasoning_envelope


def test_qwen_prefix_reasoning_is_not_released_before_closing_marker():
    boundary = VisibleTextFilter()

    assert boundary.feed("private planning text") == ""
    assert boundary.feed("</thi") == ""
    assert boundary.feed('nk>\n{"') == '\n{"'
    assert boundary.feed('value":"safe"}') == 'value":"safe"}'
    assert boundary.finish() == ""


def test_tagged_reasoning_is_removed_across_chunk_boundaries():
    boundary = VisibleTextFilter()

    output = (
        "".join(
            boundary.feed(chunk)
            for chunk in (
                '{"value":"safe",',
                '"note":"shown"}<thi',
                "nk>private",
                " reasoning</think>",
            )
        )
        + boundary.finish()
    )

    assert output == '{"value":"safe","note":"shown"}'
    assert "private" not in output


def test_untagged_natural_language_is_released_only_at_completion():
    boundary = VisibleTextFilter()
    assert boundary.feed("A user-visible answer.") == ""
    assert boundary.finish() == "A user-visible answer."


def test_complete_reasoning_envelope_is_removed_for_schema_parsing():
    assert (
        strip_reasoning_envelope('<think>private</think>{"value":"safe"}')
        == '{"value":"safe"}'
    )


def test_unicode_case_expansion_before_close_marker_preserves_json_boundary():
    value = 'reasoning about İpek</THINK>{"value":"safe"}'
    assert strip_reasoning_envelope(value) == '{"value":"safe"}'


def test_unicode_case_expansion_does_not_shift_streamed_marker_indices():
    boundary = VisibleTextFilter()
    assert boundary.feed("private İ reasoning</TH") == ""
    assert boundary.feed('INK>{"value":"safe"}') == '{"value":"safe"}'
    assert boundary.finish() == ""
