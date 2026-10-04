"""app/services/tool_result_clip.py: long tool results are cut without breaking JSON."""

import json

from app.services.tool_result_clip import ToolResultClip


def test_a_short_result_is_left_alone() -> None:
    assert ToolResultClip(500).fit('{"a": 1}') == '{"a": 1}'


def test_a_long_json_result_stays_valid_json_within_the_limit() -> None:
    doc = {"search_id": "s1", "results": [{"url": f"https://x/{i}", "excerpts": ["word " * 400]} for i in range(8)]}

    fitted = ToolResultClip(500).fit(json.dumps(doc, indent=2))

    assert len(fitted) <= 500
    parsed = json.loads(fitted)
    assert parsed["search_id"] == "s1" and parsed["results"][0]["url"] == "https://x/0"


def test_plain_text_is_cut_at_a_word_boundary_with_a_note() -> None:
    fitted = ToolResultClip(100).fit("hello world " * 50)

    assert fitted.startswith("hello world") and "truncated" in fitted
    assert not fitted.split("\n")[0].endswith("hel")


def test_json_that_cannot_shrink_enough_falls_back_to_a_cut() -> None:
    fitted = ToolResultClip(40).fit(json.dumps({f"key{i}": i for i in range(100)}))

    assert "truncated" in fitted
