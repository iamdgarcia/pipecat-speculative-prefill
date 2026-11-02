"""ToolCallParser unit tests, including piece-boundary straddling."""

import json

from pipecat_speculative_prefill.tools import CLOSE_TAG, OPEN_TAG, ToolCallParser

PAYLOAD = json.dumps({"name": "get_weather", "arguments": {"city": "Madrid"}})
BLOCK = OPEN_TAG + chr(10) + PAYLOAD + chr(10) + CLOSE_TAG
CALL_TEXT = "Let me check." + BLOCK + "Done."


def _feed_all(parser: ToolCallParser, text: str, n: int = 1):
    """Feed text in n pieces; return (visible, calls) after finalize."""
    out = []
    size = max(1, len(text) // n)
    for i in range(0, len(text), size):
        out.extend(parser.feed(text[i : i + size]))
    leftover, calls = parser.finalize()
    out.extend(leftover)
    return "".join(out), calls


class TestToolCallParser:
    def test_plain_text_passthrough(self):
        p = ToolCallParser()
        visible, calls = _feed_all(p, "Hello there, general Kenobi")
        assert visible == "Hello there, general Kenobi"
        assert calls == []

    def test_block_straddling_every_boundary(self):
        """No split point may leak tool-call JSON into visible text."""
        for n in range(1, len(CALL_TEXT) + 1, 3):
            p = ToolCallParser()
            visible, calls = _feed_all(p, CALL_TEXT, n)
            assert visible == "Let me check.Done.", f"failed at n={n}: {visible!r}"
            assert len(calls) == 1 and calls[0].name == "get_weather", f"n={n}"

    def test_multiple_calls(self):
        text = BLOCK + BLOCK + BLOCK.replace("Madrid", "Paris")
        p = ToolCallParser()
        _, calls = _feed_all(p, text)
        assert [c.arguments["city"] for c in calls] == ["Madrid", "Madrid", "Paris"]

    def test_malformed_json_dropped_not_crashed(self):
        p = ToolCallParser()
        bad = OPEN_TAG + chr(10) + "not-json at all" + chr(10) + CLOSE_TAG
        visible, calls = _feed_all(p, "xy" + bad)
        assert visible == "xy"
        assert calls == []

    def test_unterminated_block_dropped_at_finalize(self):
        p = ToolCallParser()
        visible, calls = _feed_all(p, "hello " + OPEN_TAG + chr(10) + '{"name": "f"}')
        assert visible == "hello "
        assert calls == []

    def test_empty_arguments_defaults(self):
        p = ToolCallParser()
        _, calls = _feed_all(p, OPEN_TAG + '{"name": "ping"}' + CLOSE_TAG)
        assert calls[0].arguments == {}

    def test_open_tag_never_leaks(self):
        """Open tag split across feeds must not leak nor false-positive."""
        p = ToolCallParser()
        out = []
        out += p.feed("text " + OPEN_TAG[:4])
        out += p.feed(OPEN_TAG[4:] + '{"name": "f", "arguments": {}}' + CLOSE_TAG)
        out += p.feed("more text after this is long enough to flush")
        leftover, calls = p.finalize()
        out += leftover
        assert "".join(out) == "text more text after this is long enough to flush"
        assert calls[0].name == "f"


def test_tags_assembled():
    assert OPEN_TAG == "<" + "tool_call" + ">"
    assert CLOSE_TAG == "</" + "tool_call" + ">"
