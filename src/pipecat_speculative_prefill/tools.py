#
# Copyright (c) 2026, Daniel Garcia Pena (iamdgarcia)
#
# SPDX-License-Identifier: MIT
#

"""Tool-call parsing for the Hermes/Qwen format.

Open small models (Qwen 2.5, Hermes, Nous) emit tool calls wrapped in
tool_call open/close tags, with a JSON payload naming the function and its
arguments. ToolCallParser consumes the generation stream piece by piece and
separates text meant for the user (pushed to TTS) from tool calls meant for
the framework (collected and executed via pipecat machinery).

Malformed or unterminated blocks are dropped with a warning: broken JSON is a
model/sampling problem, not something to repair here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

OPEN_TAG = "<" + "tool_call" + ">"
CLOSE_TAG = "</" + "tool_call" + ">"


@dataclass
class ToolCall:
    """One parsed tool call: name + arguments dict."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class ToolCallParser:
    """Streaming splitter: feed generation pieces, get user-visible text back."""

    def __init__(self):
        self._buf = ""
        self._in_tool = False
        self._calls: list[ToolCall] = []

    def feed(self, piece: str) -> list[str]:
        """Consume one generation piece; return user-visible text pieces.

        Holds back a tail of the buffer so an open tag cannot straddle two
        pieces and leak into TTS.
        """
        self._buf += piece
        return self._drain()

    def finalize(self) -> tuple[list[str], list[ToolCall]]:
        """Flush at end of generation: leftover visible text + all tool calls.

        An unterminated tool-call block (max_tokens cut mid-call) is dropped
        with a warning; a tool call is never half-executed.
        """
        visible = self._drain(final=True)
        calls = self._calls
        self._calls = []
        return visible, calls

    def _drain(self, final: bool = False) -> list[str]:
        visible: list[str] = []
        while True:
            if not self._in_tool:
                start = self._buf.find(OPEN_TAG)
                if start < 0:
                    if final:
                        visible.append(self._buf)
                        self._buf = ""
                    else:
                        hold = len(OPEN_TAG) - 1
                        if len(self._buf) > hold:
                            visible.append(self._buf[:-hold])
                            self._buf = self._buf[-hold:]
                    return visible
                visible.append(self._buf[:start])
                self._buf = self._buf[start + len(OPEN_TAG) :]
                self._in_tool = True
            else:
                end = self._buf.find(CLOSE_TAG)
                if end < 0:
                    if final:
                        logger.warning(
                            f"ToolCallParser: unterminated tool-call block dropped: "
                            f"{self._buf[:80]!r}"
                        )
                        self._buf = ""
                        self._in_tool = False
                    return visible
                raw = self._buf[:end]
                self._buf = self._buf[end + len(CLOSE_TAG) :]
                self._in_tool = False
                self._collect(raw)

    def _collect(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"ToolCallParser: malformed tool call dropped: {raw[:80]!r}")
            return
        if not isinstance(payload, dict) or not payload.get("name"):
            logger.warning(f"ToolCallParser: tool call without name dropped: {raw[:80]!r}")
            return
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        self._calls.append(ToolCall(name=payload["name"], arguments=arguments))
