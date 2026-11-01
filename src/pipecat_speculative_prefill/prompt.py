#
# Copyright (c) 2026, Daniel Garcia Peña (iamdgarcia)
#
# SPDX-License-Identifier: MIT
#

"""Prompt building: pipecat context messages -> chat-template-ready messages."""

from __future__ import annotations

from typing import Any


def longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the longest shared prefix of two token id lists."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _flatten_content(content: Any) -> Any:
    """Reduce a message content to text the local model can consume.

    String content passes through. List content (multimodal messages) keeps its
    text parts and drops the rest; images/audio can't be prefilled into a text
    model, so that's the honest best effort.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return content


def build_messages(
    context_messages: list[dict[str, Any]],
    system_instruction: str | None = None,
    partial_user_text: str | None = None,
) -> list[dict[str, Any]]:
    """Build the chat-template message list for the model.

    Args:
        context_messages: Messages from the shared ``LLMContext`` (live view of
            what the aggregators hold).
        system_instruction: The service's system instruction, prepended as a
            system message like the OpenAI adapter does.
        partial_user_text: The user's transcript so far (interim or final).
            When given, it becomes the last message: a user turn in progress.

    Returns:
        Message dicts ready for the tokenizer's chat template. ``developer``
        roles are mapped to ``system`` (local chat templates don't know the
        OpenAI developer role).
    """
    messages: list[dict[str, Any]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    for m in context_messages:
        role = m.get("role")
        if role == "developer":
            messages.append({"role": "system", "content": _flatten_content(m.get("content"))})
        else:
            messages.append({"role": role, "content": _flatten_content(m.get("content"))})
    if partial_user_text is not None:
        messages.append({"role": "user", "content": partial_user_text})
    return messages
