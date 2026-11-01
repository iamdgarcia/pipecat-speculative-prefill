#
# Copyright (c) 2026, Daniel Garcia Peña (iamdgarcia)
#
# SPDX-License-Identifier: MIT
#

"""Prefix-cache warming for OpenAI-compatible servers (vLLM, SGLang, TGI...).

In-process KV handoff needs a local model. But a server with automatic prefix
caching gives you the same win from the outside: fire a cheap `max_tokens=1`
request with the prompt-so-far on every partial transcription, and by the time
the turn ends the server has the whole prefix cached, so the real request
skips most of its prefill.

This is best-effort by design: it never blocks the pipeline and failures are
only logged.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from loguru import logger

from pipecat.frames.frames import (
    EagerTranscriptionFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, is_given
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .prompt import build_messages


class PrefixCacheWarmer(FrameProcessor):
    """Pipeline tap that warms a server-side prefix cache on STT partials.

    Place it between the STT service and the user context aggregator, sharing
    the same `LLMContext` you give to the aggregators and pointing it at the
    same OpenAI-compatible endpoint (and model) your LLM service uses. Tools
    are not included in the warming request, so with tool-using bots the cache
    hit is weaker (note logged once).

    Args:
        context: The shared `LLMContext`.
        base_url: Base URL of the server, e.g. `http://localhost:8000/v1`.
        model: The model name to send, matching your LLM service's model.
        api_key: Optional bearer token.
        system_instruction: System instruction, if your LLM service uses one
            (it must match, or the prefixes won't align).
        timeout_secs: Per-request timeout; warming is dropped on timeout.
    """

    def __init__(
        self,
        *,
        context: LLMContext,
        base_url: str,
        model: str,
        api_key: str | None = None,
        system_instruction: str | None = None,
        timeout_secs: float = 2.0,
        client: httpx.AsyncClient | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._context = context
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._system_instruction = system_instruction
        self._timeout_secs = timeout_secs
        self._client = client
        self._owns_client = client is None
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._inflight: asyncio.Task | None = None
        self._last_messages: Any = None
        self._last_warm_ok = False
        self._warned_tools = False

    async def cleanup(self):
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
        await super().cleanup()

    def _warm(self, partial: str | None) -> None:
        if self._inflight is not None and not self._inflight.done():
            self._inflight.cancel()
            self._last_warm_ok = False  # cancelled warm is not a warm
        if is_tooly(self._context) and not self._warned_tools:
            self._warned_tools = True
            logger.warning(
                f"{self}: context has tools; warming request omits them, so the "
                "server prefix hit will be partial for tool-using turns."
            )
        messages = build_messages(self._context.messages, self._system_instruction, partial)
        if messages == self._last_messages and self._last_warm_ok:
            return  # same prompt already warmed successfully
        self._last_messages = messages
        self._last_warm_ok = False
        self._inflight = self.create_task(self._warm_request(messages))

    async def _warm_request(self, messages: list[dict[str, Any]]) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_secs)
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 1,
            "stream": False,
        }
        start = time.perf_counter()
        try:
            response = await self._client.post(self._url, json=payload, headers=self._headers)
            response.raise_for_status()
            self._last_warm_ok = True
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"{self}: warmed prefix ({len(messages)} messages) in {elapsed_ms:.0f}ms")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"{self}: prefix warming skipped: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (InterimTranscriptionFrame, EagerTranscriptionFrame)):
            self._warm(frame.text)
        elif isinstance(frame, TranscriptionFrame):
            self._warm(frame.text)

        await self.push_frame(frame, direction)


def is_tooly(context: LLMContext) -> bool:
    """Whether the context advertises tools (for the one-time warning)."""
    tools = context.tools
    if not is_given(tools) or tools is None:
        return False
    return len(tools.standard_tools) > 0
