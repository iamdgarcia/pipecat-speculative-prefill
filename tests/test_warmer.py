#
# Copyright (c) 2026, Daniel Garcia Peña (iamdgarcia)
#
# SPDX-License-Identifier: MIT
#

"""Prefix cache warmer tests with a mocked HTTP transport."""

import asyncio
import json
import time

import httpx
from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager

from pipecat_speculative_prefill.warmer import PrefixCacheWarmer


def _make_warmer(requests: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    context = LLMContext()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    warmer = PrefixCacheWarmer(
        context=context,
        base_url="http://localhost:8000/v1",
        model="test-model",
        system_instruction="be brief",
        client=client,
    )
    warmer._task_manager = TaskManager()
    return warmer, context


async def _wait_for(warmer: PrefixCacheWarmer, requests: list[dict], count: int, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if warmer._inflight is not None and warmer._inflight.done() and len(requests) >= count:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"expected {count} warm requests")


class TestPrefixCacheWarmer:
    async def test_warm_request_shape(self):
        requests: list[dict] = []
        warmer, context = _make_warmer(requests)

        await warmer.process_frame(
            InterimTranscriptionFrame("hello", "", 0), FrameDirection.DOWNSTREAM
        )
        await _wait_for(warmer, requests, 1)

        body = requests[0]
        assert body["model"] == "test-model"
        assert body["max_tokens"] == 1
        assert body["stream"] is False
        assert body["messages"][-1] == {"role": "user", "content": "hello"}
        assert body["messages"][0] == {"role": "system", "content": "be brief"}

    async def test_final_supersedes_partial(self):
        requests: list[dict] = []
        warmer, context = _make_warmer(requests)

        await warmer.process_frame(
            InterimTranscriptionFrame("hello", "", 0), FrameDirection.DOWNSTREAM
        )
        await _wait_for(warmer, requests, 1)

        await warmer.process_frame(
            TranscriptionFrame("hello world", "", 1), FrameDirection.DOWNSTREAM
        )
        await _wait_for(warmer, requests, 2)

        assert requests[-1]["messages"][-1] == {"role": "user", "content": "hello world"}

    async def test_duplicate_content_not_rewarmed(self):
        requests: list[dict] = []
        warmer, context = _make_warmer(requests)

        await warmer.process_frame(
            InterimTranscriptionFrame("hello", "", 0), FrameDirection.DOWNSTREAM
        )
        await _wait_for(warmer, requests, 1)

        await warmer.process_frame(
            TranscriptionFrame("hello", "", 1), FrameDirection.DOWNSTREAM
        )
        await asyncio.sleep(0.05)
        assert len(requests) == 1  # identical messages already warmed: skipped
