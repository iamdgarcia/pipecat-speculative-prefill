"""Tool-calling through the service: FakeBackend streams a Hermes-format
tool call in pieces; the service must parse it, execute the registered
handler via pipecat machinery, and never push tool-call JSON as text."""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMContextFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager

from pipecat_speculative_prefill.backends import PrefillBackend, SyncStats
from pipecat_speculative_prefill.prompt import longest_common_prefix_len
from pipecat_speculative_prefill.service import SpeculativeLLMService
from pipecat_speculative_prefill.tools import CLOSE_TAG, OPEN_TAG

from test_service import _settings

pytestmark = pytest.mark.asyncio

CALL_PAYLOAD = json.dumps({"name": "get_weather", "arguments": {"city": "Madrid"}})
PIECES = [
    OPEN_TAG[:6],
    OPEN_TAG[6:],
    chr(10),
    CALL_PAYLOAD[:20],
    CALL_PAYLOAD[20:],
    chr(10),
    CLOSE_TAG,
]


class ToolBackend(PrefillBackend):
    """Char-per-token tokenizer; yields the canned pieces as the generation."""

    def __init__(self, pieces):
        self.pieces = pieces
        self.vocab = {}
        self.sync_calls = []
        self.tokenize_tools = "unset"

    def _tok(self, s):
        if s not in self.vocab:
            self.vocab[s] = 1000 + len(self.vocab)
        return self.vocab[s]

    async def tokenize_prompt(self, messages, tools=None):
        self.tokenize_tools = tools
        ids = [self._tok("<bos>")]
        for m in messages:
            ids.append(self._tok(f"<{m['role']}>"))
            body = m.get("tool_calls") or m.get("content") or ""
            ids.extend(self._tok(ch) for ch in str(body))
            ids.append(self._tok(f"</{m['role']}>"))
        ids.append(self._tok("<gen>"))
        return ids

    async def sync(self, tokens) -> SyncStats:
        prev = self.sync_calls[-1] if self.sync_calls else []
        lcp = longest_common_prefix_len(prev, list(tokens))
        self.sync_calls.append(list(tokens))
        return SyncStats(total=len(tokens), kept=lcp)

    def decode_stream(self, *, max_new_tokens, temperature, top_p, top_k, stop_token_ids=None, stop_flag):
        yield from self.pieces

    async def reset(self) -> None:
        pass


def _make(pieces=PIECES, with_tools=True):
    calls_made = []

    async def get_weather(params, city: str):
        calls_made.append(city)
        await params.result_callback(json.dumps({"temperature_c": 18, "sky": "clear"}))

    kwargs = {"messages": [{"role": "user", "content": "What is the weather in Madrid?"}]}
    if with_tools:
        kwargs["tools"] = [get_weather]
    context = LLMContext(**kwargs)
    service = SpeculativeLLMService(
        backend=ToolBackend(pieces),
        context=context,
        settings=_settings(),
    )
    service._task_manager = TaskManager()
    # run_function_calls reads self.pipeline_worker -> processor_setup; a stub
    # with pipeline_worker=None is enough for a service-only test.
    service._setup = SimpleNamespace(
        pipeline_worker=SimpleNamespace(app_resources=None, worker_runner=None),
        observer=None,
    )
    pushed = []
    service.push_frame = AsyncMock(
        side_effect=lambda f, d=FrameDirection.DOWNSTREAM: pushed.append(f)
    )
    return service, context, pushed, calls_made


async def _wait(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


class TestServiceToolCalling:
    async def test_tool_call_parsed_and_executed(self):
        service, context, pushed, calls_made = _make()
        await service.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
        # The runner settles the result frame asynchronously after the handler.
        await _wait(lambda: any(isinstance(f, FunctionCallResultFrame) for f in pushed))
        assert calls_made == ["Madrid"]
        kinds = [type(f).__name__ for f in pushed]
        assert "FunctionCallsStartedFrame" in kinds
        assert "FunctionCallInProgressFrame" in kinds
        assert "FunctionCallResultFrame" in kinds
        # Tool-call JSON must never reach TTS as visible text.
        text = "".join(f.text for f in pushed if isinstance(f, LLMTextFrame))
        assert "get_weather" not in text and "tool_call" not in text

    async def test_tools_passed_to_tokenizer(self):
        service, context, pushed, calls_made = _make()
        await service.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
        await _wait(lambda: len(calls_made) == 1)
        assert isinstance(service._backend.tokenize_tools, list)
        assert service._backend.tokenize_tools[0]["type"] == "function"
        assert service._backend.tokenize_tools[0]["function"]["name"] == "get_weather"

    async def test_text_around_call_is_visible(self):
        pieces = ["Let me check.", *PIECES, "Done."]
        service, context, pushed, calls_made = _make(pieces=pieces)
        await service.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
        await _wait(lambda: len(calls_made) == 1)
        text = "".join(f.text for f in pushed if isinstance(f, LLMTextFrame))
        assert text == "Let me check.Done."

    async def test_no_tools_no_function_frames(self):
        pieces = ["Just", " an", " answer."]
        service, context, pushed, calls_made = _make(pieces=pieces, with_tools=False)
        await service.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.05)
        assert calls_made == []
        kinds = [type(f).__name__ for f in pushed]
        assert "FunctionCallsStartedFrame" not in kinds
        text = "".join(f.text for f in pushed if isinstance(f, LLMTextFrame))
        assert text == "Just an answer."

    async def test_tokenizer_gets_none_tools_when_absent(self):
        service, context, pushed, calls_made = _make(pieces=["ok"], with_tools=False)
        await service.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.05)
        assert service._backend.tokenize_tools is None
