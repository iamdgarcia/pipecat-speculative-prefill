"""Real-model E2E: Qwen2.5-0.5B-Instruct, function calling through the full
speculative-prefill path.

Turn 1: the user asks for the weather; the model emits a Hermes-format tool
call; the service parses it, executes the registered handler, and pushes no
tool-call JSON as text.
Turn 2 (after the tool result is added to the context, mimicking the
assistant aggregator): the model answers from the tool result, and the
speculative cache is reused (hit ratio asserted).
"""

import json
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
from pipecat.services.settings import LLMSettings
from pipecat.utils.asyncio.task_manager import TaskManager

from pipecat_speculative_prefill.backends import TransformersPrefillBackend
from pipecat_speculative_prefill.service import SpeculativeLLMService

pytestmark = pytest.mark.asyncio

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _settings() -> LLMSettings:
    return LLMSettings(
        model=MODEL,
        system_instruction=None,
        temperature=0.0,
        max_tokens=64,
        top_p=None,
        top_k=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        filter_incomplete_user_turns=None,
        user_turn_completion_config=None,
    )


@pytest.fixture(scope="module")
def backend():
    return TransformersPrefillBackend.from_pretrained(MODEL)


def _make(backend):
    calls_made = []

    async def get_weather(params, city: str):
        calls_made.append(city)
        await params.result_callback(json.dumps({"temperature_c": 18, "sky": "clear"}))

    context = LLMContext(
        messages=[{"role": "user", "content": "What's the weather in Madrid right now?"}],
        tools=[get_weather],
    )
    service = SpeculativeLLMService(backend=backend, context=context, settings=_settings())
    service._task_manager = TaskManager()
    service._setup = SimpleNamespace(
        pipeline_worker=SimpleNamespace(app_resources=None, worker_runner=None),
        observer=None,
    )
    pushed = []
    service.push_frame = AsyncMock(
        side_effect=lambda f, d=FrameDirection.DOWNSTREAM: pushed.append(f)
    )
    return service, context, pushed, calls_made


def _pushed_text(pushed):
    return "".join(f.text for f in pushed if isinstance(f, LLMTextFrame))


async def test_tool_call_then_answer_with_cache_hit(backend):
    service, context, pushed, calls_made = _make(backend)

    # Turn 1: question -> tool call -> handler executed. The runner settles
    # asynchronously after process_frame returns, so wait for the result.
    import asyncio

    async def _wait(predicate, timeout=10.0):
        import time as _t
        deadline = _t.monotonic() + timeout
        while _t.monotonic() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.02)
        raise AssertionError("condition never became true")

    await service.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
    await _wait(lambda: any(isinstance(f, FunctionCallResultFrame) for f in pushed))
    assert calls_made == ["Madrid"]
    kinds = [type(f).__name__ for f in pushed]
    assert "FunctionCallsStartedFrame" in kinds
    assert "FunctionCallInProgressFrame" in kinds
    assert "FunctionCallResultFrame" in kinds
    text1 = _pushed_text(pushed)
    assert "get_weather" not in text1 and "tool_call" not in text1
    call_id = next(
        f.tool_call_id for f in pushed if isinstance(f, FunctionCallInProgressFrame)
    )

    # Mimic the assistant aggregator's context bookkeeping.
    context.add_message(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": call_id,
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Madrid"}),
                    },
                }
            ],
        }
    )
    context.add_message(
        {
            "role": "tool",
            "content": json.dumps({"temperature_c": 18, "sky": "clear"}),
            "tool_call_id": call_id,
        }
    )

    # Turn 2: the model answers from the tool result.
    pushed.clear()
    await service.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
    kinds2 = [type(f).__name__ for f in pushed]
    assert "FunctionCallsStartedFrame" not in kinds2
    answer = _pushed_text(pushed)
    assert "18" in answer, f"answer missing temperature: {answer!r}"

    stats2 = service.last_turn_stats
    print(f"[e2e] turn-2 cache hit ratio: {stats2.hit_ratio:.0%} "
          f"({stats2.kept}/{stats2.total}), TTFT {stats2.ttfb_ms:.0f}ms")
    assert stats2.hit_ratio >= 0.5, (
        f"speculative cache not reused after tool turn: {stats2.hit_ratio:.0%}"
    )
