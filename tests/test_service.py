#
# Copyright (c) 2026, Daniel Garcia Peña (iamdgarcia)
#
# SPDX-License-Identifier: MIT
#

"""End-to-end service tests with an in-memory backend (no model downloads).

Tokenization is char-per-token over a fixed template, so partial transcripts
grow append-only exactly like real ASR partials do at token level.
"""

import asyncio
import time
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import LLMSettings
from pipecat.utils.asyncio.task_manager import TaskManager

from pipecat_speculative_prefill.backends import PrefillBackend, SyncStats
from pipecat_speculative_prefill.prompt import longest_common_prefix_len
from pipecat_speculative_prefill.service import SpeculativeLLMService


class FakeBackend(PrefillBackend):
    """Records syncs; tokenizes append-only; yields canned pieces."""

    def __init__(self, pieces: list[str], block_after_first: bool = False):
        self.vocab: dict[str, int] = {}
        self.sync_calls: list[list[int]] = []
        self.stats: list[SyncStats] = []
        self.pieces = pieces
        self.block_after_first = block_after_first
        # threading.Events: decode runs in a worker thread; is_set()/set() must
        # be safe from both the event loop and the thread.
        self.decode_started = asyncio.Event()
        self.release_decode = asyncio.Event()

    def _tok(self, s: str) -> int:
        if s not in self.vocab:
            self.vocab[s] = 1000 + len(self.vocab)
        return self.vocab[s]

    async def tokenize_prompt(self, messages: list[dict]) -> list[int]:
        ids = [self._tok("<bos>")]
        for m in messages:
            ids.append(self._tok(f"<{m['role']}>"))
            ids.extend(self._tok(ch) for ch in str(m["content"]))
            ids.append(self._tok(f"</{m['role']}>"))
        ids.append(self._tok("<gen>"))
        return ids

    async def sync(self, tokens) -> SyncStats:
        prev = self.sync_calls[-1] if self.sync_calls else []
        lcp = longest_common_prefix_len(prev, list(tokens))
        self.sync_calls.append(list(tokens))
        stats = SyncStats(total=len(tokens), kept=lcp)
        self.stats.append(stats)
        return stats

    def decode_stream(self, *, max_new_tokens, temperature, top_p, top_k, stop_token_ids=None, stop_flag):
        self.decode_started.set()
        first = True
        for piece in self.pieces:
            if first and self.block_after_first:
                first = False
                while not self.release_decode.is_set():
                    if stop_flag.is_set():
                        return
                    time.sleep(0.001)
            if stop_flag.is_set():
                return
            yield piece

    async def reset(self) -> None:
        self.sync_calls.append([])


def _settings(**kw) -> LLMSettings:
    return LLMSettings(
        model="fake-model",
        system_instruction=kw.get("system_instruction", "be brief"),
        temperature=0.0,
        max_tokens=16,
        top_p=None,
        top_k=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        filter_incomplete_user_turns=None,
        user_turn_completion_config=None,
    )


def _make_service(pieces=None, block_after_first=False):
    backend = FakeBackend(pieces or ["Hi", " there"], block_after_first)
    context = LLMContext()
    service = SpeculativeLLMService(backend=backend, context=context, settings=_settings())
    service._task_manager = TaskManager()
    # Capture everything the service pushes.
    pushed: list[Frame] = []
    service.push_frame = AsyncMock(
        side_effect=lambda f, d=FrameDirection.DOWNSTREAM: pushed.append(f)
    )
    return service, backend, context, pushed


async def _drain_prefill(service: SpeculativeLLMService, timeout: float = 2.0):
    """Wait until every requested speculative prefill has completed."""
    deadline = time.monotonic() + timeout
    last = service._prefill_done
    while time.monotonic() < deadline:
        quiet = (
            service._prefill_text is None
            and not service._prefill_wake.is_set()
            and not service._prefill_busy
        )
        if quiet and service._prefill_done == last:
            return
        last = service._prefill_done
        await asyncio.sleep(0.001)
    raise AssertionError("prefill worker never went idle")


class TestSpeculativePrefillFlow:
    async def test_partials_prefill_and_final_generates_from_cache(self):
        service, backend, context, pushed = _make_service()

        tap = service.transcription_tap()
        await tap.process_frame(
            InterimTranscriptionFrame("hello", "", 0), FrameDirection.DOWNSTREAM
        )
        await _drain_prefill(service)
        assert len(backend.sync_calls) == 1
        assert service.last_turn_stats.spec_syncs == 1

        await tap.process_frame(
            InterimTranscriptionFrame("hello wor", "", 1), FrameDirection.DOWNSTREAM
        )
        await _drain_prefill(service)
        # The second sync kept the longest common prefix with the first (the
        # template boundary shifts, like real subword tokenization) and only
        # recomputed the tail.
        lcp = longest_common_prefix_len(backend.sync_calls[0], backend.sync_calls[1])
        assert backend.stats[1].kept == lcp
        assert lcp > 0
        assert backend.stats[1].total > backend.stats[0].total

        await tap.process_frame(TranscriptionFrame("hello wor", "", 2), FrameDirection.DOWNSTREAM)
        await _drain_prefill(service)
        assert len(backend.sync_calls) == 3

        # Aggregator adds the final user turn, then the context frame arrives.
        context.add_message({"role": "user", "content": "hello wor"})
        await service.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)

        texts = [f.text for f in pushed if isinstance(f, LLMTextFrame)]
        assert texts == ["Hi", " there"]
        kinds = [type(f) for f in pushed]
        assert kinds[0] is LLMFullResponseStartFrame
        assert kinds[-1] is LLMFullResponseEndFrame
        # The generation-time sync found the whole prompt already cached.
        assert service.last_turn_stats.kept == service.last_turn_stats.total
        assert service.last_turn_stats.hit_ratio == 1.0
        assert service.last_turn_stats.rewound is False

    async def test_diverged_final_rewinds_and_recomputes(self):
        service, backend, context, pushed = _make_service()

        tap = service.transcription_tap()
        await tap.process_frame(
            InterimTranscriptionFrame("hello wor", "", 0), FrameDirection.DOWNSTREAM
        )
        await _drain_prefill(service)

        # Final transcript diverges from the partial (ASR revised itself).
        context.add_message({"role": "user", "content": "hello moon"})
        await service.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)

        assert service.last_turn_stats.rewound is True
        assert service.last_turn_stats.kept < service.last_turn_stats.total

    async def test_no_prefill_while_generating(self):
        service, backend, context, pushed = _make_service(
            pieces=["a", "b"], block_after_first=True
        )
        context.add_message({"role": "user", "content": "hi"})

        gen_task = asyncio.create_task(
            service.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
        )
        deadline = time.monotonic() + 2
        while not backend.decode_started.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.001)
        assert backend.decode_started.is_set(), "decode never started"
        assert service._generating is True

        service.note_partial("late partial")
        await asyncio.sleep(0.01)
        # no speculative sync while generating (only the generation-time one)
        assert len(backend.sync_calls) == 1

        backend.release_decode.set()
        await asyncio.wait_for(gen_task, 2)

    async def test_tap_passes_frames_through(self):
        service, backend, context, pushed = _make_service()
        tap = service.transcription_tap()
        tap.push_frame = AsyncMock()

        frame = TranscriptionFrame("final words", "", 0)
        await tap.process_frame(frame, FrameDirection.DOWNSTREAM)
        tap.push_frame.assert_awaited_once_with(frame, FrameDirection.DOWNSTREAM)

    async def test_empty_partial_ignored(self):
        service, backend, context, pushed = _make_service()
        service.note_partial("   ")
        await asyncio.sleep(0.005)
        assert backend.sync_calls == []
