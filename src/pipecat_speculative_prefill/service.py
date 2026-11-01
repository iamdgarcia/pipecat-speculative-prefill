#
# Copyright (c) 2026, Daniel Garcia Peña (iamdgarcia)
#
# SPDX-License-Identifier: MIT
#

"""Speculative LLM service: prefill the KV cache while the user speaks.

Place a `SpeculativePrefillTap` between the STT and the user context
aggregator, and a `SpeculativeLLMService` where the LLM would go, sharing the
same `LLMContext`. Every interim transcription speculatively prefills the
prompt-so-far into the KV cache; when the turn ends and the context frame
arrives, generation starts from a cache that is already almost full.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from loguru import logger

from pipecat.frames.frames import (
    EagerTranscriptionFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, is_given
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

from .backends import PrefillBackend, SyncStats
from .prompt import build_messages


@dataclass
class TurnStats:
    """What speculative prefill did for the last completed turn.

    - spec_syncs: speculative prefills served while the user was talking.
    - work_offloaded_ms: prefill compute moved off the critical path.
    - kept/total: tokens already cached vs. prompt size at generation time
      (kept == total means the hot path: zero prefill before decoding).
    - rewound: whether the final transcript diverged from the last partial
      and the cache had to be truncated.
    - ttfb_ms: time from context frame to first generated piece.
    """

    spec_syncs: int = 0
    work_offloaded_ms: float = 0.0
    kept: int = 0
    total: int = 0
    rewound: bool = False
    ttfb_ms: float = 0.0

    @property
    def hit_ratio(self) -> float:
        return self.kept / self.total if self.total else 0.0


class SpeculativeLLMService(LLMService):
    """LLM service that speculatively prefills its KV cache from STT partials.

    Args:
        backend: A `PrefillBackend` (e.g. `TransformersPrefillBackend`) owning
            model, tokenizer and KV cache.
        context: The shared `LLMContext` — the same object you pass to
            `LLMContextAggregatorPair`. Reading it live is what lets the
            service prefill history + partial user text before the turn ends.
        settings: `LLMSettings` (system_instruction, temperature, max_tokens,
            top_p, top_k, model name for logs).

    Function calling is not supported: tools advertised in the context are
    ignored with a warning. This service is for fast plain-text voice replies.
    """

    def __init__(
        self,
        *,
        backend: PrefillBackend,
        context: LLMContext,
        settings: LLMSettings | None = None,
        **kwargs,
    ):
        super().__init__(settings=settings, **kwargs)
        self._backend = backend
        self._context = context
        self._prefill_lock = asyncio.Lock()
        self._prefill_text: str | None = None
        self._prefill_wake = asyncio.Event()
        self._prefill_busy = False  # worker is inside a processing iteration
        self._prefill_done = 0  # completed prefill iterations (observability)
        self._prefill_task: asyncio.Task | None = None
        self._generating = False
        self._turn = TurnStats()
        self._warned_tools = False

    # -- wiring ------------------------------------------------------------

    def transcription_tap(self) -> "SpeculativePrefillTap":
        """Build the pipeline tap that feeds this service STT partials.

        Place it between the STT service and the user context aggregator.
        """
        return SpeculativePrefillTap(self)

    @property
    def last_turn_stats(self) -> TurnStats:
        """Stats of the last completed generation (see `TurnStats`)."""
        return self._turn

    # -- speculative prefill -----------------------------------------------

    def note_partial(self, text: str) -> None:
        """Feed a partial transcript (called by the tap; fire-and-forget)."""
        self._request_prefill(text)

    def note_final(self, text: str) -> None:
        """Feed the final transcript of the turn (called by the tap)."""
        self._request_prefill(text)

    def _request_prefill(self, text: str) -> None:
        if self._generating or not text or not text.strip():
            return
        self._prefill_text = text
        self._prefill_wake.set()
        if self._prefill_task is None or self._prefill_task.done():
            self._prefill_task = self.create_task(self._prefill_worker())

    async def _prefill_worker(self) -> None:
        """Apply the latest requested prefill, coalescing stale ones.

        Always syncs to the newest text: a queued sync for a stale partial
        becomes a no-op against the cache a newer sync already wrote.
        """
        while True:
            await self._prefill_wake.wait()
            # Mark busy before taking the request so an observer that sees the
            # wake cleared and the slot empty knows whether work is in flight.
            self._prefill_busy = True
            self._prefill_wake.clear()
            text = self._prefill_text
            self._prefill_text = None
            if text is None:
                self._prefill_busy = False
                continue
            try:
                async with self._prefill_lock:
                    if self._generating:
                        continue
                    messages = build_messages(
                        self._context.messages, self._system_instruction(), text
                    )
                    tokens = await self._backend.tokenize_prompt(messages)
                    start = time.perf_counter()
                    stats = await self._backend.sync(tokens)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    self._turn.spec_syncs += 1
                    self._turn.work_offloaded_ms += elapsed_ms
                    self._turn.kept, self._turn.total = stats.kept, stats.total
                    logger.debug(
                        f"{self}: speculative prefill +{stats.recomputed} tokens "
                        f"({elapsed_ms:.0f}ms off the critical path)"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"{self}: speculative prefill failed: {e}")
            finally:
                self._prefill_done += 1
                self._prefill_busy = False

    # -- generation ---------------------------------------------------------

    def _system_instruction(self) -> str | None:
        si = self._settings.system_instruction
        return si if isinstance(si, str) else None

    def _opt(self, value: Any, default: Any) -> Any:
        return default if value is None or not is_given(value) else value

    async def _process_context(self, context: LLMContext) -> None:
        if is_given(context.tools) and context.tools and not self._warned_tools:
            self._warned_tools = True
            logger.warning(
                f"{self}: tools advertised but {type(self).__name__} does not support "
                "function calling; running without tools."
            )

        await self.start_ttfb_metrics()
        messages = build_messages(context.messages, self._system_instruction())
        tokens = await self._backend.tokenize_prompt(messages)

        async with self._prefill_lock:
            self._generating = True
            stop_flag = threading.Event()
            start = time.perf_counter()
            stats = SyncStats(total=len(tokens), kept=0)
            first_piece = True
            try:
                stats = await self._backend.sync(tokens)
                self._turn.kept, self._turn.total = stats.kept, stats.total
                self._turn.rewound = stats.recomputed > 0
                async for piece in self._decode(
                    max_new_tokens=int(self._opt(self._settings.max_tokens, 512)),
                    temperature=float(self._opt(self._settings.temperature, 0.7)),
                    top_p=float(self._opt(self._settings.top_p, 1.0)),
                    top_k=int(self._opt(self._settings.top_k, 0)),
                    stop_flag=stop_flag,
                ):
                    if first_piece:
                        self._turn.ttfb_ms = (time.perf_counter() - start) * 1000
                        await self.stop_ttfb_metrics()
                        first_piece = False
                    await self._push_llm_text(piece)
                logger.info(
                    f"{self}: spec-prefill hit {self._turn.hit_ratio:.0%} "
                    f"({self._turn.kept}/{self._turn.total} tokens), "
                    f"{self._turn.spec_syncs} speculative sync(s) moved "
                    f"{self._turn.work_offloaded_ms:.0f}ms off the critical path, "
                    f"TTFT {self._turn.ttfb_ms:.0f}ms"
                )
            finally:
                stop_flag.set()
                self._generating = False

    async def _decode(self, **kwargs) -> AsyncIterator[str]:
        """Run the backend's sync decode generator in a thread, bridging pieces."""
        queue: asyncio.Queue = asyncio.Queue()

        def work() -> None:
            try:
                for piece in self._backend.decode_stream(**kwargs):
                    queue.put_nowait(piece)
            except BaseException as e:  # noqa: BLE001 - forwarded to the loop
                queue.put_nowait(e)
            queue.put_nowait(None)

        thread_task = asyncio.create_task(asyncio.to_thread(work))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # On cancellation, tell the generation thread to stop at the next
            # token; the cache stays consistent and the next sync reconciles.
            kwargs["stop_flag"].set()
            await asyncio.gather(thread_task, return_exceptions=True)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(frame.context)
            except Exception as e:
                await self.push_error(error_msg=f"Error during completion: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


class SpeculativePrefillTap(FrameProcessor):
    """Pipeline tap between STT and the user context aggregator.

    Passes every frame through untouched; feeds interim/final transcriptions
    to the linked `SpeculativeLLMService` so it can prefill speculatively.
    """

    def __init__(self, service: SpeculativeLLMService, **kwargs):
        super().__init__(**kwargs)
        self._service = service

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (InterimTranscriptionFrame, EagerTranscriptionFrame)):
            self._service.note_partial(frame.text)
        elif isinstance(frame, TranscriptionFrame):
            self._service.note_final(frame.text)

        await self.push_frame(frame, direction)
