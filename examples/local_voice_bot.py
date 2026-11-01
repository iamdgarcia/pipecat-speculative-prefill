#
# Copyright (c) 2026, Daniel Garcia Peña (iamdgarcia)
#
# SPDX-License-Identifier: MIT
#

"""Local voice bot with speculative KV prefill: mic in, speech out, zero cloud.

While you talk, your partial transcriptions are prefilled into the LLM's KV
cache; the reply generation starts the moment your turn ends. Watch the log
line per turn: "spec-prefill hit NN% (x/y tokens) ... moved NNNms off the
critical path, TTFT NNms".

Install:
    pip install pipecat-speculative-prefill[transformers]
    pip install faster-whisper kokoro-onnx soundfile

Run:
    python examples/local_voice_bot.py
    # pick models with env vars: SPEC_MODEL, SPEC_WHISPER, SPEC_TTS_VOICE
"""

import asyncio
import os
import sys

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.settings import LLMSettings
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

from pipecat_speculative_prefill import (
    SpeculativeLLMService,
    TransformersPrefillBackend,
)

logger.remove(0)
logger.add(sys.stderr, level="INFO")


async def run_bot():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    stt = WhisperSTTService(
        model=os.environ.get("SPEC_WHISPER", "base.en"), device="cpu", compute_type="int8"
    )

    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(voice=os.environ.get("SPEC_TTS_VOICE", "af_heart"))
    )

    # The KV cache lives here. Partials stream in through the tap and warm it
    # while you speak; generation continues straight from the warm cache.
    backend = TransformersPrefillBackend.from_pretrained(
        os.environ.get("SPEC_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"), dtype=None
    )

    context = LLMContext()
    llm = SpeculativeLLMService(
        backend=backend,
        context=context,
        settings=LLMSettings(
            model=os.environ.get("SPEC_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
            system_instruction=(
                "You are a helpful assistant in a voice conversation. Your responses "
                "will be spoken aloud, so avoid emojis, bullet points, or other "
                "formatting that can't be spoken. Keep answers short."
            ),
            temperature=0.7,
            max_tokens=256,
        ),
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            llm.transcription_tap(),  # feeds STT partials to the LLM's cache
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=False if sys.platform == "win32" else True)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Speak when ready. Ctrl+C to stop.")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await worker.queue_frames([EndFrame()])

    try:
        await runner.run()
    except KeyboardInterrupt:
        await runner.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        pass
