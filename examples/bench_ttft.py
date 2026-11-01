#
# Copyright (c) 2026, Daniel Garcia Peña (iamdgarcia)
#
# SPDX-License-Identifier: MIT
#

"""Benchmark: speculative prefill vs cold prefill, time to first token.

Simulates a voice turn: partial transcriptions stream in while the user
talks, the final transcription arrives, generation starts. Measures TTFT with
and without speculative KV prefill.

Usage:
    python examples/bench_ttft.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import asyncio
import threading
import time

from pipecat_speculative_prefill.backends import TransformersPrefillBackend
from pipecat_speculative_prefill.prompt import build_messages

PARTIALS = [
    "Wh",
    "What is the",
    "What is the largest",
    "What is the largest ocean",
    "What is the largest ocean on",
    "What is the largest ocean on Earth",
]


async def main(model_id: str, max_new: int, filler: int):
    from pipecat_speculative_prefill.backends import TransformersPrefillBackend as B

    print(f"Loading {model_id} ...")
    backend = B.from_pretrained(model_id)

    system = "You are a concise voice assistant. Answer in one short sentence."
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! What can I do for you?"},
    ]
    # Optional filler turns to show how the win scales with context length.
    for i in range(filler):
        history += [
            {"role": "user", "content": f"Question number {i}: tell me something interesting about the sea."},
            {"role": "assistant", "content": f"Fact {i}: the ocean covers about 71 percent of Earth's surface."},
        ]

    # -- path A: cold prefill (no speculation) ---------------------------------
    full = await backend.tokenize_prompt(
        build_messages(history + [{"role": "user", "content": PARTIALS[-1]}], system)
    )
    await backend.reset()
    t0 = time.perf_counter()
    stats = await backend.sync(full)
    first = await _first_piece(backend, max_new)
    cold = (time.perf_counter() - t0) * 1000
    print(f"\ncold path : TTFT {cold:7.1f}ms  (prefill {stats.total} tokens)")
    print(f"            first piece: {first!r}")

    # -- path B: speculative prefill while the user talks ----------------------
    await backend.reset()
    offloaded = 0.0
    for partial in PARTIALS[:-1]:
        tokens = await backend.tokenize_prompt(
            build_messages(history + [{"role": "user", "content": partial}], system)
        )
        t = time.perf_counter()
        await backend.sync(tokens)
        offloaded += (time.perf_counter() - t) * 1000

    t0 = time.perf_counter()  # final transcription arrives
    final = await backend.tokenize_prompt(
        build_messages(history + [{"role": "user", "content": PARTIALS[-1]}], system)
    )
    stats = await backend.sync(final)
    first = await _first_piece(backend, max_new)
    warm = (time.perf_counter() - t0) * 1000
    print(f"\nwarm path : TTFT {warm:7.1f}ms  (cache hit {stats.hit_ratio:.0%}: "
          f"{stats.kept}/{stats.total} tokens, recomputed {stats.recomputed})")
    print(f"            speculative work moved off the critical path: {offloaded:.1f}ms")
    print(f"            first piece: {first!r}")
    print(f"\nTTFT improvement: {cold - warm:+.1f}ms ({(1 - warm / cold) * 100:.0f}% faster)")


async def _first_piece(backend: TransformersPrefillBackend, max_new: int) -> str:
    import concurrent.futures

    loop = asyncio.get_running_loop()

    def run():
        stop = threading.Event()
        gen = backend.decode_stream(
            max_new_tokens=max_new,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            stop_flag=stop,
        )
        try:
            return next(gen, "")
        finally:
            stop.set()
            # drain the generator so the thread finishes cleanly
            for _ in gen:
                pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return await loop.run_in_executor(pool, run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-new", type=int, default=8)
    parser.add_argument("--filler", type=int, default=0, help="filler dialogue turns before the measured turn")
    args = parser.parse_args()
    asyncio.run(main(args.model, args.max_new, args.filler))
