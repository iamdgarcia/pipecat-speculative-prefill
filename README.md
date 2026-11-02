# pipecat-speculative-prefill

Speculative KV-cache prefill for [pipecat](https://github.com/pipecat-ai/pipecat) voice agents.

**What you'll learn here (measured):** with a local model, the LLM's prefill can run *while the user is still talking*. On this dev machine (Intel corporate laptop, CPU-only, fp32, Qwen2.5-0.5B-Instruct):

| Prompt | Cold TTFT | With speculative prefill | Faster |
|---|---|---|---|
| 53 tokens | 1155.8 ms | 317.0 ms | **73%** |
| 873 tokens | 10868.3 ms | 367.8 ms | **97%** |

TTFT = time from final transcription to first generated token. Reproduce on your machine with `python examples/bench_ttft.py --model Qwen/Qwen2.5-0.5B-Instruct`.

**What to do:** add a `SpeculativePrefillTap` between your STT and its context aggregator, and swap your `LLMService` for a `SpeculativeLLMService`. Every interim transcription prefills the prompt-so-far into the KV cache; when the turn ends, generation starts from a cache that is already warm. The service logs one line per turn telling you exactly what the trick bought you:

```
SpeculativeLLMService#0: spec-prefill hit 99% (867/873 tokens), 5 speculative sync(s)
moved 14881.6ms off the critical path, TTFT 368ms
```

## How it works

```text
 mic → STT ──► [SpeculativePrefillTap] ──► user aggregator ──► [SpeculativeLLMService] → TTS
              interim: "what is the"           (turn ends)          │
                        │  speculative sync                        │ generation-time sync
                        ▼                                          ▼
                 KV cache: prompt-so-far ────────────────► cache hit 99% → decode now
```

1. **Speculative prefill (while the user talks).** The tap feeds every `InterimTranscriptionFrame` / `EagerTranscriptionFrame` / final `TranscriptionFrame` to the service. The service applies the chat template to the live `LLMContext` plus the partial text, tokenizes, and syncs the KV cache to it: keep the longest common prefix with what's already cached, evaluate only the delta.
2. **Generation (final transcription arrives).** The context frame triggers the final tokenization and sync. In the common case ASR partials only extend, so the final prompt is already ~100% cached and decoding starts immediately. If the final transcript diverged from the last partial (ASR revision), the cache is rewound to the common prefix and only the divergent tail is recomputed.
3. **KV carry-over across turns.** After a response, the cache holds history + assistant reply. The next turn's partials align against it, so even multi-turn conversations keep paying near-zero prefill.

True KV handoff requires a local model (the cache is an in-process object). For OpenAI-compatible servers with prefix caching (vLLM, SGLang, llama.cpp server), the `PrefixCacheWarmer` tap gets you the same effect from the outside: cheap `max_tokens=1` requests warm the server's prefix cache on every partial, so the real request skips most of its prefill.

## Install

```bash
pip install pipecat-speculative-prefill[transformers]
```

Requires `pipecat-ai>=1.0.0`.

## Quickstart

```python
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair, LLMUserAggregatorParams,
)
from pipecat.services.settings import LLMSettings

from pipecat_speculative_prefill import SpeculativeLLMService, TransformersPrefillBackend

backend = TransformersPrefillBackend.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

context = LLMContext()          # shared: aggregator pair AND service
llm = SpeculativeLLMService(
    backend=backend,
    context=context,
    settings=LLMSettings(system_instruction="You are a concise voice assistant."),
)
user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
)

pipeline = Pipeline([
    transport.input(),
    stt,
    llm.transcription_tap(),    # ← the one new processor: place between STT and the aggregator
    user_aggregator,
    llm,
    tts,
    transport.output(),
    assistant_aggregator,
])
```

With an STT that emits interims (Deepgram, AssemblyAI, Azure, Gladia...) the cache warms while the user speaks. With final-only local STT (Whisper), the prefill still happens in the gap between the final transcription and the end of the turn — less offloaded work, same correctness.

For a vLLM-style server instead of an in-process model:

```python
from pipecat_speculative_prefill import PrefixCacheWarmer

warmer = PrefixCacheWarmer(context=context, base_url="http://localhost:8000/v1",
                           model="Qwen/Qwen2.5-0.5B-Instruct", api_key="dummy")
# pipeline: [stt, warmer, user_aggregator, your_openai_service, ...]
```

## Backends

| Backend | Where the cache lives | KV handoff | Notes |
|---|---|---|---|
| `TransformersPrefillBackend` | in-process (HF `DynamicCache`) | yes | the tested path; any causal-LM checkpoint with a chat template |
| `PrefixCacheWarmer` | server-side prefix cache | no (cache hit instead) | any OpenAI-compatible server with prefix caching; your normal LLM service keeps tools/streaming |

## Function calling

`SpeculativeLLMService` supports tools in the Hermes/Qwen format (the ```<tool_call>```
block most open small models emit: Qwen 2.5, Hermes, Nous). Tools come from
the shared context like any pipecat LLM service (direct functions or
`FunctionSchema`s), calls are parsed out of generation, never shown to TTS,
and executed through the standard pipecat machinery: registered handlers,
`FunctionCallsStartedFrame`, parallel-call grouping, and the speculation
guard (a tool call attempted during speculative inference is withdrawn and
re-run after the committed transcript).

Tool results flow back through the context (role `tool`), and the next turn's
prefill reuses the cache: on our E2E test (Qwen2.5-0.5B-Instruct, one tool),
the turn after the tool call hit 81% cache reuse. Other tool-call formats
(e.g. Llama-3.1, Mistral `[TOOL_CALLS]`) are out of scope for now.

## Honest limits

- Tool-call parsing covers the Hermes/Qwen format only (see above).
- Sampling is `temperature` / `top_p` / `top_k` (settings-driven); no stop strings, no repetition penalty. EOS + `max_tokens` end generation.
- Multimodal messages are flattened to their text parts (a text model can't prefill images).
- A cancelled generation stops at the next token boundary; the cache stays consistent and the next sync reconciles.
- Run one speculative service per model: the backend serializes cache access with a lock, so concurrent generation and prefill take turns.

## Tests

```bash
pip install -e ".[dev,transformers]"
pytest tests/
```

The load-bearing test: incremental prefill (prefix cached → crop → delta eval) must produce next-token logits identical to a direct full prefill. It does, to the last bit (`max |logit diff| = 0.0`).

## Attribution

Built by iamdgarcia (Daniel García Peña) on top of [pipecat](https://github.com/pipecat-ai/pipecat) (BSD-2, Daily). The idea rhymes with what pipecat's own eager-end-of-turn speculation does for turn timing; this plugin does it for the prompt side: don't wait for the user to finish talking to start reading their sentence.
