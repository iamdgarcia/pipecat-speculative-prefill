# Changelog

## 0.1.1

**Fixed**

- `SpeculativeLLMService.cleanup()` cancels the speculative prefill worker at
  teardown (pipecat's base cleanup does not release `create_task` tasks).
- `PrefixCacheWarmer.cleanup()` cancels an in-flight warm request.
- `TurnStats` now means the last completed generation for every field:
  `spec_syncs` and `work_offloaded_ms` were cumulative since startup.

## 0.1.0 (2026-11-06)

First release.

- `SpeculativePrefillTap`: prefills the KV cache from streaming ASR partials
  while the user is still talking.
- `SpeculativeLLMService`: syncs the cache at turn end and starts decoding from
  a warm cache; carries history across turns.
- `PrefixCacheWarmer`: same effect for OpenAI-compatible servers with prefix
  caching (vLLM, SGLang, llama.cpp server) via cheap `max_tokens=1` requests.
- Function calling in Hermes/Qwen tool-call format with streaming parse and
  holdback (tool-call tags never reach the TTS).
