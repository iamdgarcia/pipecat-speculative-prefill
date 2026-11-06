# Changelog

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
