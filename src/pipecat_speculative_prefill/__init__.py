"""pipecat-speculative-prefill: speculative KV-cache prefill for pipecat.

While the user is still talking, their partial transcriptions are already
prefilled into the LLM's KV cache. When the final transcription arrives, the
generation phase starts with almost no prefill left to do.
"""

from .backends import PrefillBackend, SyncStats, TransformersPrefillBackend
from .prompt import build_messages, longest_common_prefix_len
from .service import SpeculativeLLMService, SpeculativePrefillTap, TurnStats
from .warmer import PrefixCacheWarmer

__all__ = [
    "PrefillBackend",
    "PrefixCacheWarmer",
    "SpeculativeLLMService",
    "SpeculativePrefillTap",
    "SyncStats",
    "TransformersPrefillBackend",
    "TurnStats",
    "build_messages",
    "longest_common_prefix_len",
]
