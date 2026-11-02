#
# Copyright (c) 2026, Daniel Garcia Peña (iamdgarcia)
#
# SPDX-License-Identifier: MIT
#

"""KV-cache backends for speculative prefill.

A backend owns the model, its KV cache, and the token list the cache holds.
The service asks it to `sync` to a target token list; the backend keeps as
much of the cache as the longest common prefix allows and evaluates only the
delta. Generation then continues straight from the warm cache.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from loguru import logger

from .prompt import longest_common_prefix_len


@dataclass
class SyncStats:
    """Result of syncing the KV cache to a target prompt.

    kept: tokens already in the cache (prefix hit). recomputed = total - kept.
    """

    total: int
    kept: int

    @property
    def recomputed(self) -> int:
        return self.total - self.kept

    @property
    def hit_ratio(self) -> float:
        return self.kept / self.total if self.total else 0.0


class PrefillBackend(ABC):
    """Protocol every speculative-prefill backend implements.

    All methods are async; heavy model work runs in a worker thread. The
    backend is the single writer of its KV cache: the caller must not run
    `sync` and `decode_stream` concurrently (the service serializes them with
    a lock).
    """

    @abstractmethod
    async def tokenize_prompt(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> list[int]:
        """Render `messages` with the model's chat template and tokenize.

        `add_generation_prompt=True`, so the returned ids end exactly where
        generation would start. `tools` are OpenAI-format tool definitions
        (from ``to_provider_tools_format``); templates that don't use them
        simply ignore the extra variable.
        """

    @abstractmethod
    async def sync(self, tokens: Sequence[int]) -> SyncStats:
        """Make the KV cache hold exactly `tokens`, reusing the common prefix."""

    @abstractmethod
    def decode_stream(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_token_ids: set[int] | None = None,
        stop_flag: Any,
    ) -> Iterator[str]:
        """Yield generated text pieces, continuing from the prefilled cache.

        A sync generator on purpose: the caller runs it in a thread and
        bridges pieces back to the event loop. `stop_flag` is a
        `threading.Event`; the generator checks it every token so a cancelled
        request stops promptly and the cache stays consistent.
        `stop_token_ids` overrides the backend's own stop set when given.
        """

    @abstractmethod
    async def reset(self) -> None:
        """Drop the whole cache (e.g. when the conversation is reset)."""


def _crop_cache(cache: Any, target_len: int) -> None:
    """Truncate a `DynamicCache` to `target_len` tokens.

    transformers 4.x takes an absolute length; 5.x takes the number of tokens
    to remove as a negative value and rejects positives.
    """
    current = cache.get_seq_length()
    if target_len >= current:
        return
    try:
        cache.crop(target_len)  # transformers 4.x: absolute length
        if cache.get_seq_length() > target_len:
            cache.crop(target_len - current)  # transformers 5.x: negative delta
    except ValueError:
        cache.crop(target_len - current)


def _sample(
    logits: Any, temperature: float, top_p: float, top_k: int
) -> int:  # torch imported lazily by the backend
    """Pick the next token id from last-position logits."""
    import torch

    if temperature is None or temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        threshold = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = (cumulative - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


class TransformersPrefillBackend(PrefillBackend):
    """HuggingFace transformers backend: true in-process KV-cache handoff.

    The cache is a `DynamicCache` kept across turns. Prefill on partial
    transcripts evaluates only the new tokens; the final request usually finds
    the whole prompt cached and goes straight to decoding.
    """

    def __init__(self, model: Any, tokenizer: Any, device: str | None = None):
        """Use a loaded model + tokenizer directly, or `from_pretrained`.

        Args:
            model: A `PreTrainedModel` for causal LM (e.g. Qwen2.5-0.5B-Instruct).
            tokenizer: The matching fast tokenizer (its chat template is used).
            device: Optional device string the model was moved to (e.g. "cuda").
        """
        import torch

        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._cache = None
        self._tokens: list[int] = []
        self._last_logits: Any = None
        self._logits_valid = False
        # Stop ids: the generation config's eos (int or list) plus the tokenizer's.
        self._stop_token_ids: set[int] = set()
        eos = getattr(model.generation_config, "eos_token_id", None)
        if isinstance(eos, int):
            self._stop_token_ids.add(eos)
        elif eos:
            self._stop_token_ids.update(int(e) for e in eos)
        if tokenizer.eos_token_id is not None:
            self._stop_token_ids.add(int(tokenizer.eos_token_id))

    @classmethod
    def from_pretrained(
        cls, model_id: str, *, dtype: Any = None, device_map: str | None = None, **kwargs: Any
    ) -> "TransformersPrefillBackend":
        """Load the model and tokenizer from the Hub and build the backend.

        Args:
            model_id: Hub id, e.g. "Qwen/Qwen2.5-0.5B-Instruct".
            dtype: Torch dtype for weights (e.g. `torch.bfloat16`); default keeps
                the checkpoint's native dtype.
            device_map: e.g. "auto", "cuda:0"; default is CPU.
        """
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map)
        device = None
        if device_map:
            device = next(model.parameters()).device.type
        return cls(model, tokenizer, device=device)

    def _tensor(self, ids: list[int]) -> Any:
        t = self._torch.tensor([ids])
        return t.to(self._device) if self._device else t

    async def tokenize_prompt(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> list[int]:
        def work() -> list[int]:
            text = self._tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False
            )
            return list(self._tokenizer(text, add_special_tokens=False).input_ids)

        return await asyncio.to_thread(work)

    async def sync(self, tokens: Sequence[int]) -> SyncStats:
        target = list(tokens)

        def work() -> SyncStats:
            import torch
            from transformers import DynamicCache

            with torch.inference_mode():
                current = len(self._tokens)
                if self._cache is None:
                    self._cache = DynamicCache()
                lcp = longest_common_prefix_len(self._tokens, target)
                if lcp < current:
                    _crop_cache(self._cache, lcp)
                    self._logits_valid = False
                delta = target[lcp:]
                if delta:
                    out = self._model(
                        input_ids=self._tensor(delta),
                        past_key_values=self._cache,
                        use_cache=True,
                    )
                    self._last_logits = out.logits
                    self._logits_valid = True
                elif not self._logits_valid and target:
                    # Pure rewind to an exact prefix (partial shrank): re-arm
                    # next-token logits by re-forwarding the last kept token.
                    _crop_cache(self._cache, len(target) - 1)
                    out = self._model(
                        input_ids=self._tensor(target[-1:]),
                        past_key_values=self._cache,
                        use_cache=True,
                    )
                    self._last_logits = out.logits
                    self._logits_valid = True
                self._tokens = target
                return SyncStats(total=len(target), kept=lcp)

        return await asyncio.to_thread(work)

    def decode_stream(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_token_ids: set[int] | None = None,
        stop_flag: Any,
    ) -> Iterator[str]:
        import torch

        stops = stop_token_ids if stop_token_ids is not None else self._stop_token_ids
        with torch.inference_mode():
            generated: list[int] = []
            prev_text = ""
            for _ in range(max_new_tokens):
                if stop_flag.is_set():
                    return
                token = _sample(self._last_logits[0, -1].float(), temperature, top_p, top_k)
                if token in stops:
                    return
                generated.append(token)
                # The cache now contains the generated token too; keep the
                # mirror in lockstep so the next sync's LCP is correct.
                self._tokens.append(token)
                text = self._tokenizer.decode(generated, skip_special_tokens=True)
                piece = text[len(prev_text) :] if text.startswith(prev_text) else text
                prev_text = text
                if piece:
                    yield piece
                out = self._model(
                    input_ids=self._tensor([token]),
                    past_key_values=self._cache,
                    use_cache=True,
                )
                self._last_logits = out.logits

    async def reset(self) -> None:
        def work() -> None:
            from transformers import DynamicCache

            self._cache = DynamicCache()
            self._tokens = []
            self._last_logits = None
            self._logits_valid = False

        await asyncio.to_thread(work)
