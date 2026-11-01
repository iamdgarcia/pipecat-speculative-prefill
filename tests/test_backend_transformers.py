#
# Copyright (c) 2026, Daniel Garcia Peña (iamdgarcia)
#
# SPDX-License-Identifier: MIT
#

"""Real-backend tests against a tiny in-repo-cached Llama model (~5MB).

The load-bearing assertion: incremental prefill (prefix cached, crop, delta
eval) produces the same next-token logits as a direct full prefill. That is
the whole correctness case for speculative KV prefill.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from pipecat_speculative_prefill.backends import (  # noqa: E402
    TransformersPrefillBackend,
    _crop_cache,
)
from pipecat_speculative_prefill.prompt import build_messages  # noqa: E402

MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def backend():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    return TransformersPrefillBackend(model, tokenizer)


def _ids(backend, user_text, history=None):
    messages = build_messages(
        history or [], "You are helpful.", partial_user_text=user_text
    )
    return backend.tokenize_prompt(messages)


class TestTransformersBackend:
    async def test_delta_eval_equals_full_prefill(self, backend):
        """Crop + delta eval must reproduce the full prefill's logits exactly."""
        from transformers import DynamicCache

        full = await _ids(backend, "hello there friend")
        half = len(full) // 2

        cache_full = DynamicCache()
        with torch.inference_mode():
            logits_full = backend._model(
                input_ids=torch.tensor([full]), past_key_values=cache_full, use_cache=True
            ).logits

        await backend.sync(full[:half])
        await backend.sync(full)  # crop + delta eval path

        assert (backend._last_logits[0, -1] - logits_full[0, -1]).abs().max().item() == 0.0

    async def test_append_only_partial_keeps_prefix(self, backend):
        short = await _ids(backend, "hello")
        long = await _ids(backend, "hello there")

        await backend.reset()
        stats1 = await backend.sync(short)
        assert stats1.kept == 0

        stats2 = await backend.sync(long)
        # Subword merges mean the boundary isn't exact; what matters is that
        # most of the prefix was kept and only the new tail recomputed.
        assert stats2.kept >= stats2.total - 8
        assert stats2.kept > 0
        assert stats2.total == len(long)

    async def test_rewind_on_divergence(self, backend):
        first = await _ids(backend, "hello wor")
        second = await _ids(backend, "hello moon")  # same prefix, different tail

        await backend.reset()
        await backend.sync(first)
        stats = await backend.sync(second)

        from pipecat_speculative_prefill.prompt import longest_common_prefix_len

        assert stats.kept == longest_common_prefix_len(first, second)
        assert stats.recomputed > 0
        # Cache must hold exactly the new tokens.
        assert backend._cache.get_seq_length() == len(second)
        assert backend._tokens == second

    async def test_decode_stream_from_warm_cache(self, backend):
        ids = await _ids(backend, "say something")
        await backend.reset()
        stats = await backend.sync(ids)
        assert stats.total == len(ids)

        import threading

        stop = threading.Event()
        pieces = list(
            backend.decode_stream(
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                stop_token_ids=set(),
                stop_flag=stop,
            )
        )
        assert len(pieces) > 0
        assert "".join(pieces) == "".join(pieces)  # sanity: text is str-joinable

    async def test_max_new_tokens_respected(self, backend):
        ids = await _ids(backend, "count")
        await backend.reset()
        await backend.sync(ids)

        import threading

        stop = threading.Event()
        pieces = list(
            backend.decode_stream(
                max_new_tokens=3,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                stop_token_ids=set(),
                stop_flag=stop,
            )
        )
        assert len(pieces) <= 3


class TestCropCompat:
    def test_crop_shrinks_and_grows_noop(self, backend):
        from transformers import DynamicCache

        cache = DynamicCache()
        with torch.inference_mode():
            backend._model(input_ids=torch.tensor([[1, 2, 3, 4, 5]]), past_key_values=cache)
        _crop_cache(cache, 3)
        assert cache.get_seq_length() == 3
        _crop_cache(cache, 5)  # grow is a no-op
        assert cache.get_seq_length() == 3
        _crop_cache(cache, 0)
        assert cache.get_seq_length() == 0
