"""A GGUF model must load if it CAN load, and get the window the card allows.

Two defects, both found by a real load failing on 2026-09-07:

  1. `pooling_type=MEAN` (added for embeddings) is refused outright by some
     architectures. llama.cpp logs "model default pooling_type is [-1], but [1]
     was specified" and context creation fails at EVERY context length, because
     the context size was never the problem. The operator was told "may not fit
     on this GPU at all — try a smaller quantization" while the card was 98%
     free and the file was 7.8 GiB. That advice cannot work: the next
     quantization fails identically.

  2. The ladder started at a fixed GGUF_CONTEXT_LENGTH, so a model declaring
     262144 was served 8192 and nothing said so. That same model creates a
     context at 131072 on this 24 GiB card — 16x what it was being given.

MEASURED, on ByteOtter/Qwen3.8-27B-TAK-Reasoning-GGUF on the RTX 3090:
  * with embeddings, n_ctx 8192/4096/2048 -> all fail
  * without embeddings, n_ctx=2048        -> loads
  * without embeddings, up to n_ctx=131072 -> loads
  * declared n_ctx_train                   -> 262144

MUTATION CONTROLS (each must turn this file red):
  * remove the retry-without-embeddings branch -> "loads without embeddings" fails
  * keep supports_embeddings True on the fallback -> "says so" fails
  * start the ladder at the ceiling again      -> "starts from declared" fails
  * ignore `declared` and always use ceiling   -> same
  * restore "may not fit on this GPU" wording  -> "does not blame VRAM" fails
"""

from unittest.mock import MagicMock, patch

import pytest

from millm.core.config import settings as config_settings
from millm.core.errors import ModelLoadError


class TestEmbeddingsAreDroppedRatherThanTheModel:
    """Embeddings are optional. Serving the model is not."""

    @staticmethod
    def _llama_that_refuses_pooling(record):
        """A Llama stand-in refusing any construction with pooling set.

        This is the REAL failure shape: it depends on the pooling kwarg and not
        at all on n_ctx, which is why every rung failed.
        """

        def _factory(**kwargs):
            record.append(dict(kwargs))
            if kwargs.get("embedding") or kwargs.get("pooling_type") is not None:
                raise ValueError("Failed to create llama_context")
            return MagicMock(name=f"llama(n_ctx={kwargs['n_ctx']})")

        return _factory

    def test_it_loads_without_embeddings_when_pooling_is_refused(self, tmp_path):
        from millm.ml import model_loader

        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x" * 32)
        calls: list[dict] = []

        with patch.object(model_loader, "Llama", self._llama_that_refuses_pooling(calls)), \
             patch.object(model_loader, "declared_context", return_value=8192), \
             patch.object(config_settings, "GGUF_ENABLE_EMBEDDINGS", True), \
             patch.object(config_settings, "GGUF_CONTEXT_LENGTH", 8192):
            loaded = model_loader.load_gguf_model(1, "m", str(tmp_path), "m.gguf")

        assert loaded is not None, (
            "the model was refused entirely because a pooling type it does not "
            "support was requested — the card was never the problem"
        )
        assert any(c.get("embedding") for c in calls), "embeddings were never tried"
        assert not calls[-1].get("embedding"), "the successful call still set embedding"

    def test_it_says_embeddings_are_unavailable(self, tmp_path):
        """A caller must be TOLD, not left to infer it from a runtime failure."""
        from millm.ml import model_loader

        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x" * 32)

        with patch.object(model_loader, "Llama", self._llama_that_refuses_pooling([])), \
             patch.object(model_loader, "declared_context", return_value=8192), \
             patch.object(config_settings, "GGUF_ENABLE_EMBEDDINGS", True), \
             patch.object(config_settings, "GGUF_CONTEXT_LENGTH", 8192):
            loaded = model_loader.load_gguf_model(1, "m", str(tmp_path), "m.gguf")

        assert loaded.supports_embeddings is False

    def test_embeddings_are_kept_when_the_architecture_accepts_them(self, tmp_path):
        """Specificity: the fallback must not fire on models that are fine."""
        from millm.ml import model_loader

        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x" * 32)
        calls: list[dict] = []

        def _accepts(**kwargs):
            calls.append(dict(kwargs))
            return MagicMock()

        with patch.object(model_loader, "Llama", _accepts), \
             patch.object(model_loader, "declared_context", return_value=8192), \
             patch.object(config_settings, "GGUF_ENABLE_EMBEDDINGS", True), \
             patch.object(config_settings, "GGUF_CONTEXT_LENGTH", 8192):
            loaded = model_loader.load_gguf_model(1, "m", str(tmp_path), "m.gguf")

        assert loaded.supports_embeddings is True
        assert len(calls) == 1, "it should not have retried anything"

    def test_a_genuinely_unloadable_model_does_not_blame_the_GPU(self, tmp_path):
        """The message that sent an operator to download another model.

        With the card 98% free, "may not fit on this GPU at all — try a smaller
        quantization" is not merely unhelpful, it is a wrong instruction whose
        next step fails identically.
        """
        from millm.ml import model_loader

        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x" * 32)

        def _always_fails(**kwargs):
            raise ValueError("Failed to create llama_context")

        with patch.object(model_loader, "Llama", _always_fails), \
             patch.object(model_loader, "declared_context", return_value=8192), \
             patch.object(config_settings, "GGUF_ENABLE_EMBEDDINGS", True), \
             patch.object(config_settings, "GGUF_CONTEXT_LENGTH", 8192):
            with pytest.raises(ModelLoadError) as exc:
                model_loader.load_gguf_model(1, "m", str(tmp_path), "m.gguf")

        message = str(exc.value)
        assert "may not fit on this GPU" not in message, (
            "the loader asserted a cause it had not established"
        )
        assert "smaller quantization" not in message
        assert "embeddings disabled" in message, (
            "it must say the fallback was already tried, or the reader will "
            "suggest it"
        )


class TestTheLadderStartsFromWhatTheModelDeclares:
    """A model trained for 262144 was being served 8192, silently."""

    @staticmethod
    def _record_n_ctx(calls):
        def _factory(**kwargs):
            calls.append(kwargs["n_ctx"])
            return MagicMock()

        return _factory

    def test_it_starts_from_the_declared_context(self, tmp_path):
        from millm.ml import model_loader

        (tmp_path / "m.gguf").write_bytes(b"x" * 32)
        calls: list[int] = []

        with patch.object(model_loader, "Llama", self._record_n_ctx(calls)), \
             patch.object(model_loader, "declared_context", return_value=16384), \
             patch.object(config_settings, "GGUF_ENABLE_EMBEDDINGS", False), \
             patch.object(config_settings, "GGUF_CONTEXT_LENGTH", 32768):
            loaded = model_loader.load_gguf_model(1, "m", str(tmp_path), "m.gguf")

        assert calls[0] == 16384, (
            "asking for more than the model was trained on is not a bigger "
            "window, it is a different model's number"
        )
        assert loaded.context_length == 16384

    def test_the_ceiling_bounds_a_model_that_declares_more(self, tmp_path):
        """llama.cpp allocates the WHOLE KV cache at context creation, and this
        GPU is shared with miStudio's training and steering work."""
        from millm.ml import model_loader

        (tmp_path / "m.gguf").write_bytes(b"x" * 32)
        calls: list[int] = []

        with patch.object(model_loader, "Llama", self._record_n_ctx(calls)), \
             patch.object(model_loader, "declared_context", return_value=262144), \
             patch.object(config_settings, "GGUF_ENABLE_EMBEDDINGS", False), \
             patch.object(config_settings, "GGUF_CONTEXT_LENGTH", 32768):
            model_loader.load_gguf_model(1, "m", str(tmp_path), "m.gguf")

        assert calls[0] == 32768

    def test_an_unreadable_declaration_falls_back_to_the_ceiling(self, tmp_path):
        """The probe is an optimisation, never a gate."""
        from millm.ml import model_loader

        (tmp_path / "m.gguf").write_bytes(b"x" * 32)
        calls: list[int] = []

        with patch.object(model_loader, "Llama", self._record_n_ctx(calls)), \
             patch.object(model_loader, "declared_context", return_value=None), \
             patch.object(config_settings, "GGUF_ENABLE_EMBEDDINGS", False), \
             patch.object(config_settings, "GGUF_CONTEXT_LENGTH", 8192):
            model_loader.load_gguf_model(1, "m", str(tmp_path), "m.gguf")

        assert calls[0] == 8192

    def test_zero_still_means_the_files_full_context(self, tmp_path):
        """The documented escape hatch must survive the ceiling change."""
        from millm.ml import model_loader

        (tmp_path / "m.gguf").write_bytes(b"x" * 32)
        calls: list[int] = []

        with patch.object(model_loader, "Llama", self._record_n_ctx(calls)), \
             patch.object(model_loader, "declared_context", return_value=262144), \
             patch.object(config_settings, "GGUF_ENABLE_EMBEDDINGS", False), \
             patch.object(config_settings, "GGUF_CONTEXT_LENGTH", 0):
            model_loader.load_gguf_model(1, "m", str(tmp_path), "m.gguf")

        assert calls == [0], "0 means 'let llama.cpp use the file's own context'"


class TestTheKVCacheIsNotLeftAtF16:
    """The KV cache is the biggest lever on context, and it sat at the default.

    gemma-4-31b spends 630 KB PER TOKEN at f16 — 60 layers x 16 KV heads x 168
    head_dim x 2 tensors x 2 bytes. At 4096 that is 2.6 GiB, more than a third
    of the free VRAM on a 24 GiB card, to hold about three thousand words. The
    context ladder searched for a length that fit while the cost of a token
    went unquestioned.

    MEASURED on gemma-4-31b IQ4_XS on the RTX 3090:
        f16  + FA : 8192 FAILS      (4096 is the ceiling)
        q8_0 + FA : 12288 LOADS     (3x, clears the ~4700-token labeling prompt)
        q4_0 + FA : 16384 LOADS
        q8_0, NO FA: 8192 FAILS     <- flash attention is a DEPENDENCY

    MUTATION CONTROLS (each must turn this file red):
      * return {} from _kv_cache_kwargs          -> "quantized by default" fails
      * pair quantization with flash_attn=False  -> "refuses the pairing" fails
      * drop the f16 retry in load_gguf_model    -> "falls back" fails
    """

    def test_the_default_is_a_quantized_cache_with_flash_attention(self):
        from millm.core.config import settings
        from millm.ml.model_loader import _kv_cache_kwargs

        kwargs = _kv_cache_kwargs(
            settings.GGUF_KV_CACHE_TYPE, settings.GGUF_FLASH_ATTENTION
        )

        assert kwargs.get("flash_attn") is True
        assert kwargs["type_k"] == kwargs["type_v"], "K and V must match"
        assert kwargs["type_k"] != _f16(), (
            "shipping f16 wastes half the KV cache; gemma-4-31b drops from "
            "12288 to 4096 tokens on the same card"
        )

    def test_quantization_without_flash_attention_is_REFUSED(self):
        """Not a preference. q8_0 without FA fails at 8192, a length it reaches
        12288 with it — so the pairing is strictly worse than plain f16."""
        from millm.ml.model_loader import _kv_cache_kwargs

        assert _kv_cache_kwargs("q8_0", False) == {}

    def test_f16_passes_nothing_and_stays_the_documented_escape_hatch(self):
        from millm.ml.model_loader import _kv_cache_kwargs

        assert _kv_cache_kwargs("f16", True) == {}

    def test_an_unknown_type_falls_back_rather_than_crashing(self):
        """A typo in config must not make every GGUF model unloadable."""
        from millm.ml.model_loader import _kv_cache_kwargs

        assert _kv_cache_kwargs("q3_k_m_typo", True) == {}

    def test_q4_is_available_but_is_not_the_default(self):
        """It buys a third more window at a real accuracy cost, which is the
        wrong trade for a judge whose whole job is discrimination."""
        from millm.core.config import settings
        from millm.ml.model_loader import _kv_cache_kwargs

        assert _kv_cache_kwargs("q4_0", True)["type_k"] is not None
        assert settings.GGUF_KV_CACHE_TYPE == "q8_0"

    def test_a_build_that_refuses_the_quantized_cache_still_gets_a_model(
        self, tmp_path
    ):
        """Same shape as the pooling defect: a kwarg the build rejects fails
        every rung for a reason unrelated to context, and cost a working model."""
        from millm.ml import model_loader

        (tmp_path / "m.gguf").write_bytes(b"x" * 32)
        calls: list[dict] = []

        def _refuses_quantized_kv(**kwargs):
            calls.append(dict(kwargs))
            if "type_k" in kwargs:
                raise ValueError("Failed to create llama_context")
            return MagicMock()

        with patch.object(model_loader, "Llama", _refuses_quantized_kv), \
             patch.object(model_loader, "declared_context", return_value=8192), \
             patch.object(config_settings, "GGUF_ENABLE_EMBEDDINGS", False), \
             patch.object(config_settings, "GGUF_KV_CACHE_TYPE", "q8_0"), \
             patch.object(config_settings, "GGUF_FLASH_ATTENTION", True), \
             patch.object(config_settings, "GGUF_CONTEXT_LENGTH", 8192):
            loaded = model_loader.load_gguf_model(1, "m", str(tmp_path), "m.gguf")

        assert loaded is not None, "a refused cache type cost the whole model"
        assert any("type_k" in c for c in calls), "quantization was never tried"
        assert "type_k" not in calls[-1], "the successful call still quantized"


def _f16():
    from millm.ml.model_loader import _KV_CACHE_TYPES

    return _KV_CACHE_TYPES["f16"]


class TestTheLadderRecoversWhatHalvingSkips:
    """Halving lands within a FACTOR OF TWO of the true ceiling, not on it.

    MEASURED on gemma-4-31b IQ4_XS: the ladder settles at 8192 while 12288
    loads on the same card. Half the usable window discarded for the sake of a
    tidy sequence — and 8192 vs 12288 is the difference between one labeling
    prompt fitting and several.

    MUTATION CONTROLS:
      * delete the bisection block          -> "reaches the true ceiling" fails
      * bisect on a gap of any size         -> "does not probe a narrow gap" fails
      * keep probing without the 3-cap      -> "is bounded" fails
    """

    @staticmethod
    def _ceiling_at(limit, calls):
        def _factory(**kwargs):
            calls.append(kwargs["n_ctx"])
            if kwargs["n_ctx"] > limit:
                raise ValueError("Failed to create llama_context")
            return MagicMock(close=MagicMock())

        return _factory

    def test_it_reaches_a_ceiling_the_ladder_steps_over(self, tmp_path):
        from millm.ml import model_loader

        (tmp_path / "m.gguf").write_bytes(b"x" * 32)
        calls: list[int] = []

        # True ceiling 12288: the ladder sees 32768 and 16384 fail, 8192 work.
        with patch.object(model_loader, "Llama", self._ceiling_at(12288, calls)), \
             patch.object(model_loader, "declared_context", return_value=32768), \
             patch.object(config_settings, "GGUF_ENABLE_EMBEDDINGS", False), \
             patch.object(config_settings, "GGUF_CONTEXT_LENGTH", 32768):
            loaded = model_loader.load_gguf_model(1, "m", str(tmp_path), "m.gguf")

        assert loaded.context_length > 8192, (
            f"halving settled at {loaded.context_length}; a window up to 12288 "
            "fits and was never tried"
        )
        assert loaded.context_length <= 12288, "it must not claim more than fits"

    def test_it_is_bounded_and_does_not_probe_forever(self, tmp_path):
        """Each probe is a real model load. The gain halves every time; the
        cost does not."""
        from millm.ml import model_loader

        (tmp_path / "m.gguf").write_bytes(b"x" * 32)
        calls: list[int] = []

        with patch.object(model_loader, "Llama", self._ceiling_at(12288, calls)), \
             patch.object(model_loader, "declared_context", return_value=32768), \
             patch.object(config_settings, "GGUF_ENABLE_EMBEDDINGS", False), \
             patch.object(config_settings, "GGUF_CONTEXT_LENGTH", 32768):
            model_loader.load_gguf_model(1, "m", str(tmp_path), "m.gguf")

        # 3 descending rungs + at most 3 bisection probes.
        assert len(calls) <= 6, f"{len(calls)} model loads is too many: {calls}"

    def test_it_does_not_probe_a_gap_too_narrow_to_matter(self, tmp_path):
        """A model load to win 1024 tokens is not worth the seconds."""
        from millm.ml import model_loader

        (tmp_path / "m.gguf").write_bytes(b"x" * 32)
        calls: list[int] = []

        with patch.object(model_loader, "Llama", self._ceiling_at(2048, calls)), \
             patch.object(model_loader, "declared_context", return_value=4096), \
             patch.object(config_settings, "GGUF_ENABLE_EMBEDDINGS", False), \
             patch.object(config_settings, "GGUF_CONTEXT_LENGTH", 4096):
            model_loader.load_gguf_model(1, "m", str(tmp_path), "m.gguf")

        assert calls == [4096, 2048], f"it probed a 2048-token gap: {calls}"


class TestAFailureTheLadderCannotHelpWithStopsAtOnce:
    """A model whose WEIGHTS fail to load was retried fifteen times.

    Observed live on 2026-09-07: "Failed to load model from file" was retried at
    five context lengths, then blamed on the KV cache type, then on the pooling
    type — 36 seconds and three wrong diagnoses for one honest error. The
    ladder and both capability fallbacks exist for ONE failure: llama.cpp
    declining to create a context.

    MUTATION CONTROL: make _is_context_related always return True -> both fail.
    """

    def test_a_weights_failure_is_not_retried(self, tmp_path):
        from millm.ml import model_loader

        (tmp_path / "m.gguf").write_bytes(b"x" * 32)
        calls: list[int] = []

        def _weights_fail(**kwargs):
            calls.append(kwargs["n_ctx"])
            raise ValueError("Failed to load model from file: /data/m.gguf")

        with patch.object(model_loader, "Llama", _weights_fail), \
             patch.object(model_loader, "declared_context", return_value=32768), \
             patch.object(config_settings, "GGUF_ENABLE_EMBEDDINGS", True), \
             patch.object(config_settings, "GGUF_CONTEXT_LENGTH", 32768):
            with pytest.raises(Exception):
                model_loader.load_gguf_model(1, "m", str(tmp_path), "m.gguf")

        assert len(calls) == 1, (
            f"one unrelated error became {len(calls)} load attempts: {calls}"
        )

    def test_a_real_context_failure_still_walks_the_ladder(self):
        """Specificity: the discriminator must not reject the case the ladder
        exists for."""
        from millm.ml.model_loader import _is_context_related

        assert _is_context_related(ValueError("Failed to create llama_context"))
        assert not _is_context_related(ValueError("Failed to load model from file"))


class TestTheReportedContextIsTheOneThatLoaded:
    """A failed bisection probe must not become the reported window.

    `_bisect_upward` assigns kwargs["n_ctx"] before each probe. A probe that
    FAILS left it there, so a model actually serving 4096 reported 5120 — a
    length that had just been rejected. Every consumer trusts this number:
    /api/health/inference publishes it, millm_list_models shows it, and a
    prompt sized against it would overflow.

    Caught by the pre-existing ladder test, which is the only reason it is not
    in production.

    MUTATION CONTROL: delete the unconditional `kwargs["n_ctx"] = reached`
    restore -> this fails with 5120 != 4096.
    """

    def test_a_failed_probe_does_not_inflate_the_reported_context(self, tmp_path):
        from millm.ml import model_loader

        (tmp_path / "m.gguf").write_bytes(b"x" * 32)
        loaded_at: list[int] = []

        def _ceiling_4096(**kwargs):
            n = kwargs["n_ctx"]
            if n > 4096:
                raise ValueError("Failed to create llama_context")
            loaded_at.append(n)
            return MagicMock(close=MagicMock())

        with patch.object(model_loader, "Llama", _ceiling_4096), \
             patch.object(model_loader, "declared_context", return_value=8192), \
             patch.object(config_settings, "GGUF_ENABLE_EMBEDDINGS", False), \
             patch.object(config_settings, "GGUF_CONTEXT_LENGTH", 8192):
            loaded = model_loader.load_gguf_model(1, "m", str(tmp_path), "m.gguf")

        assert loaded.context_length == max(loaded_at), (
            f"reported {loaded.context_length} but the largest context that "
            f"actually constructed was {max(loaded_at)}"
        )
        assert loaded.context_length == 4096
