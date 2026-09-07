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
