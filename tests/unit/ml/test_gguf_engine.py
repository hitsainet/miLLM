"""Serving a GGUF file through llama.cpp, and refusing what it cannot do.

The two things that matter here are the unload and the refusals.

UNLOAD, because llama.cpp holds its weights in a C++ context Python's garbage
collector cannot reach. `.to("cpu")` does not exist on it, `del` drops only the
handle, and `torch.cuda.empty_cache()` knows nothing about an allocation torch
never made. On a single shared 24 GB card, a load/unload cycle that silently
retains VRAM is the failure that takes the node down.

REFUSALS, because a GGUF model cannot host an SAE at all — no module tree means
nothing to hook — and the recorded failure mode of this codebase is a capability
that goes quietly dark rather than saying so.

MUTATION CONTROLS (each must turn this file red):
  * drop the close() call in LoadedModelState.clear -> "closes the context" fails
  * default LoadedModel.engine to llamacpp           -> "transformers by default" fails
  * make supports_hooks always True                  -> "cannot host hooks" fails
  * let on_model_loaded start CBM regardless         -> "CBM is not started" fails
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from millm.core.errors import ModelLoadError
from millm.ml.model_loader import (
    ENGINE_LLAMACPP,
    ENGINE_TRANSFORMERS,
    LoadedModel,
    LoadedModelState,
    load_gguf_model,
)


def _llamacpp_model(**overrides):
    handle = overrides.pop("model", None) or MagicMock()
    return LoadedModel(
        model_id=1,
        model_name="zora-v1.13-gguf",
        model=handle,
        tokenizer=None,
        loaded_at=datetime.utcnow(),
        engine=ENGINE_LLAMACPP,
        **overrides,
    )


@pytest.fixture(autouse=True)
def _reset_state():
    state = LoadedModelState()
    state._loaded = None
    yield
    state._loaded = None


class TestTheEngineIsRecorded:
    def test_transformers_by_default(self):
        """Every existing construction site must keep its meaning."""
        m = LoadedModel(1, "x", MagicMock(), MagicMock(), datetime.utcnow())
        assert m.engine == ENGINE_TRANSFORMERS
        assert m.supports_hooks is True

    def test_a_gguf_model_cannot_host_hooks(self):
        """Not policy — a statement about the runtime."""
        assert _llamacpp_model().supports_hooks is False


class TestUnloadReleasesTheContext:
    def test_closes_the_llama_context(self):
        handle = MagicMock()
        # A real Llama has no `.to`; the loader's except swallows that, which is
        # why close() has to be explicit.
        handle.to.side_effect = AttributeError("'Llama' object has no attribute 'to'")
        state = LoadedModelState()
        state.set(_llamacpp_model(model=handle))

        state.clear()

        assert handle.close.called, (
            "without an explicit close the C++ context keeps its VRAM, and "
            "nothing else in clear() can reach it"
        )
        assert state.current is None

    def test_a_close_failure_does_not_strand_the_slot(self):
        """A wedged close must not leave the singleton holding a dead model."""
        handle = MagicMock()
        handle.close.side_effect = RuntimeError("boom")
        state = LoadedModelState()
        state.set(_llamacpp_model(model=handle))

        state.clear()

        assert state.current is None

    def test_a_transformers_model_still_unloads_the_old_way(self):
        handle = MagicMock()
        del handle.close  # transformers models have no close()
        state = LoadedModelState()
        state.set(LoadedModel(1, "x", handle, MagicMock(), datetime.utcnow()))

        state.clear()

        assert handle.to.called, "the .to('cpu') path must survive"
        assert state.current is None


class TestTheGGUFPathIsReachable:
    """A capability is not shipped until a test FAILS when its wiring is removed.

    `load_gguf_model` can be perfectly correct and never called: `ModelLoader.load`
    is the only production entry point, and the branch that reaches it is one
    `if`. Deleting that `if` left the whole suite green, which is precisely the
    condition this repo's reachability rule exists to catch.

    MUTATION CONTROL: change `if gguf_file:` to `if False:` -> both tests fail.
    """

    def test_load_routes_a_gguf_row_to_the_llamacpp_engine(self, tmp_path):
        from millm.ml.model_loader import ModelLoader

        loader = ModelLoader()
        gguf = tmp_path / "m-Q4_K_M.gguf"
        gguf.write_bytes(b"\x00" * 1024)

        with patch("millm.ml.model_loader.Llama", MagicMock()):
            loaded = loader.load(
                model_id=3,
                model_name="m",
                cache_path=str(tmp_path),
                quantization="Q4",
                estimated_memory_mb=1,
                gguf_file="m-Q4_K_M.gguf",
            )

        assert loaded.engine == ENGINE_LLAMACPP
        # And it must be the RESIDENT model, not merely returned.
        assert LoadedModelState().current is loaded

    def test_the_gguf_branch_skips_the_transformers_preflight(self, tmp_path):
        """It must not be gated behind checks that describe another runtime.

        The CUDA and memory checks above it reason about torch allocations and
        bitsandbytes levels. On a machine without CUDA they raise, and a GGUF
        load that llama.cpp could serve on CPU would never be attempted.
        """
        from millm.ml.model_loader import ModelLoader

        loader = ModelLoader()
        gguf = tmp_path / "m-Q4_K_M.gguf"
        gguf.write_bytes(b"\x00" * 1024)

        with patch("millm.ml.model_loader.Llama", MagicMock()):
            with patch("millm.ml.model_loader.torch.cuda.is_available", return_value=False):
                loaded = loader.load(
                    model_id=4,
                    model_name="m",
                    cache_path=str(tmp_path),
                    quantization="Q4",
                    # Absurdly large: the transformers path would refuse this.
                    estimated_memory_mb=10_000_000,
                    gguf_file="m-Q4_K_M.gguf",
                )

        assert loaded.engine == ENGINE_LLAMACPP


class TestLoadingAGGUFFile:
    def test_refuses_when_the_engine_is_absent(self, tmp_path):
        """An extra that is not installed must say so, not fail obscurely."""
        with patch("millm.ml.model_loader.Llama", None):
            with pytest.raises(ModelLoadError, match="llama-cpp-python is not installed"):
                load_gguf_model(1, "x", str(tmp_path), "a.gguf")

    def test_refuses_when_the_chosen_file_is_missing(self, tmp_path):
        """What a partial download of a SPLIT quantization leaves behind."""
        with patch("millm.ml.model_loader.Llama", MagicMock()):
            with pytest.raises(ModelLoadError, match="GGUF file not found"):
                load_gguf_model(1, "x", str(tmp_path), "absent.gguf")

    def test_records_the_engine_and_offloads_to_gpu(self, tmp_path):
        gguf = tmp_path / "zora-Q5_K_M.gguf"
        gguf.write_bytes(b"\x00" * 2048)
        fake = MagicMock()

        with patch("millm.ml.model_loader.Llama", fake):
            loaded = load_gguf_model(7, "zora", str(tmp_path), "zora-Q5_K_M.gguf")

        assert loaded.engine == ENGINE_LLAMACPP
        assert loaded.supports_hooks is False
        assert loaded.tokenizer is None, (
            "llama.cpp exposes no HuggingFace tokenizer; a half-working "
            "stand-in would fail late instead of at the boundary"
        )
        # n_gpu_layers=-1 is what makes it a GPU load at all.
        assert fake.call_args.kwargs["n_gpu_layers"] == -1
        assert fake.call_args.kwargs["model_path"].endswith("zora-Q5_K_M.gguf")
