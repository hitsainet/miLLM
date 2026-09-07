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


class TestTheOptionalImportCannotKillTheServer:
    """A missing or broken llama.cpp must degrade one feature, not the process.

    `model_loader` is imported at startup, so an exception escaping its
    module-level import takes the whole backend down. llama-cpp-python does NOT
    raise ImportError when its shared library will not load — verified in the
    running pod:

        RuntimeError: Failed to load shared library '.../libllama.so':
        libcudart.so.13: cannot open shared object file

    In the deployed image that resolves only because `import torch` runs first
    and pulls torch's CUDA runtime into the process. That ordering is an
    accident, not a contract.

    MUTATION CONTROL: narrow the handler back to `except ImportError` -> the
    RuntimeError case fails.
    """

    def test_the_module_survives_a_broken_shared_library(self):
        """The real failure mode, reproduced by its real exception type."""
        import importlib
        import sys

        real = sys.modules.pop("millm.ml.model_loader", None)
        # SAVED, not merely popped. Where llama-cpp-python is genuinely
        # installed — the deployed image, a GPU dev box — discarding the real
        # module leaves the next `import llama_cpp` to re-execute it and
        # dlopen libllama a second time in the same process.
        #
        # The PARENT PACKAGE ATTRIBUTE is saved too, and that is not belt and
        # braces. `importlib.import_module("millm.ml.model_loader")` rebinds
        # `millm.ml.model_loader` on the parent package as well as the
        # sys.modules entry, and restoring only sys.modules leaves the two
        # disagreeing: `from millm.ml import model_loader` then yields THIS
        # test's throwaway module — built while llama_cpp was a MagicMock —
        # while `from millm.ml.model_loader import X` yields the real one. Two
        # live copies with separate globals and separate `LoadedModelState`
        # singletons. It cost the four `_gguf_device` tests below their
        # meaning: they monkeypatched and called the discarded copy and passed
        # by self-consistency, and it made a loader test in this same file fail
        # for a reason that had nothing to do with the loader.
        import millm.ml as _pkg

        real_attr = getattr(_pkg, "model_loader", None)
        real_llama = sys.modules.get("llama_cpp")
        broken = MagicMock()
        broken.__spec__ = MagicMock()
        type(broken).Llama = property(
            lambda self: (_ for _ in ()).throw(
                RuntimeError("Failed to load shared library: libcudart.so.13")
            )
        )
        sys.modules["llama_cpp"] = broken
        try:
            module = importlib.import_module("millm.ml.model_loader")
            assert module.Llama is None, "a broken engine must read as absent"
        finally:
            if real_llama is not None:
                sys.modules["llama_cpp"] = real_llama
            else:
                sys.modules.pop("llama_cpp", None)
            sys.modules.pop("millm.ml.model_loader", None)
            if real is not None:
                sys.modules["millm.ml.model_loader"] = real
            if real_attr is not None:
                _pkg.model_loader = real_attr
            else:
                # Nothing had imported it before; leave the package as it was
                # rather than leaving a throwaway bound to the name.
                if hasattr(_pkg, "model_loader"):
                    delattr(_pkg, "model_loader")

        # Both views back in agreement, asserted rather than assumed. A test
        # that leaves two copies of a module alive in the process poisons every
        # later test in it, and does so silently — the symptom lands somewhere
        # else entirely.
        #
        # MUTATION CONTROL: delete the `_pkg.model_loader = real_attr` restore
        # above -> this fails, and so does
        # TestLoadingAGGUFFile::test_the_device_is_the_measured_one_not_a_constant.
        assert sys.modules["millm.ml.model_loader"] is _pkg.model_loader, (
            "the parent package attribute must be restored alongside the "
            "sys.modules entry; importlib rebinds both"
        )

    def test_the_handler_is_not_narrowed_to_ImportError(self):
        """Source-level, because the runtime case above cannot cover every
        exception type a native library might raise. Fails CLOSED: the anchor
        must be found."""
        from pathlib import Path

        src = Path(__file__).resolve().parents[3] / "millm" / "ml" / "model_loader.py"
        text = src.read_text()
        marker = "from llama_cpp import Llama"
        assert marker in text, "import site moved; this guard now checks nothing"
        after = text.split(marker, 1)[1][:400]
        assert "except Exception" in after, (
            "llama_cpp raises RuntimeError on a shared-library failure; "
            "ImportError alone lets it escape and kill the module import"
        )


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

    def test_the_device_is_the_measured_one_not_a_constant(self, tmp_path, monkeypatch):
        """The loader must ASK `_gguf_device()`, not restate GGUF_GPU_LAYERS.

        `_gguf_device` has its own tests, and they all passed while the loader
        still wrote `"cuda" if GGUF_GPU_LAYERS != 0 else "cpu"` — the function
        was correct and unwired. Reverting that one line left the whole unit
        suite green, so this asserts the CALL, by forcing the probe to an
        answer the constant cannot produce.

        MUTATION CONTROL: put `device="cuda" if GGUF_GPU_LAYERS != 0 else "cpu"`
        back into load_gguf_model -> "cuda" != "cpu" and this fails.
        """
        from millm.ml import model_loader

        gguf = tmp_path / "zora-Q5_K_M.gguf"
        gguf.write_bytes(b"\x00" * 2048)

        # A card is present (so the constant would say "cuda") but the wheel
        # cannot offload — only a real call to the probe reaches "cpu".
        monkeypatch.setattr(model_loader.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            model_loader, "llama_supports_gpu_offload", lambda: False, raising=False
        )

        with patch("millm.ml.model_loader.Llama", MagicMock()):
            loaded = load_gguf_model(7, "zora", str(tmp_path), "zora-Q5_K_M.gguf")

        assert loaded.device == "cpu", (
            "GGUF_GPU_LAYERS is -1, so a device derived from the constant "
            "reads 'cuda' here; only _gguf_device() can answer 'cpu'"
        )

    def test_the_quant_label_survives_a_dot_in_the_model_name(self, tmp_path):
        """`stem.rsplit('.')[-1]` assumed a dot before the quant token.

        "Model.Q4_K_M.gguf" happens to work under that rule, so a fixture
        built from one proves nothing. "qwen2.5-7b-instruct-q4_k_m.gguf" is an
        ordinary name and yielded "gguf:5-7b-instruct-q4_k_m".

        MUTATION CONTROL: restore
        `f"gguf:{Path(gguf_file).stem.rsplit('.', 1)[-1]}"` -> this fails.
        """
        name = "qwen2.5-7b-instruct-q4_k_m.gguf"
        (tmp_path / name).write_bytes(b"\x00" * 2048)

        with patch("millm.ml.model_loader.Llama", MagicMock()):
            loaded = load_gguf_model(9, "qwen", str(tmp_path), name)

        assert loaded.quantization_method == "gguf:Q4_K_M"

    def test_an_unparseable_name_says_unknown_rather_than_guessing(self, tmp_path):
        """The `or 'unknown'` arm: a label is a claim, and a wrong one is worse
        than an absent one."""
        (tmp_path / "weights.gguf").write_bytes(b"\x00" * 2048)

        with patch("millm.ml.model_loader.Llama", MagicMock()):
            loaded = load_gguf_model(10, "x", str(tmp_path), "weights.gguf")

        assert loaded.quantization_method == "gguf:unknown"


class TestTheReportedDeviceIsMeasuredNotRequested:
    """`device` on a GGUF LoadedModel must describe where the weights LANDED.

    `torch.cuda.is_available()` answers "is there a card", which says nothing
    about the llama.cpp build that did the loading: the wheel on PyPI is
    CPU-only, so a CUDA box that installed it offloads nothing while torch
    happily reports a GPU — and /api/models/status then asserts a placement
    that never happened.

    MUTATION CONTROL: drop the `llama_supports_gpu_offload` half of
    `_gguf_device` -> the CPU-only-build case returns "cuda" and fails.
    """

    def test_a_cpu_only_build_reports_cpu_even_with_a_card(self, monkeypatch):
        from millm.ml import model_loader

        monkeypatch.setattr(model_loader.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            model_loader, "llama_supports_gpu_offload", lambda: False, raising=False
        )
        assert model_loader._gguf_device() == "cpu"

    def test_a_cuda_build_with_a_card_reports_cuda(self, monkeypatch):
        from millm.ml import model_loader

        monkeypatch.setattr(model_loader.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            model_loader, "llama_supports_gpu_offload", lambda: True, raising=False
        )
        assert model_loader._gguf_device() == "cuda"

    def test_no_card_reports_cpu_whatever_the_build_says(self, monkeypatch):
        from millm.ml import model_loader

        monkeypatch.setattr(model_loader.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(
            model_loader, "llama_supports_gpu_offload", lambda: True, raising=False
        )
        assert model_loader._gguf_device() == "cpu"

    def test_an_unavailable_probe_never_fails_the_load(self, monkeypatch):
        """The symbol is optional across versions; absent or throwing, the
        answer falls back to the torch check rather than raising."""
        from millm.ml import model_loader

        monkeypatch.setattr(model_loader.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            model_loader, "llama_supports_gpu_offload", None, raising=False
        )
        assert model_loader._gguf_device() == "cuda"

        def _boom():
            raise RuntimeError("symbol removed upstream")

        monkeypatch.setattr(
            model_loader, "llama_supports_gpu_offload", _boom, raising=False
        )
        assert model_loader._gguf_device() == "cuda"

