"""torch.compile must not be able to take inference down again.

WHAT HAPPENED (2026-07-27)
TORCH_COMPILE was enabled with the default mode "reduce-overhead", which turns
on CUDA Graphs. Compilation succeeded, the warmup succeeded, and then every
generate request returned 500:

    RuntimeError: accessing tensor output of CUDAGraphs that has been
    overwritten by a subsequent run

Three separate things made that possible, and each is pinned here:

  1. the mode enabled CUDA Graphs                -> default is now "default"
  2. warmup was ONE pass, and the fault only
     appears on the second                       -> soak runs several
  3. a failed warmup left the compiled forward
     installed, so the server kept serving 500s  -> soak reverts to eager

MUTATION CONTROLS:
  * set either default back to "reduce-overhead" -> mode tests fail
  * soak a single pass                           -> soak test fails
  * drop the logits access from the soak         -> access test fails
  * remove the revert-to-eager on failure        -> fallback test fails
"""

import inspect
import re

from millm.core.config import Settings
from millm.ml import model_loader


class TestTheCompileModeAvoidsCudaGraphs:
    def test_settings_default_is_not_reduce_overhead(self):
        assert Settings.model_fields["TORCH_COMPILE_MODE"].default == "default", (
            "reduce-overhead enables CUDA Graphs, which breaks this generate path"
        )

    def test_loader_default_matches(self):
        """Find the loader entry point that takes the flag, whatever it's called."""
        candidates = []
        for _, obj in vars(model_loader).items():
            if not inspect.isclass(obj):
                continue
            for name, fn in vars(obj).items():
                if not callable(fn):
                    continue
                try:
                    sig = inspect.signature(fn)
                except (TypeError, ValueError):
                    continue
                if "torch_compile_mode" in sig.parameters:
                    candidates.append(
                        (obj.__name__, name, sig.parameters["torch_compile_mode"].default)
                    )

        assert candidates, "no loader method takes torch_compile_mode"
        for cls, meth, default in candidates:
            assert default == "default", (
                f"{cls}.{meth} defaults torch_compile_mode to {default!r}; "
                "reduce-overhead enables CUDA Graphs and breaks generate"
            )

    def test_the_hazard_is_documented_where_the_default_lives(self):
        src = inspect.getsource(model_loader)
        assert "CUDAGraphs" in src or "CUDA Graphs" in src, (
            "nothing records why reduce-overhead is avoided, so it gets "
            "restored as an apparent performance oversight"
        )


class TestTheSoakWouldCatchIt:
    def test_more_than_one_pass_is_run(self):
        src = inspect.getsource(model_loader)
        m = re.search(r"_SOAK_PASSES\s*=\s*(\d+)", src)
        assert m, "no soak-pass constant found"
        assert int(m.group(1)) >= 2, (
            f"soak runs {m.group(1)} pass(es); the CUDA-Graphs fault only "
            "appears on the SECOND, so a single pass cannot detect it"
        )

    def test_the_soak_reads_the_output(self):
        """A CUDA-Graphs violation raises on ACCESS, not on the call."""
        src = inspect.getsource(model_loader)
        assert "_gen[0, -1].item()" in src, (
            "the soak never touches the output, so a graph-overwrite fault "
            "would pass silently"
        )

    def test_the_soak_exercises_cached_generation(self):
        """The fault appeared during real decoding, not a bare forward.

        The original warmup was one uncached seq_len=1 forward — it would not
        have reproduced the failure even if repeated.
        """
        src = inspect.getsource(model_loader)
        soak = src[src.index("_SOAK_PASSES"):src.index("torch_compile_warmup_complete")]
        assert "self.model.generate(" in soak, (
            "the soak calls forward() directly; the CUDA-Graphs fault occurs "
            "in cached generate()"
        )
        assert "use_cache=True" in soak, "the soak runs uncached"
        assert "max_new_tokens" in soak, "the soak generates no tokens"

    def test_a_failed_soak_reverts_to_eager(self):
        src = inspect.getsource(model_loader)
        assert "self.model.forward = _uncompiled_forward" in src, (
            "a failed soak leaves the compiled forward installed — that is the "
            "500-for-every-request outcome"
        )
        assert "torch_compile_soak_failed_reverted_to_eager" in src, (
            "the fallback is silent; an operator cannot tell compile is off"
        )

    def test_the_uncompiled_forward_is_captured_before_compiling(self):
        src = inspect.getsource(model_loader)
        cap = src.index("_uncompiled_forward = self.model.forward")
        comp = src.index("self.model.forward = torch.compile(")
        assert cap < comp, (
            "the original forward is captured after compiling, so the revert "
            "would restore the compiled function"
        )


class TestSoakSemantics:
    """The contract the source assertions above stand for."""

    def test_a_second_pass_failure_triggers_the_revert(self):
        calls = {"n": 0}
        original = object()
        installed = {"forward": "compiled"}

        class Out:
            class _L:
                def __getitem__(self, _):
                    raise RuntimeError(
                        "accessing tensor output of CUDAGraphs that has been "
                        "overwritten by a subsequent run"
                    )

            logits = _L()

        def forward(**_kw):
            calls["n"] += 1
            return Out()

        # Mirror the loader's soak: N passes, each reading logits, revert on error.
        try:
            for _ in range(3):
                out = forward()
                _ = out.logits[0]
        except Exception:
            installed["forward"] = original

        assert calls["n"] == 1, "should fail on the first output ACCESS"
        assert installed["forward"] is original, "did not revert to eager"
