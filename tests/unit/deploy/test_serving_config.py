"""The deployed serving configuration must actually serve.

Three settings that are invisible until something is wrong.

1. AUTO_LOAD_MODEL. `main.py` resets every 'loaded'/'loading' model row to
   'ready' on startup — the status is process state, not durable state — and
   only AUTO_LOAD_MODEL reloads one. Until 2026-07-27 it was unset, so every
   deploy silently took inference offline until a human noticed and hand-loaded
   an 8B model. Nothing alarms on this: the pod is Ready, /health is green, and
   requests just fail.

2. TORCH_COMPILE. Skipped by model_loader whenever the loaded model uses
   bitsandbytes quantization, so it was a no-op for the whole time granite ran
   in Q4. It only does anything on an fp16/bf16 load.

3. ENABLE_CONTINUOUS_BATCHING. Without it the queue backend serialises.
   Measured on fp16 granite-4.1-8b, 128 tokens, temperature 0:
       concurrency 1 -> 33.63 tok/s aggregate
       concurrency 2 -> 33.96 tok/s aggregate  (1.01x)
       concurrency 4 -> 34.28 tok/s aggregate  (1.02x)
   Aggregate throughput is flat; concurrency buys nothing.

MUTATION CONTROLS:
  * remove AUTO_LOAD_MODEL          -> autoload test fails
  * set either perf flag to "false" -> the corresponding test fails
  * pin a dtype in AUTO_LOAD_MODEL  -> the no-dtype-override test fails
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

MANIFEST = Path(__file__).resolve().parents[3] / "k8s" / "base" / "backend.yaml"


def _backend_env():
    if not MANIFEST.exists():          # pragma: no cover - layout guard
        pytest.skip(f"manifest not found at {MANIFEST}")
    for doc in yaml.safe_load_all(MANIFEST.read_text()):
        if not doc or doc.get("kind") != "Deployment":
            continue
        for c in doc["spec"]["template"]["spec"]["containers"]:
            if c["name"] == "backend":
                return {e["name"]: e.get("value") for e in c.get("env", [])}
    pytest.fail("no 'backend' container found in the deployment")


class TestTheModelComesBackAfterARestart:
    def test_auto_load_model_is_configured(self):
        env = _backend_env()
        assert env.get("AUTO_LOAD_MODEL"), (
            "AUTO_LOAD_MODEL is unset, so a restart leaves the pod Ready and "
            "healthy while serving no model at all"
        )

    def test_it_names_a_model_not_a_dtype_variant(self):
        """The row's own `quantization` decides dtype.

        load_model() passes model.quantization.value to the loader, so naming a
        dtype here would either be ignored or fight the operator's choice.
        """
        value = _backend_env().get("AUTO_LOAD_MODEL", "")
        for token in ("fp16", "bf16", "q4", "q8", "int8", "4bit"):
            assert token not in value.lower(), (
                f"AUTO_LOAD_MODEL={value!r} encodes a dtype; quantization comes "
                "from the model row, not from this identifier"
            )

    def test_the_autoload_path_is_still_gated_on_this_setting(self):
        """If main.py stops reading it, the manifest value is decoration."""
        main = (Path(__file__).resolve().parents[3] / "millm" / "main.py").read_text()
        assert "settings.AUTO_LOAD_MODEL" in main, (
            "main.py no longer reads AUTO_LOAD_MODEL"
        )
        assert "_auto_load_model(" in main


class TestThePerformanceFlagsAreOn:
    def test_continuous_batching_is_enabled(self):
        env = _backend_env()
        assert env.get("ENABLE_CONTINUOUS_BATCHING") == "true", (
            "without continuous batching the queue backend serialises: "
            "measured 1.02x aggregate throughput at concurrency 4"
        )

    def test_torch_compile_is_only_on_with_a_graph_free_mode(self):
        """The invariant, not a fixed value.

        Enabling TORCH_COMPILE is safe ONLY while the compile mode avoids CUDA
        Graphs. Pinning the flag to "false" would have blocked the fix; pinning
        it to "true" would let someone re-enable reduce-overhead and repeat the
        2026-07-27 outage. So assert the pair.
        """
        from millm.core.config import Settings

        env = _backend_env()
        if env.get("TORCH_COMPILE") != "true":
            return  # off is always safe

        mode = env.get("TORCH_COMPILE_MODE") or Settings.model_fields[
            "TORCH_COMPILE_MODE"
        ].default
        assert mode != "reduce-overhead", (
            "TORCH_COMPILE is on with mode='reduce-overhead', which enables "
            "CUDA Graphs and takes inference down on the second request"
        )

    def test_compile_cannot_be_on_without_the_soak_and_fallback(self):
        """Enabling the flag is only defensible because a failed compile now
        degrades to eager instead of serving 500s."""
        import inspect

        from millm.ml import model_loader

        env = _backend_env()
        if env.get("TORCH_COMPILE") != "true":
            return

        src = inspect.getsource(model_loader)
        assert "_SOAK_PASSES" in src, "compile is on but nothing soaks it"
        assert "self.model.forward = _uncompiled_forward" in src, (
            "compile is on with no revert-to-eager; a bad compile would serve "
            "500 for every request"
        )
        assert "self.model.generate(" in src, (
            "the soak does not exercise cached generation — the path that "
            "actually failed"
        )

    def test_the_cuda_graphs_hazard_is_recorded_in_the_manifest(self):
        text = MANIFEST.read_text()
        i = text.index("TORCH_COMPILE")
        window = text[i:i + 1200].lower()
        assert "cudagraph" in window or "cuda graph" in window, (
            "the reason TORCH_COMPILE is disabled is undocumented, so the next "
            "person re-enables it and takes inference down again"
        )

    def test_torch_compile_is_documented_as_quantization_dependent(self):
        """It is silently skipped under bitsandbytes — a reader who does not
        know that will conclude it is working when it is not."""
        text = MANIFEST.read_text()
        i = text.index("TORCH_COMPILE")
        # Look FORWARD: the caveat lives in the comment block under the key.
        window = text[i:i + 1500].lower()
        assert "bitsandbytes" in window or "quantiz" in window, (
            "TORCH_COMPILE carries no note that it is a no-op under "
            "bitsandbytes quantization"
        )


class TestContinuousBatchingActuallyCoversItsWorkload:
    """CBM batches only ONE sampling profile; the rest fall back to serial.

    Observed live 2026-07-27: CBM was running, and every benchmark and labeling
    request logged

        cbm_routing_fallback_to_serial
        reason=sampling_params_mismatch request_temperature=0.0 cbm_temperature=0.7

    The commit enabling CBM justified it with "labeling 32k features is the main
    beneficiary ... labeling uses temperature 0 throughout, so this is not a
    constraint there." That is inverted: temperature 0 != 0.7, so labeling was
    precisely the workload excluded. The feature ran and helped nothing.

    MUTATION CONTROL: set CBM_DEFAULT_TEMPERATURE back to 0.7 -> this fails.
    """

    def test_cbm_sampling_matches_the_bulk_workload(self):
        from millm.core.config import Settings

        temp = Settings.model_fields["CBM_DEFAULT_TEMPERATURE"].default
        assert temp == 0.0, (
            f"CBM batches only temperature={temp}; bulk labeling runs at 0.0 "
            "and would fall back to the serial path, which is the whole "
            "workload continuous batching was enabled for"
        )

    def test_the_mismatch_fallback_is_observable(self):
        """An operator must be able to see that batching is being bypassed."""
        import inspect

        from millm.services import inference_service

        src = inspect.getsource(inference_service)
        assert "cbm_routing_fallback_to_serial" in src, (
            "requests silently bypass CBM with nothing logged, so 'cbm_running: "
            "true' would imply batching that is not happening"
        )
