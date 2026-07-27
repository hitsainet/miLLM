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

    def test_torch_compile_is_enabled(self):
        env = _backend_env()
        assert env.get("TORCH_COMPILE") == "true"

    def test_torch_compile_is_documented_as_quantization_dependent(self):
        """It is silently skipped under bitsandbytes — a reader who does not
        know that will conclude it is working when it is not."""
        text = MANIFEST.read_text()
        i = text.index("TORCH_COMPILE")
        window = text[max(0, i - 500):i + 200].lower()
        assert "bitsandbytes" in window or "quantiz" in window, (
            "TORCH_COMPILE carries no note that it is a no-op under "
            "bitsandbytes quantization"
        )
