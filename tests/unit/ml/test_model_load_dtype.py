"""miLLM loads a row at the checkpoint's own precision, and reports THAT precision.

This server loaded every non-GGUF row in bfloat16 and reported `model.dtype`. The rule it now shares
with miStudio (`millm/ml/native_dtype.py`, `docs/schemas/native-dtype-cases.json`) loads a
bfloat16 checkpoint at bfloat16 — unchanged — a float16 checkpoint at float16 and an FP32 row at
float32, and the reported dtype is the resolved one, which probe identity compares against a
definition's `model.load_dtype`. These assert what `from_pretrained` actually receives.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from millm.ml import model_loader
from millm.ml.gpu_placement import choose_gpu
from tests.support.fake_gpus import fake_gpus
from tests.unit.ml.test_model_load_placement import NODE, FakeModel, _load, _reset_state  # noqa: F401

CUDA1 = torch.device("cuda", 1)


@pytest.fixture(autouse=True)
def _use_the_fake_factory(monkeypatch):
    """With a config present the load picks a REAL class for it, which then reads the fixture's
    nonexistent path; the class choice is not what these tests are about."""
    def _no_class(config):
        raise LookupError("class choice not under test")

    monkeypatch.setattr(model_loader, "_get_auto_model_class", _no_class)


def _config(dtype):
    return SimpleNamespace(model_type="llama", torch_dtype=dtype)


@pytest.mark.parametrize("recorded,expected", [("bfloat16", torch.bfloat16), ("float16", torch.float16)])
def test_a_16_bit_row_loads_at_the_checkpoints_dtype_and_reports_it(recorded, expected):
    with fake_gpus(*NODE) as fake:
        loaded, kwargs = _load(fake, choose_gpu(8_000), config=_config(recorded))
    assert kwargs["torch_dtype"] is expected
    assert loaded.dtype == recorded


def test_an_fp32_row_really_loads_float32():
    with fake_gpus(*NODE) as fake:
        loaded, kwargs = _load(fake, choose_gpu(8_000), quantization="FP32", config=_config("bfloat16"))
    assert kwargs["torch_dtype"] is torch.float32
    assert loaded.dtype == "float32"


def test_the_reported_dtype_is_the_resolved_one_not_model_dtype():
    """⚠ On a bitsandbytes model `model.dtype` is whichever parameter transformers checks first."""
    model = FakeModel([CUDA1])
    model.dtype = torch.float32          # what a bnb model can report
    with fake_gpus(*NODE) as fake:
        loaded, _ = _load(fake, choose_gpu(8_000), model=model, config=_config("bfloat16"))
    assert loaded.dtype == "bfloat16"


def test_a_q4_row_computes_at_the_checkpoints_dtype():
    seen = {}

    def bnb(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(**kwargs)

    with fake_gpus(*NODE) as fake:
        _load(fake, choose_gpu(8_000), quantization="Q4", bnb=bnb, config=_config("float16"))
    assert seen["bnb_4bit_compute_dtype"] is torch.float16


def test_no_readable_config_loads_bfloat16():
    """The rule's default when nothing is recorded — the old behaviour, now stated."""
    with fake_gpus(*NODE) as fake:
        loaded, kwargs = _load(fake, choose_gpu(8_000))
    assert kwargs["torch_dtype"] is torch.bfloat16 and loaded.dtype == "bfloat16"


def test_kv_bytes_follow_the_load_dtype():
    assert model_loader.kv_bytes_for("float32") == 4
    assert model_loader.kv_bytes_for("bfloat16") == model_loader.kv_bytes_for("float16") == 2


def test_the_planning_paths_take_the_resolved_dtype():
    """The split preflight, the per-card fit and the meta model must plan at the dtype the load
    uses; asserted by AST so a reintroduced `torch.bfloat16` literal on any of them goes red."""
    import ast
    import inspect

    for fn in (model_loader.preflight_split, model_loader.transformers_fit,
               model_loader.checkpoint_materialised_mb):
        tree = ast.parse(inspect.getsource(fn))
        literals = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                    and n.attr in ("bfloat16", "float16") and ast.unparse(n.value) == "torch"]
        assert literals == [], f"{fn.__name__} plans at a hardcoded precision"
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "resolve_for_config"]
        assert calls, f"{fn.__name__} never resolves the load dtype"
