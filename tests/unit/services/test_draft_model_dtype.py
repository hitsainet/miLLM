"""The speculative-decoding draft loads at the TARGET's precision, not a hardcoded bfloat16."""

from types import SimpleNamespace

import torch

from millm.ml.model_loader import LoadedModelState
from millm.services import inference_service


def test_the_draft_follows_the_loaded_models_precision(monkeypatch):
    state = LoadedModelState()
    for name, expected in (("float16", torch.float16), ("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        monkeypatch.setattr(state, "_loaded", SimpleNamespace(dtype=name), raising=False)
        monkeypatch.setattr(LoadedModelState, "current", property(lambda self: self._loaded), raising=False)
        assert inference_service._target_torch_dtype() is expected


def test_with_nothing_loaded_it_is_bfloat16(monkeypatch):
    monkeypatch.setattr(LoadedModelState, "current", property(lambda self: None), raising=False)
    assert inference_service._target_torch_dtype() is torch.bfloat16


def test_the_draft_load_uses_it():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(inference_service))
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "from_pretrained"
             and any(k.arg == "torch_dtype" for k in n.keywords)]
    assert loads and all(ast.unparse(next(k.value for k in c.keywords if k.arg == "torch_dtype")) == "_target_torch_dtype()"
                         for c in loads)
