"""The speculative-decoding draft loads at ITS OWN checkpoint's precision, by the shared rule."""

from types import SimpleNamespace

import torch

from millm.services import inference_service


def _with_config(monkeypatch, config):
    import transformers

    def from_pretrained(*a, **k):
        if isinstance(config, Exception):
            raise config
        return config

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)


def test_a_float16_draft_loads_float16(monkeypatch):
    _with_config(monkeypatch, SimpleNamespace(torch_dtype="float16"))
    assert inference_service._draft_torch_dtype("org/draft") is torch.float16


def test_a_bfloat16_draft_loads_bfloat16(monkeypatch):
    _with_config(monkeypatch, SimpleNamespace(torch_dtype="bfloat16"))
    assert inference_service._draft_torch_dtype("org/draft") is torch.bfloat16


def test_an_unreadable_config_is_the_rules_default(monkeypatch):
    _with_config(monkeypatch, OSError("offline"))
    assert inference_service._draft_torch_dtype("org/draft") is torch.bfloat16


def test_the_draft_load_uses_it():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(inference_service))
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "from_pretrained"
             and any(k.arg == "torch_dtype" for k in n.keywords)]
    assert loads and all(
        ast.unparse(next(k.value for k in c.keywords if k.arg == "torch_dtype")).startswith("_draft_torch_dtype(")
        for c in loads)
