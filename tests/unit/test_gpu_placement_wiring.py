"""Placement is WIRED, not merely declared: each load path calls the resolver
and passes the request through.

This repo has shipped capabilities that were implemented, unit-tested and never
called (see CLAUDE.md, "A capability is not shipped until a test FAILS when its
wiring is removed"). The behavioural tests in tests/unit/ml/test_model_load_placement.py
and test_gguf_placement.py drive the real paths; these assert the CALL SITES
themselves, on the AST — the call and its argument, not a name in the text,
which a comment describing the mechanism would satisfy.

MUTATION CONTROLS (each must turn this file red):
  * ModelLoader.load: choose_gpu(estimated_memory_mb) without requested=gpu
  * load_gguf_model: drop **_gguf_placement_kwargs(placement) from the kwargs
  * ModelLoadContext.load: device_map back to "auto"
  * LoadedModelState.clear: _release_cuda_memory([]) instead of gpu_indices
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from millm.api.routes.management import models as models_route
from millm.ml import model_loader
from millm.services import inference_service, model_service


def _function(module, qualname: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(module)))
    parts = qualname.split(".")
    scope: list[ast.stmt] = tree.body
    node = None
    for part in parts:
        node = next(
            (
                n
                for n in scope
                if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == part
            ),
            None,
        )
        assert node is not None, f"{module.__name__}.{qualname} not found"
        scope = node.body
    return node


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    found = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if (isinstance(func, ast.Name) and func.id == name) or (
                isinstance(func, ast.Attribute) and func.attr == name
            ):
                found.append(sub)
    return found


def _kw(call: ast.Call, name: str) -> ast.AST | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def _is_name(node: ast.AST | None, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_attr(node: ast.AST | None, owner: str, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == owner
    )


class TestTransformersLoad:
    def test_model_loader_resolves_with_the_request(self):
        fn = _function(model_loader, "ModelLoader.load")
        [call] = _calls(fn, "choose_gpu")
        assert _is_name(call.args[0], "estimated_memory_mb")
        assert _is_name(_kw(call, "requested"), "gpu")

    def test_the_decision_is_handed_to_the_context(self):
        fn = _function(model_loader, "ModelLoader.load")
        loads = [c for c in _calls(fn, "load") if _kw(c, "placement") is not None]
        assert len(loads) == 1 and _is_name(_kw(loads[0], "placement"), "placement")

    def test_the_context_takes_its_device_map_from_the_placement(self):
        fn = _function(model_loader, "ModelLoadContext.load")
        maps = [
            value
            for d in ast.walk(fn)
            if isinstance(d, ast.Dict)
            for key, value in zip(d.keys, d.values)
            if isinstance(key, ast.Constant) and key.value == "device_map"
        ]
        assert len(maps) == 1
        call = maps[0]
        assert isinstance(call, ast.Call) and _is_attr(call.func, "placement", "transformers_device_map")

    def test_bitsandbytes_max_memory_comes_from_the_placement(self):
        fn = _function(model_loader, "ModelLoadContext.load")
        [call] = _calls(fn, "bitsandbytes_max_memory")
        assert _is_attr(call.func, "placement", "bitsandbytes_max_memory")


class TestGgufLoad:
    def test_the_entry_point_passes_the_request(self):
        fn = _function(model_loader, "ModelLoader.load")
        [call] = _calls(fn, "load_gguf_model")
        assert _is_name(_kw(call, "gpu"), "gpu")

    def test_the_gguf_loader_plans_with_the_request(self):
        fn = _function(model_loader, "load_gguf_model")
        [call] = _calls(fn, "plan_gguf_placement")
        assert _is_name(_kw(call, "requested"), "gpu")

    def test_the_plan_is_decided_by_choose_gpu(self):
        fn = _function(model_loader, "plan_gguf_placement")
        calls = _calls(fn, "choose_gpu")
        assert len(calls) >= 2
        assert any(_is_name(_kw(c, "requested"), "wanted") for c in calls)

    def test_llama_kwargs_carry_the_placement(self):
        fn = _function(model_loader, "load_gguf_model")
        spreads = [
            value
            for d in ast.walk(fn)
            if isinstance(d, ast.Dict)
            for key, value in zip(d.keys, d.values)
            if key is None  # a ** entry
        ]
        placement_spreads = [
            v for v in spreads
            if isinstance(v, ast.Call) and _is_name(v.func, "_gguf_placement_kwargs")
            and _is_name(v.args[0], "placement")
        ]
        assert len(placement_spreads) == 1

    def test_the_context_prediction_is_budgeted_on_the_placement(self):
        fn = _function(model_loader, "load_gguf_model")
        assigns = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and any(_is_name(t, "free_mb") for t in n.targets)
        ]
        assert [a.value for a in assigns if _is_attr(a.value, "placement", "capacity_mb")]
        [predict] = _calls(fn, "predicted_max_context")
        assert _is_name(predict.args[2], "free_mb")


class TestCleanup:
    def test_unload_releases_the_models_cards(self):
        fn = _function(model_loader, "LoadedModelState.clear")
        [call] = _calls(fn, "_release_cuda_memory")
        assert _is_name(call.args[0], "gpu_indices")

    def test_a_failed_load_releases_the_placement_cards(self):
        fn = _function(model_loader, "ModelLoadContext.__exit__")
        [call] = _calls(fn, "_release_cuda_memory")
        assert _is_attr(call.args[0], "self", "gpu_indices")


class TestServiceAndRoute:
    def test_route_passes_gpu(self):
        fn = _function(models_route, "load_model")
        [call] = _calls(fn, "load_model")
        assert _kw(call, "gpu") is not None

    def test_service_validates_and_forwards_the_card(self):
        fn = _function(model_service, "ModelService.load_model")
        assert _calls(fn, "find_gpu"), "a named card is no longer checked before the load starts"
        [call] = _calls(fn, "run_in_executor")
        assert _is_name(call.args[-1], "wanted")

    def test_worker_forwards_the_card(self):
        fn = _function(model_service, "ModelService._load_worker")
        [call] = [c for c in _calls(fn, "load") if _kw(c, "gguf_file") is not None]
        assert _is_name(_kw(call, "gpu"), "gpu")


class TestInference:
    def test_draft_model_on_the_input_device(self):
        fn = _function(inference_service, "InferenceService._get_draft_model")
        [call] = _calls(fn, "from_pretrained")
        device_map = _kw(call, "device_map")
        assert isinstance(device_map, ast.Dict)
        [value] = device_map.values
        assert isinstance(value, ast.Call) and value.func.attr == "_get_input_device"

    def test_kv_sizing_asks_the_models_cards(self):
        fn = _function(inference_service, "InferenceService._chunk_batch_for_memory")
        [call] = _calls(fn, "_kv_fits")
        assert _is_name(call.args[1], "gpu_indices")
        fits = _function(inference_service, "InferenceService._kv_fits")
        [verify] = _calls(fits, "verify_memory_available")
        assert _is_name(_kw(verify, "device"), "index")


@pytest.mark.parametrize(
    "module, qualname",
    [
        (model_loader, "ModelLoader.load"),
        (model_loader, "load_gguf_model"),
        (model_loader, "ModelLoadContext.load"),
    ],
)
def test_no_load_path_reads_memory_utils_gpu0_helpers(module, qualname):
    """The Phase 0 helpers answered for GPU 0 or for all cards regardless of
    where the model goes; the load paths must decide through placement."""
    fn = _function(module, qualname)
    for name in ("get_available_memory_mb", "get_total_free_memory_mb", "get_largest_free_memory_mb"):
        assert not _calls(fn, name), f"{qualname} calls {name}"
