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
Phase 2, 2026-09-14 (mutate.py; restored and sha256-verified):
  M9b on_model_unloading no longer calls _release_draft_model
      -> test_the_draft_is_released_on_every_model_change
  M13b _chunk_batch_for_memory passes no layer shares to _kv_fits
      -> test_kv_sizing_asks_the_models_cards
  M23 the compile warm-up keeps its own input-device lookup
      -> test_the_compile_warm_up_asks_the_shared_input_device
Review round 2, 2026-09-14: three call sites moved (plan_transformers_load, the
preflight, _draft_device). Each retargeted test pins BOTH links of the new chain,
not only the call that moved (mutate.py; restored and sha256-verified):
  R2-M2c plan_transformers_load does not pass the factor
      -> test_model_loader_resolves_with_the_request_against_live_memory
  R2-M4  ModelLoader.load skips the preflight   -> test_the_split_preflight_runs_before_the_load
  R2-M5  the pre-check skips the preflight       -> test_the_precheck_runs_the_split_preflight_on_its_plan
  R1-M3c re-run (pre-check plans without the checkpoint)
      -> test_the_precheck_projects_the_resident_models_memory_back
Review round 3, 2026-09-14: a pre-quantized checkpoint is sized by building its
model with its quantizer, which needs the row's trust_remote_code at both call
sites (mutate.py; restored and sha256-verified):
  R3-M4  the estimate uses the stored weights only
      -> test_a_pre_quantized_checkpoint_is_sized_as_transformers_will_load_it
  R3-M4d plan_transformers_load drops trust_remote_code -> test_model_loader_resolves_...
  R3-M4e ModelLoader.load drops it                     -> test_model_loader_resolves_...
  R3-M4f the pre-check drops it                        -> test_the_precheck_projects_...
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
    def test_model_loader_resolves_with_the_request_against_live_memory(self):
        """The authoritative check, after the unload: live inventory, not a projection.

        Review round 2 moved the decision behind plan_transformers_load, which
        reads the checkpoint. BOTH links are pinned: retargeting only the first
        would leave a hole the size of the refactor."""
        fn = _function(model_loader, "ModelLoader.load")
        [call] = _calls(fn, "plan_transformers_load")
        assert _is_name(call.args[0], "estimated_memory_mb")
        assert _is_name(_kw(call, "requested"), "gpu")
        gpus = _kw(call, "gpus")
        assert isinstance(gpus, ast.Call) and _is_name(gpus.func, "list_gpus")
        assert _is_name(_kw(call, "cache_path"), "cache_path")
        assert _is_name(_kw(call, "trust_remote_code"), "trust_remote_code")

        plan = _function(model_loader, "plan_transformers_load")
        [decide] = _calls(plan, "decide_transformers_placement")
        assert _is_name(_kw(decide, "requested"), "requested")
        assert _is_name(_kw(decide, "gpus"), "gpus")
        estimate = decide.args[0]
        assert isinstance(estimate, ast.Call) and _is_name(estimate.func, "transformers_estimate_mb")
        assert _is_name(_kw(estimate, "trust_remote_code"), "trust_remote_code")
        factor = _kw(decide, "pre_quantized_max_memory_factor")
        assert isinstance(factor, ast.Call) and _is_name(factor.func, "pre_quantized_max_memory_factor")

    def test_a_transformers_plan_is_judged_per_card_before_the_slack(self):
        """Decision 7, 2026-09-14: plan_transformers_load sizes the checkpoint as it
        loads (transformers_fit) and hands it to the per-card decision with the
        split's quantizer factor; the slack decides only when the fit is None."""
        plan = _function(model_loader, "plan_transformers_load")
        [fit] = _calls(plan, "transformers_fit")
        assert _is_name(fit.args[0], "cache_path") and _is_name(fit.args[1], "quantization")
        assert _is_name(fit.args[2], "pre_quantized")
        assert _is_name(_kw(fit, "trust_remote_code"), "trust_remote_code")
        [decide] = _calls(plan, "decide_transformers_fit")
        assert _is_name(decide.args[0], "fit")
        assert _is_name(_kw(decide, "gpus"), "gpus")
        assert _is_name(_kw(decide, "requested"), "requested")
        factor = _kw(decide, "max_memory_factor")
        assert isinstance(factor, ast.Call) and _is_name(factor.func, "split_max_memory_factor")
        [refusal] = _calls(plan, "refuse_unsupported_quantization")
        assert _is_name(refusal.args[1], "pre_quantized")

        per_card = _function(model_loader, "decide_transformers_fit")
        assert len(_calls(per_card, "_check_split_fit")) == 2, "\"all\" and Auto's split"
        [left_out] = _calls(per_card, "refuse_cards_left_out_of_all")
        assert _is_name(left_out.args[1], "inventory")

        slack = _function(model_loader, "decide_transformers_placement")
        [q2] = _calls(slack, "refuse_unsupported_quantization")
        assert _is_name(q2.args[1], "is_pre_quantized")
        [factor] = _calls(slack, "split_max_memory_factor")
        assert _is_name(factor.args[2], "pre_quantized_max_memory_factor")

    def test_a_pre_quantized_checkpoint_is_sized_as_transformers_will_load_it(self):
        """Review round 3: what a checkpoint stores is a floor; the estimate also
        asks what transformers materialises (it dequantizes FP8 on these cards)."""
        fn = _function(model_loader, "transformers_estimate_mb")
        [materialised] = _calls(fn, "checkpoint_materialised_mb")
        assert _is_name(materialised.args[0], "cache_path")
        assert _is_name(materialised.args[1], "trust_remote_code")
        [stored] = _calls(fn, "checkpoint_weights_mb")
        [larger] = _calls(fn, "max")
        assert materialised in larger.args and stored in larger.args

    def test_the_split_preflight_runs_before_the_load(self):
        """preflight_split computes the real device map before any weight is
        read; it must run on the placement the load uses, and before it."""
        fn = _function(model_loader, "ModelLoader.load")
        [preflight] = _calls(fn, "preflight_split")
        assert [_is_name(arg, name) for arg, name in zip(
            preflight.args, ("model_name", "cache_path", "quantization", "placement", "trust_remote_code")
        )] == [True] * 5
        [load] = [c for c in _calls(fn, "load") if _kw(c, "placement") is not None]
        assert preflight.lineno < load.lineno

    def test_the_shared_decision_is_choose_gpu(self):
        fn = _function(model_loader, "decide_transformers_placement")
        [call] = _calls(fn, "choose_gpu")
        assert _is_name(_kw(call, "requested"), "requested")
        assert _is_name(_kw(call, "gpus"), "gpus")
        shard = _kw(call, "shard")
        assert isinstance(shard, ast.Call) and _is_name(shard.func, "transformers_shard_rule")
        assert _is_name(_kw(shard, "max_memory_factor"), "max_memory_factor")

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

    def test_max_memory_comes_from_the_placement_for_every_quantization(self):
        fn = _function(model_loader, "ModelLoadContext.load")
        [call] = _calls(fn, "transformers_max_memory")
        assert _is_attr(call.func, "placement", "transformers_max_memory")
        assert not _calls(fn, "get_available_cpu_memory_mb"), "a CPU budget is back in the load"

    def test_the_load_refuses_what_landed_off_the_gpu(self):
        """Both refusals raise from the load itself: bitsandbytes' own, and the
        check of where the weights actually landed."""
        fn = _function(model_loader, "ModelLoadContext.load")
        raises = [
            node for node in ast.walk(fn)
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
            and _is_name(node.exc.func, "_off_gpu_refusal")
        ]
        assert len(raises) == 2

    def test_the_compile_warm_up_asks_the_shared_input_device(self):
        fn = _function(model_loader, "ModelLoadContext.load")
        [call] = _calls(fn, "model_input_device")
        assert _is_attr(call.args[0], "self", "model")


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
        for call in calls:
            shard = _kw(call, "shard")
            assert isinstance(shard, ast.Call) and _is_name(shard.func, "_gguf_shard_rule"), (
                "a GGUF split must be sized with llama.cpp's per-card overhead"
            )

    def test_llama_kwargs_carry_the_placement(self):
        fn = _function(model_loader, "load_gguf_model")
        [call] = _calls(fn, "_gguf_placement_kwargs")
        assert _is_name(call.args[0], "placement")
        parse = call.args[1]
        assert isinstance(parse, ast.Call) and _is_name(parse.func, "parse_gguf_tensor_split")
        assert _is_attr(parse.args[0], "_settings", "GGUF_TENSOR_SPLIT")
        [assign] = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and n.value is call]
        assert _is_name(assign.targets[0], "placement_kwargs")
        spreads = [
            value
            for d in ast.walk(fn)
            if isinstance(d, ast.Dict)
            for key, value in zip(d.keys, d.values)
            if key is None  # a ** entry
        ]
        assert len([v for v in spreads if _is_name(v, "placement_kwargs")]) == 1

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
        n_cards = _kw(predict, "n_cards")
        assert isinstance(n_cards, ast.Call) and _is_name(n_cards.func, "len")
        assert _is_attr(n_cards.args[0], "placement", "gpu_indices")


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
        [precheck] = _calls(fn, "_precheck_placement")
        assert _is_name(precheck.args[1], "wanted")
        [call] = _calls(fn, "run_in_executor")
        assert _is_name(call.args[-1], "wanted")

    def test_the_precheck_runs_before_the_resident_model_is_unloaded(self):
        fn = _function(model_service, "ModelService.load_model")
        [precheck] = _calls(fn, "_precheck_placement")
        [unload] = _calls(fn, "unload_model")
        assert precheck.lineno < unload.lineno

    def test_the_precheck_projects_the_resident_models_memory_back(self):
        fn = _function(model_service, "ModelService._precheck_placement")
        [project] = _calls(fn, "project_free_after_unload")
        assert isinstance(project.args[0], ast.Call) and _is_name(project.args[0].func, "list_gpus")
        [plan] = _calls(fn, "plan_transformers_load")
        assert _is_name(_kw(plan, "gpus"), "gpus")
        assert _is_name(_kw(plan, "cache_path"), "cache_path")
        # Review round 3: sizing a pre-quantized checkpoint builds its model, and a
        # remote-code architecture only builds with the row's trust_remote_code.
        trust = _kw(plan, "trust_remote_code")
        assert isinstance(trust, ast.Call) and _is_name(trust.func, "bool")
        [gguf] = _calls(fn, "plan_gguf_placement")
        assert _is_name(_kw(gguf, "gpus"), "gpus")

    def test_the_precheck_runs_the_split_preflight_on_its_plan(self):
        """Review round 2: the pre-check computes the split's real device map
        before the unload, on the placement it just planned."""
        fn = _function(model_service, "ModelService._precheck_placement")
        [plan] = _calls(fn, "plan_transformers_load")
        [assign] = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and n.value is plan]
        assert _is_name(assign.targets[0], "placement")
        [preflight] = _calls(fn, "preflight_split")
        assert _is_name(preflight.args[1], "cache_path")
        assert _is_name(preflight.args[3], "placement")
        assert plan.lineno < preflight.lineno

    def test_worker_forwards_the_card(self):
        fn = _function(model_service, "ModelService._load_worker")
        [call] = [c for c in _calls(fn, "load") if _kw(c, "gguf_file") is not None]
        assert _is_name(_kw(call, "gpu"), "gpu")


class TestInference:
    def test_draft_model_on_the_card_draft_device_chooses(self):
        """Review round 2 put a split model's draft on its most-free card
        (`_draft_device`), not the input device. Both links pinned: the load
        uses the chosen device, and the choice reads the model's own cards and
        falls back to the input device."""
        fn = _function(inference_service, "InferenceService._get_draft_model")
        [call] = _calls(fn, "from_pretrained")
        device_map = _kw(call, "device_map")
        assert isinstance(device_map, ast.Dict)
        [value] = device_map.values
        assert _is_name(value, "device")
        [chosen] = _calls(fn, "_draft_device")
        [assign] = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and n.value is chosen]
        assert _is_name(assign.targets[0], "device")

        choose = _function(inference_service, "InferenceService._draft_device")
        [free] = _calls(choose, "free_mb_by_index")
        assert _is_name(free.args[0], "indices")
        assert len(_calls(choose, "_get_input_device")) == 1

    def test_kv_sizing_asks_the_models_cards(self):
        fn = _function(inference_service, "InferenceService._chunk_batch_for_memory")
        [call] = _calls(fn, "_kv_fits")
        assert _is_name(call.args[1], "gpu_indices")
        assert _is_name(call.args[2], "shares")
        [shares] = _calls(fn, "layer_share_by_index")
        assert _is_attr(shares.args[0], "self", "_model")
        assert _is_name(shares.args[1], "gpu_indices")
        fits = _function(inference_service, "InferenceService._kv_fits")
        [verify] = _calls(fits, "verify_memory_available")
        assert _is_name(_kw(verify, "device"), "index")

    def test_the_draft_is_released_on_every_model_change(self):
        for hook in ("on_model_loaded", "on_model_unloading"):
            fn = _function(inference_service, f"InferenceService.{hook}")
            assert len(_calls(fn, "_release_draft_model")) == 1, hook
        # ...and the service calls both hooks.
        assert _calls(_function(model_service, "ModelService._load_worker"), "on_model_loaded")
        assert _calls(_function(model_service, "ModelService.unload_model"), "on_model_unloading")

    def test_inputs_follow_the_shared_input_device(self):
        fn = _function(inference_service, "InferenceService._get_input_device")
        assert len(_calls(fn, "model_input_device")) == 1


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
