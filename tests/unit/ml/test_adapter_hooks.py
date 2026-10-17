"""LoRA forward hooks (`millm/ml/adapter_hooks.py`), on tiny REAL models on the CPU.

The fixtures are chosen to disagree with the defects they guard against:

- projections are NON-SQUARE (hidden 48, intermediate 80) with grouped-query attention (k/v project
  to 24), so a swapped A and B is a shape error rather than a silent transpose;
- lora_B is NON-ZERO on every target (PEFT initialises it to zero, which would make every numeric
  check pass with the hook doing nothing);
- `alpha != rank`, so a dropped or inverted scale changes the answer;
- every numeric check runs in float32 AND bfloat16 — in float32 the "compute the delta in float32"
  rule agrees with its defect by construction; only bfloat16 can tell them apart.
"""

from __future__ import annotations

import copy
import hashlib

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from millm.ml import adapter_hooks
from millm.ml.adapter_hooks import (
    AdapterInstallError,
    AdapterPlanError,
    LoraHookSet,
    adapter_shapes,
    check_target_modules,
    live_module_path,
    lora_scaling,
    parse_peft_key,
    plan_targets,
    target_entry_matches,
)

RANK = 4
ALPHA = 6.0  # scaling 1.5: neither 1 nor rank, so a dropped or inverted scale is visible
SCALING = ALPHA / RANK

LLAMA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def tiny_llama(dtype: torch.dtype = torch.float32) -> nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=48,
        intermediate_size=80,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=12,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config).to(dtype).eval()


def tiny_lfm2(dtype: torch.dtype = torch.float32) -> nn.Module:
    from transformers import Lfm2Config, Lfm2ForCausalLM

    torch.manual_seed(0)
    config = Lfm2Config(
        vocab_size=64,
        hidden_size=48,
        intermediate_size=80,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention"],
        block_auto_adjust_ff_dim=False,
        max_position_embeddings=64,
    )
    return Lfm2ForCausalLM(config).to(dtype).eval()


def peft_key(path: str, which: str) -> str:
    return f"base_model.model.{path}.lora_{which}.weight"


def make_adapter(model: nn.Module, paths, *, rank: int = RANK, seed: int = 1) -> dict:
    """PEFT-keyed fp32 tensors with a NON-ZERO B on every target."""
    gen = torch.Generator().manual_seed(seed)
    tensors = {}
    for path in paths:
        module = model.get_submodule(path)
        tensors[peft_key(path, "A")] = torch.randn(rank, module.in_features, generator=gen) * 0.2
        tensors[peft_key(path, "B")] = torch.randn(module.out_features, rank, generator=gen) * 0.2
    return tensors


def llama_paths(model: nn.Module) -> list[str]:
    return [
        n for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n.split(".")[-1] in LLAMA_TARGETS
    ]


LFM2_PATHS = [
    "model.layers.0.conv.in_proj",
    "model.layers.1.self_attn.q_proj",
    "model.layers.0.feed_forward.w2",
    "model.layers.1.feed_forward.w2",
]
LFM2_ENTRIES = ["in_proj", "q_proj", "w2"]


def prepare(model, tensors, entries, *, rank: int = RANK, scaling: float = SCALING):
    targets = plan_targets(model, adapter_shapes(tensors), rank=rank, target_modules=entries)
    return LoraHookSet.build(model, targets, tensors, scaling)


def ids() -> torch.Tensor:
    return torch.tensor([[1, 5, 9, 2, 33, 17, 4, 60]])


@torch.no_grad()
def logits(model) -> torch.Tensor:
    return model(input_ids=ids()).logits


def fingerprint(model) -> dict:
    out = {}
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        raw = p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        out[name] = (p.data_ptr(), p.dtype, hashlib.sha256(raw).hexdigest())
    return out


def hook_count(model) -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


# ---------------------------------------------------------------------------------------------
# Keys, targets, scaling
# ---------------------------------------------------------------------------------------------


class TestPeftKeys:
    def test_a_and_b_parse(self):
        assert parse_peft_key("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight") == (
            "model.layers.0.self_attn.q_proj", "A",
        )
        assert parse_peft_key("base_model.model.model.layers.0.mlp.down_proj.lora_B.weight") == (
            "model.layers.0.mlp.down_proj", "B",
        )

    @pytest.mark.parametrize("key", [
        "model.layers.0.self_attn.q_proj.lora_A.weight",           # no PEFT prefix
        "base_model.model.model.layers.0.q_proj.lora_A.default.weight",  # adapter-named key
        "base_model.model.model.layers.0.q_proj.lora_magnitude_vector",  # DoRA
        "base_model.model.model.layers.0.q_proj.lora_B.bias",       # lora bias
        "base_model.model..lora_A.weight",
        "base_model.model.lora_A.weight",
    ])
    def test_anything_else_does_not_parse(self, key):
        assert parse_peft_key(key) is None


class TestTargetMatching:
    def test_suffix_on_a_dot_boundary(self):
        assert target_entry_matches("model.layers.0.self_attn.q_proj", "q_proj")
        assert target_entry_matches("model.layers.0.self_attn.q_proj", "self_attn.q_proj")
        assert not target_entry_matches("model.layers.0.self_attn.xq_proj", "q_proj")
        assert not target_entry_matches("model.layers.0.self_attn.q_proj", "proj")

    def test_a_full_path_matches_itself(self):
        path = "model.layers.0.self_attn.q_proj"
        assert target_entry_matches(path, path)

    def test_an_entry_is_literal_never_a_regex(self):
        path = "model.layers.0.self_attn.q_proj"
        for regex in ("self_attn.*_proj", "(q|v)_proj", "q_pro[j]"):
            offenders = check_target_modules({path: (path,)}, [regex])
            assert sorted(o["reason"] for o in offenders) == ["target_not_declared", "target_unused"]

    def test_every_entry_must_match_a_path(self):
        offenders = check_target_modules({"a.q_proj": ("a.q_proj",)}, ["q_proj", "v_proj"])
        assert [o["reason"] for o in offenders] == ["target_unused"]
        assert "v_proj" in offenders[0]["detail"]

    def test_every_path_must_be_declared(self):
        offenders = check_target_modules({"a.q_proj": ("a.q_proj",), "a.v_proj": ("a.v_proj",)},
                                         ["q_proj"])
        assert offenders == [{
            "reason": "target_not_declared", "module": "a.v_proj",
            "detail": "the weights file targets a module no target_modules entry names",
        }]

    @pytest.mark.parametrize("entry", ["", ".q_proj", "q_proj.", "a..q_proj"])
    def test_malformed_entries_are_refused(self, entry):
        offenders = check_target_modules({"a.q_proj": ("a.q_proj",)}, ["q_proj", entry])
        assert [o["reason"] for o in offenders] == ["invalid_target_entry"]


class TestScaling:
    def test_alpha_over_r_is_the_default_when_absent(self):
        assert lora_scaling(6, 4) == 1.5
        assert lora_scaling(6, 4, "alpha_over_r") == 1.5

    def test_any_other_rule_is_refused(self):
        with pytest.raises(AdapterPlanError) as e:
            lora_scaling(6, 4, "rslora")
        assert e.value.reason == "scaling"

    @pytest.mark.parametrize("alpha,rank", [(0, 4), (-1, 4), (float("nan"), 4), (6, 0), (6, True)])
    def test_degenerate_values_are_refused(self, alpha, rank):
        with pytest.raises(AdapterPlanError):
            lora_scaling(alpha, rank)


# ---------------------------------------------------------------------------------------------
# Planning refusals
# ---------------------------------------------------------------------------------------------


class TestPlanning:
    def test_a_valid_llama_adapter_plans_every_target(self):
        model = tiny_llama()
        paths = llama_paths(model)
        tensors = make_adapter(model, paths)
        targets = plan_targets(model, adapter_shapes(tensors), rank=RANK,
                               target_modules=list(LLAMA_TARGETS))
        assert [t.module_path for t in targets] == sorted(paths)
        k = next(t for t in targets if t.module_path.endswith("0.self_attn.k_proj"))
        assert (k.in_features, k.out_features, k.rank) == (48, 24, RANK)

    def test_a_module_of_another_rank_is_refused(self):
        """D-10: one `adapter.rank`, one scale. A rank-2 module among rank-4 ones would be served
        at alpha/4 while trained at alpha/2."""
        model = tiny_llama()
        tensors = make_adapter(model, ["model.layers.0.self_attn.q_proj"])
        tensors.update(make_adapter(model, ["model.layers.1.self_attn.q_proj"], rank=2))
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(model, adapter_shapes(tensors), rank=RANK, target_modules=["q_proj"])
        assert e.value.reason == "rank"
        assert [o["module"] for o in e.value.offenders] == ["model.layers.1.self_attn.q_proj"]

    def test_one_dimension_of_one_module_off_by_one_is_refused(self):
        model = tiny_llama()
        paths = llama_paths(model)
        tensors = make_adapter(model, paths)
        key = peft_key("model.layers.1.mlp.down_proj", "A")
        tensors[key] = torch.randn(RANK, 79)  # in_features is 80
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(model, adapter_shapes(tensors), rank=RANK, target_modules=list(LLAMA_TARGETS))
        assert e.value.reason == "shape"
        assert [o["module"] for o in e.value.offenders] == ["model.layers.1.mlp.down_proj"]

    def test_out_features_off_by_one_is_refused(self):
        model = tiny_llama()
        tensors = make_adapter(model, ["model.layers.0.self_attn.k_proj"])
        tensors[peft_key("model.layers.0.self_attn.k_proj", "B")] = torch.randn(25, RANK)
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(model, adapter_shapes(tensors), rank=RANK, target_modules=["k_proj"])
        assert e.value.reason == "shape"

    def test_swapped_a_and_b_is_refused(self):
        model = tiny_llama()
        path = "model.layers.0.mlp.gate_proj"  # 48 -> 80
        tensors = make_adapter(model, [path])
        a, b = tensors[peft_key(path, "A")], tensors[peft_key(path, "B")]
        tensors = {peft_key(path, "A"): b.T.contiguous(), peft_key(path, "B"): a.T.contiguous()}
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(model, adapter_shapes(tensors), rank=RANK, target_modules=["gate_proj"])
        assert e.value.reason in {"rank", "shape"}

    def test_a_missing_module_is_refused(self):
        model = tiny_llama()
        tensors = make_adapter(model, ["model.layers.0.self_attn.q_proj"])
        a = tensors.pop(peft_key("model.layers.0.self_attn.q_proj", "A"))
        b = tensors.pop(peft_key("model.layers.0.self_attn.q_proj", "B"))
        tensors[peft_key("model.layers.7.self_attn.q_proj", "A")] = a
        tensors[peft_key("model.layers.7.self_attn.q_proj", "B")] = b
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(model, adapter_shapes(tensors), rank=RANK, target_modules=["q_proj"])
        assert e.value.reason == "missing_module"

    def test_an_lfm2_conv1d_is_refused_as_not_a_linear(self):
        model = tiny_lfm2()
        shapes = {
            peft_key("model.layers.0.conv.conv", "A"): (RANK, 3),
            peft_key("model.layers.0.conv.conv", "B"): (48, RANK),
        }
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(model, shapes, rank=RANK, target_modules=["conv.conv"])
        assert e.value.reason == "unsupported_module"
        assert "Conv1d" in e.value.offenders[0]["detail"]

    def test_an_integer_weight_linear_is_refused(self):
        """A quantized linear subclasses nn.Linear and carries integer weights."""
        model = tiny_llama()
        q = model.get_submodule("model.layers.0.self_attn.q_proj")
        q.weight = nn.Parameter(torch.zeros(48, 48, dtype=torch.int8), requires_grad=False)
        tensors = make_adapter(model, ["model.layers.0.self_attn.q_proj"])
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(model, adapter_shapes(tensors), rank=RANK, target_modules=["q_proj"])
        assert e.value.reason == "unsupported_module"

    def test_a_non_peft_key_is_refused_not_skipped(self):
        model = tiny_llama()
        tensors = make_adapter(model, ["model.layers.0.self_attn.q_proj"])
        tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector"] = torch.ones(48)
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(model, adapter_shapes(tensors), rank=RANK, target_modules=["q_proj"])
        assert e.value.reason == "unsupported_tensor"

    def test_an_incomplete_pair_is_refused(self):
        model = tiny_llama()
        tensors = make_adapter(model, ["model.layers.0.self_attn.q_proj"])
        del tensors[peft_key("model.layers.0.self_attn.q_proj", "B")]
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(model, adapter_shapes(tensors), rank=RANK, target_modules=["q_proj"])
        assert e.value.reason == "incomplete_pair"

    def test_an_empty_file_is_refused(self):
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(tiny_llama(), {}, rank=RANK, target_modules=["q_proj"])
        assert e.value.reason == "empty"

    def test_target_modules_must_agree_with_the_weights_file(self):
        model = tiny_llama()
        tensors = make_adapter(model, ["model.layers.0.self_attn.q_proj",
                                       "model.layers.0.self_attn.v_proj"])
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(model, adapter_shapes(tensors), rank=RANK, target_modules=["q_proj", "o_proj"])
        reasons = sorted(o["reason"] for o in e.value.offenders)
        assert reasons == ["target_not_declared", "target_unused"]

    def test_every_offender_is_collected(self):
        model = tiny_llama()
        tensors = make_adapter(model, ["model.layers.0.self_attn.q_proj",
                                       "model.layers.1.self_attn.q_proj"])
        tensors[peft_key("model.layers.0.self_attn.q_proj", "A")] = torch.randn(2, 48)
        tensors[peft_key("model.layers.1.self_attn.q_proj", "B")] = torch.randn(47, RANK)
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(model, adapter_shapes(tensors), rank=RANK, target_modules=["q_proj"])
        assert sorted(o["reason"] for o in e.value.offenders) == ["rank", "shape"]
        assert "+1 more" in str(e.value)


class TestTransformersRenaming:
    def test_the_renaming_api_is_available_on_this_transformers(self):
        """If this goes red, keys are being used unrenamed and fused targets read as
        `missing_module` rather than `fused_by_transformers`."""
        assert adapter_hooks._conversion_renamer(tiny_llama()) is not None

    def test_a_llama_key_maps_to_itself(self):
        renamer = adapter_hooks._conversion_renamer(tiny_llama())
        path = "model.layers.0.self_attn.q_proj"
        assert live_module_path(path, renamer) == (path, None)

    @pytest.fixture(scope="class")
    def mixtral(self):
        from transformers import MixtralConfig, MixtralForCausalLM

        torch.manual_seed(0)
        return MixtralForCausalLM(MixtralConfig(
            vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, num_local_experts=2,
            num_experts_per_tok=1,
        )).eval()

    def test_a_per_expert_key_is_refused_as_fused(self, mixtral):
        path = "model.layers.0.block_sparse_moe.experts.0.w1"
        shapes = {peft_key(path, "A"): (RANK, 32), peft_key(path, "B"): (48, RANK)}
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(mixtral, shapes, rank=RANK, target_modules=["w1"])
        assert e.value.reason == "fused_by_transformers"
        assert "gate_up_proj" in e.value.offenders[0]["detail"]

    def test_a_renamed_checkpoint_key_reaches_the_live_module(self, mixtral):
        """`block_sparse_moe.gate` is the checkpoint name; transformers serves it as `mlp.gate`.
        The refusal names the LIVE module (a router, not a Linear), which proves the key was
        renamed rather than reported missing."""
        path = "model.layers.0.block_sparse_moe.gate"
        shapes = {peft_key(path, "A"): (RANK, 32), peft_key(path, "B"): (2, RANK)}
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(mixtral, shapes, rank=RANK, target_modules=["gate"])
        assert e.value.reason == "unsupported_module"
        assert e.value.offenders[0]["module"] == "model.layers.0.mlp.gate"

    def test_a_converter_whose_target_is_a_weight_is_still_refused_as_fused(self, monkeypatch):
        """Some architectures fuse dense `gate_proj`/`up_proj` into one `gate_up_proj.weight`
        (`conversion_mapping.py`). The renamed key still ends in `.weight`, so only the converter
        check names the real reason; without it the key would read as an ordinary module."""
        def renamer(key):
            if key.endswith(".gate_proj.weight"):
                return key.replace(".gate_proj.", ".gate_up_proj."), r"\.gate_proj.weight"
            return key, None

        monkeypatch.setattr(adapter_hooks, "_conversion_renamer", lambda m: renamer)
        model = tiny_llama()
        path = "model.layers.0.mlp.gate_proj"
        tensors = make_adapter(model, [path])
        with pytest.raises(AdapterPlanError) as e:
            plan_targets(model, adapter_shapes(tensors), rank=RANK, target_modules=["gate_proj"])
        assert e.value.reason == "fused_by_transformers"
        assert e.value.offenders[0]["module"] == path

    def test_two_keys_renaming_onto_one_module_are_refused(self):
        def renamer(key):
            return key.replace(".old_proj.", ".q_proj."), None

        model = tiny_llama()
        a = "model.layers.0.self_attn.q_proj"
        b = "model.layers.0.self_attn.old_proj"
        shapes = {peft_key(a, "A"): (RANK, 48), peft_key(a, "B"): (48, RANK),
                  peft_key(b, "A"): (RANK, 48), peft_key(b, "B"): (48, RANK)}
        original = adapter_hooks._conversion_renamer
        adapter_hooks._conversion_renamer = lambda m: renamer
        try:
            with pytest.raises(AdapterPlanError) as e:
                plan_targets(model, shapes, rank=RANK, target_modules=["q_proj", "old_proj"])
        finally:
            adapter_hooks._conversion_renamer = original
        assert e.value.reason == "duplicate_target"


# ---------------------------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------------------------


DTYPES = [torch.float32, torch.bfloat16]


class TestTheHookComputesTheLoraDelta:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_module_output_is_base_plus_the_fp32_delta(self, dtype):
        """Exact (`torch.equal`): the hooked output is the base output plus
        `scaling · B·(A·x)`, the delta computed in float32, ADDED in float32, and the sum cast to
        the module dtype once — PEFT's arithmetic (see `test_adapter_peft_golden.py`)."""
        model = tiny_llama(dtype)
        path = "model.layers.0.mlp.gate_proj"
        tensors = make_adapter(model, [path])
        module = model.get_submodule(path)
        x = torch.randn(2, 5, 48, generator=torch.Generator().manual_seed(3)).to(dtype)
        with torch.no_grad():
            base = module(x)
            handles = LoraHookSet.install(model, prepare(model, tensors, ["gate_proj"]))
            hooked = module(x)
        a, b = tensors[peft_key(path, "A")], tensors[peft_key(path, "B")]
        expected = (base.float() + F.linear(F.linear(x.float(), a), b) * SCALING).to(dtype)
        assert hooked.dtype == dtype
        assert torch.equal(hooked, expected)
        LoraHookSet.remove(handles)

    def test_bf16_fixture_distinguishes_fp32_from_bf16_compute(self):
        """The test above can only catch a bf16-compute defect if the two paths differ here."""
        model = tiny_llama(torch.bfloat16)
        path = "model.layers.0.mlp.gate_proj"
        tensors = make_adapter(model, [path])
        a, b = tensors[peft_key(path, "A")], tensors[peft_key(path, "B")]
        x = torch.randn(2, 5, 48, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
        fp32 = (F.linear(F.linear(x.float(), a), b) * SCALING).to(torch.bfloat16)
        bf16 = F.linear(F.linear(x, a.bfloat16()), b.bfloat16()) * SCALING
        assert not torch.equal(fp32, bf16)

    def test_bf16_fixture_distinguishes_one_rounding_from_two(self):
        """The exact test above can only catch "round the delta to bf16, then add in bf16" (two
        roundings; PEFT rounds once) if the two differ on this fixture."""
        model = tiny_llama(torch.bfloat16)
        path = "model.layers.0.mlp.gate_proj"
        tensors = make_adapter(model, [path])
        module = model.get_submodule(path)
        a, b = tensors[peft_key(path, "A")], tensors[peft_key(path, "B")]
        x = torch.randn(2, 5, 48, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
        with torch.no_grad():
            base = module(x)
        delta = F.linear(F.linear(x.float(), a), b) * SCALING
        assert not torch.equal((base.float() + delta).to(torch.bfloat16),
                               base + delta.to(torch.bfloat16))

    def test_matches_the_hand_formula_independently(self):
        """`x·Wᵀ + s·(x·Aᵀ)·Bᵀ`, written with matmul rather than the hook's F.linear."""
        model = tiny_llama()
        path = "model.layers.1.self_attn.k_proj"
        tensors = make_adapter(model, [path])
        module = model.get_submodule(path)
        x = torch.randn(3, 48, generator=torch.Generator().manual_seed(4))
        handles = LoraHookSet.install(model, prepare(model, tensors, ["k_proj"]))
        with torch.no_grad():
            got = module(x)
        a, b = tensors[peft_key(path, "A")], tensors[peft_key(path, "B")]
        want = x @ module.weight.T + SCALING * ((x @ a.T) @ b.T)
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)
        LoraHookSet.remove(handles)

    @pytest.mark.parametrize("builder,paths,entries", [
        (tiny_llama, None, list(LLAMA_TARGETS)),
        (tiny_lfm2, LFM2_PATHS, LFM2_ENTRIES),
    ])
    def test_whole_model_equals_merged_weights(self, builder, paths, entries):
        """Every target at once: the hooked model's logits equal a copy whose weights were merged
        `W + s·B·A` (float32, so the comparison is about the formula, not rounding)."""
        model = builder()
        paths = paths or llama_paths(model)
        tensors = make_adapter(model, paths)
        merged = copy.deepcopy(model)
        with torch.no_grad():
            for path in paths:
                a, b = tensors[peft_key(path, "A")], tensors[peft_key(path, "B")]
                merged.get_submodule(path).weight += SCALING * (b @ a)
        handles = LoraHookSet.install(model, prepare(model, tensors, entries))
        try:
            assert torch.allclose(logits(model), logits(merged), atol=1e-4, rtol=1e-4)
        finally:
            LoraHookSet.remove(handles)

    def test_every_target_contributes(self):
        """Dropping any single target changes the logits, so a hook set that installs all but one
        cannot pass the merged-weights test."""
        model = tiny_llama()
        paths = llama_paths(model)
        tensors = make_adapter(model, paths)
        handles = LoraHookSet.install(model, prepare(model, tensors, list(LLAMA_TARGETS)))
        full = logits(model)
        LoraHookSet.remove(handles)
        for path in paths:
            partial = {k: v for k, v in tensors.items() if f".{path}." not in k}
            entries = [e for e in LLAMA_TARGETS
                       if any(target_entry_matches(p, e) for p in paths if p != path)]
            handles = LoraHookSet.install(model, prepare(model, partial, entries))
            assert not torch.equal(logits(model), full), path
            LoraHookSet.remove(handles)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_a_and_b_are_float32_on_the_weight_device(self, dtype):
        model = tiny_lfm2(dtype)
        prepared = prepare(model, make_adapter(model, LFM2_PATHS), LFM2_ENTRIES)
        for path, (a, b) in prepared.weights.items():
            weight = model.get_submodule(path).weight
            assert a.dtype == b.dtype == torch.float32
            assert a.device == b.device == weight.device

    def test_build_copies_rather_than_aliases_the_callers_tensors(self):
        model = tiny_llama()
        path = "model.layers.0.self_attn.q_proj"
        tensors = make_adapter(model, [path])
        prepared = prepare(model, tensors, ["q_proj"])
        a, _ = prepared.weights[path]
        assert a.data_ptr() != tensors[peft_key(path, "A")].data_ptr()

    def test_a_keyword_call_is_hooked_too(self):
        model = tiny_llama()
        path = "model.layers.0.self_attn.q_proj"
        tensors = make_adapter(model, [path])
        module = model.get_submodule(path)
        x = torch.randn(2, 48)
        handles = LoraHookSet.install(model, prepare(model, tensors, ["q_proj"]))
        with torch.no_grad():
            assert torch.equal(module(input=x), module(x))
            assert not torch.equal(module(x), F.linear(x, module.weight))
        LoraHookSet.remove(handles)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_tensors_are_refused_at_build(self, bad):
        model = tiny_llama()
        path = "model.layers.0.self_attn.q_proj"
        tensors = make_adapter(model, [path])
        tensors[peft_key(path, "B")][0, 0] = bad
        with pytest.raises(AdapterPlanError) as e:
            prepare(model, tensors, ["q_proj"])
        assert e.value.reason == "non_finite"


# ---------------------------------------------------------------------------------------------
# Removal and base integrity
# ---------------------------------------------------------------------------------------------


class TestRemovalRestoresTheBase:
    @pytest.mark.parametrize("builder,paths,entries", [
        (tiny_llama, None, list(LLAMA_TARGETS)),
        (tiny_lfm2, LFM2_PATHS, LFM2_ENTRIES),
    ])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_outputs_change_while_applied_and_return_exactly(self, builder, paths, entries, dtype):
        model = builder(dtype)
        paths = paths or llama_paths(model)
        base = logits(model)
        before = fingerprint(model)
        names = [n for n, _ in model.named_parameters()]
        handles = LoraHookSet.install(model, prepare(model, make_adapter(model, paths), entries))
        assert hook_count(model) == len(paths)
        assert not torch.equal(logits(model), base)
        assert fingerprint(model) == before
        assert [n for n, _ in model.named_parameters()] == names
        LoraHookSet.remove(handles)
        assert handles == []
        assert hook_count(model) == 0
        assert torch.equal(logits(model), base)
        assert fingerprint(model) == before

    def test_dynamo_is_reset_after_install_and_after_remove(self, monkeypatch):
        calls = []
        monkeypatch.setattr(adapter_hooks, "reset_dynamo_for_hook_change", lambda: calls.append(1))
        model = tiny_llama()
        prepared = prepare(model, make_adapter(model, llama_paths(model)), list(LLAMA_TARGETS))
        handles = LoraHookSet.install(model, prepared)
        assert len(calls) == 1
        LoraHookSet.remove(handles)
        assert len(calls) == 2


class TestInstallIsAllOrNothing:
    def _five(self):
        model = tiny_llama()
        paths = [f"model.layers.{i}.self_attn.{p}" for i in (0, 1) for p in ("q_proj", "k_proj")]
        paths.append("model.layers.0.mlp.down_proj")
        tensors = make_adapter(model, paths)
        prepared = prepare(model, tensors, ["q_proj", "k_proj", "down_proj"])
        assert len(prepared.targets) == 5
        return model, prepared

    def test_a_failure_at_the_third_install_leaves_no_hook(self, monkeypatch):
        model, prepared = self._five()
        base = logits(model)
        third = model.get_submodule(prepared.targets[2].module_path)

        def boom(*args, **kwargs):
            raise RuntimeError("injected")

        monkeypatch.setattr(third, "register_forward_hook", boom)
        with pytest.raises(AdapterInstallError) as e:
            LoraHookSet.install(model, prepared)
        assert e.value.reason == "install_failed"
        assert e.value.installed_before_failure == 2
        assert e.value.target == prepared.targets[2].module_path
        assert hook_count(model) == 0
        assert torch.equal(logits(model), base)

    def test_a_module_replaced_since_build_refuses_and_rolls_back(self):
        model, prepared = self._five()
        base = logits(model)
        path = prepared.targets[3].module_path
        parent_path, _, leaf = path.rpartition(".")
        old = model.get_submodule(path)
        replacement = nn.Linear(old.in_features, old.out_features, bias=False)
        replacement.weight = old.weight
        setattr(model.get_submodule(parent_path), leaf, replacement)
        with pytest.raises(AdapterInstallError) as e:
            LoraHookSet.install(model, prepared)
        assert e.value.installed_before_failure == 3
        assert hook_count(model) == 0
        assert torch.equal(logits(model), base)
