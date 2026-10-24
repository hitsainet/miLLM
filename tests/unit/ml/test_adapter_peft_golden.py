"""miLLM's LoRA serving agrees with PEFT bit for bit (addendum D-12).

`tests/fixtures/peft_lora_golden/` was produced by real `peft` 0.21.2 (see `generate_golden.py`
there; peft is not a miLLM dependency, so the fixture is committed rather than regenerated): a
tiny bfloat16 Llama with non-square MLP projections and grouped-query k/v, a LoRA on all seven
projections with NON-ZERO B, `r=4, lora_alpha=12`, dropout 0.1, and PEFT's own logits.

The comparison is `torch.equal`, not a tolerance, ON A MACHINE THAT ROUNDS LIKE THE FIXTURE'S.
CPU bfloat16 matmuls take different kernels by instruction set (the fixture machine has
avx512_bf16; GitHub's runners did not, and CI failed both tests on 2026-10-16 while every local run
passed). `base_logits.safetensors` decides which case applies: if this machine reproduces the bare
base model's logits exactly, it shares the fixture's arithmetic and equality is required.
Everywhere, a tolerance check also runs. Its bound is this machine's own measured base drift, and it
is capped below the defect it exists to catch (0.03, see below). If the drift is too large to
separate the two, the test skips loudly rather than passing. PEFT adds the float32 delta to the bfloat16
base output IN FLOAT32 and rounds once; rounding the delta to bfloat16 first moved these logits by
0.03 against an adapter effect of 0.65, and every unit formula test still passed, because each
pinned the formula it was written from rather than the reference implementation.

The second test drives the whole directory path — the manifest tool writes the manifest for PEFT's
own `adapter_config.json`, then `prepare_adapter` resolves, verifies and builds — so a real PEFT
config (forty-odd keys from a current release) is also checked against `read_peft_config`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from millm.ml.adapter_hooks import LoraHookSet, adapter_shapes, plan_targets
from millm.services.adapter_artifact import ResidentBase, prepare_adapter
from millm.tools import write_adapter_manifest as tool

GOLDEN = Path(__file__).resolve().parents[2] / "fixtures" / "peft_lora_golden"
ENTRIES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RANK, ALPHA = 4, 12


#: The documented defect (rounding the delta to bfloat16 before adding) moved logits by 0.03; a
#: tolerance at or above half of it could no longer tell a correct adapter from that defect.
DEFECT_SHIFT = 0.03
TOLERANCE_FLOOR = 2.0**-8  # one bfloat16 ulp at the logits' scale (|logit| < 1)


def golden_base():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(**json.loads((GOLDEN / "base_config.json").read_text()))
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    model.load_state_dict(load_file(str(GOLDEN / "base_model.safetensors")))
    return model


@pytest.fixture(scope="module")
def reference():
    return load_file(str(GOLDEN / "peft_logits.safetensors"))


@torch.no_grad()
def run(model, reference):
    return model(input_ids=reference["input_ids"]).logits


@pytest.fixture(scope="module")
def base_reference():
    return load_file(str(GOLDEN / "base_logits.safetensors"))["logits"]


def assert_matches_peft(adapted, reference, base, base_reference):
    """Bit-exact where this machine rounds like the fixture's, else within its own measured drift."""
    drift = (base.float() - base_reference.float()).abs().max().item()
    tolerance = max(2.0 * drift, TOLERANCE_FLOOR)
    if tolerance >= DEFECT_SHIFT / 2:
        pytest.skip(
            f"this CPU's bfloat16 base drift ({drift:.4f}) is too large to separate a correct adapter "
            f"from the {DEFECT_SHIFT} rounding defect; the bit-exact check needs the fixture's arithmetic"
        )
    gap = (adapted.float() - reference.float()).abs().max().item()
    assert gap <= tolerance, f"adapted logits differ from PEFT's by {gap} (bound {tolerance}, drift {drift})"
    if drift == 0.0:
        assert torch.equal(adapted, reference), "same arithmetic as the fixture, so equality is required"


def test_the_fixture_is_the_peft_configuration_it_claims():
    config = json.loads((GOLDEN / "adapter_config.json").read_text())
    assert (config["r"], config["lora_alpha"], config["lora_dropout"]) == (RANK, ALPHA, 0.1)
    assert sorted(config["target_modules"]) == sorted(ENTRIES)
    assert config["peft_type"] == "LORA" and config["peft_version"] == "0.21.2"


def test_hooks_reproduce_peft_logits_exactly(reference, base_reference):
    model = golden_base()
    base = run(model, reference)
    tensors = load_file(str(GOLDEN / "adapter_model.safetensors"))
    targets = plan_targets(model, adapter_shapes(tensors), rank=RANK, target_modules=ENTRIES)
    assert len(targets) == 2 * len(ENTRIES)
    handles = LoraHookSet.install(model, LoraHookSet.build(model, targets, tensors, ALPHA / RANK))
    try:
        adapted = run(model, reference)
    finally:
        LoraHookSet.remove(handles)
    # The adapter's effect is far above bfloat16 rounding, so equality is not vacuous.
    assert (adapted.float() - base.float()).abs().max() > 0.1
    assert adapted.dtype == reference["logits"].dtype == torch.bfloat16
    assert_matches_peft(adapted, reference["logits"], base, base_reference)
    assert torch.equal(run(model, reference), base)


def test_the_whole_directory_path_reproduces_peft(tmp_path, reference, base_reference):
    root = tmp_path / "checkpoints"
    adapter_dir = root / "peft-golden" / "adapter-000000"
    adapter_dir.mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copyfile(GOLDEN / name, adapter_dir / name)
    assert tool.main([str(adapter_dir), "--base-name", "golden/tiny-llama"]) == 0

    model = golden_base()
    base = run(model, reference)
    resident = ResidentBase(manifest_sha256=None, names=frozenset({"golden/tiny-llama"}))
    artifact = prepare_adapter(
        "peft-golden/adapter-000000", model, resident, roots=[("primary", str(root))]
    )
    assert artifact.fields.scaling == ALPHA / RANK
    handles = LoraHookSet.install(model, artifact.prepared)
    try:
        assert_matches_peft(run(model, reference), reference["logits"], base, base_reference)
    finally:
        LoraHookSet.remove(handles)
