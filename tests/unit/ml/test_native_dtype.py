"""miLLM's precision rule matches miStudio's, case for case.

`docs/schemas/native-dtype-cases.json` is the rule; miStudio tests its resolver against the same
file, and the byte-identity check keeps the two copies one. Two resolvers drifting onto different
rules is the defect this ends: probes fitted on float16 activations, served here over bfloat16.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from millm.ml.native_dtype import (
    LOAD_DTYPES,
    checkpoint_dtype_of,
    resolve_for_config,
    resolve_load_dtype,
)

REPO = Path(__file__).resolve().parents[3]
CASES_PATH = REPO / "docs" / "schemas" / "native-dtype-cases.json"
MISTUDIO = Path(os.environ.get("MISTUDIO_REPO", "/home/x-sean/app/miStudio"))
CASES = json.loads(CASES_PATH.read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['quantization']}-{c['checkpoint_dtype']}-pq{c['pre_quantized']}")
def test_every_case_in_the_shared_table(case):
    resolved = resolve_load_dtype(
        case["quantization"], case["checkpoint_dtype"], pre_quantized=case["pre_quantized"]
    )
    assert resolved.name == case["loads_at"]
    assert resolved.torch_dtype is getattr(torch, case["loads_at"])
    assert resolved.storage_name == case["storage_dtype"]
    assert resolved.source == case["source"]


def test_the_case_table_is_miStudios():
    theirs = MISTUDIO / "docs" / "schemas" / "native-dtype-cases.json"
    if not theirs.exists():
        if os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1":
            pytest.fail(f"miStudio's case table is missing at {theirs}")
        pytest.skip("miStudio checkout not present")
    assert theirs.read_bytes() == CASES_PATH.read_bytes()


def test_the_llama_case_that_failed_parity():
    """An FP16 row of a bfloat16 checkpoint loads bfloat16 — what this server already did, and
    what miStudio now does too."""
    assert resolve_for_config("FP16", {"torch_dtype": "bfloat16"}).torch_dtype is torch.bfloat16


def test_a_float16_checkpoint_now_loads_float16_here():
    """The behaviour change on this side: a checkpoint published in float16 was cast to bfloat16."""
    assert resolve_for_config("FP16", {"torch_dtype": "float16"}).torch_dtype is torch.float16


def test_an_fp32_row_really_loads_float32():
    assert resolve_load_dtype("FP32", "bfloat16").torch_dtype is torch.float32


def test_nothing_recorded_is_none_not_a_default():
    assert checkpoint_dtype_of({}) == (None, "default")
    assert resolve_for_config("FP16", {}).name == "bfloat16"


def test_the_contract_enum_is_this_rules_output():
    assert set(LOAD_DTYPES) == {"float16", "bfloat16", "float32"}


def test_a_pre_quantized_checkpoint_keeps_the_16_bit_rule_whatever_its_label():
    """An FP32 label on a GPTQ/AWQ/FP8 checkpoint would load its unquantized modules at float32
    while the size plan assumed 16 bits."""
    from millm.ml.native_dtype import rule_quantization

    assert rule_quantization("FP32", is_pre_quantized=True) == "FP16"
    assert rule_quantization("FP32", is_pre_quantized=False) == "FP32"
    assert rule_quantization("Q4", is_pre_quantized=False) == "Q4"
