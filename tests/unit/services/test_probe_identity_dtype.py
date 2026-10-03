"""A definition's stated precision is compared with the precision this server loaded at.

miStudio fitted every probe at float16 while this server served bfloat16, and nothing recorded
either, so the first sign was a parity failure (combined Δ 0.251) whose cause had to be found by
experiment. These pin the four places that make the precision visible and binding.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from millm.services import probe_arm_bridge, probe_arming
from millm.services.probe_identity import DTYPE_UNRECORDED, DTYPE_UNVERIFIED, check_identity
from millm.services.probe_parity import ParityReport, dtype_comparison
from tests.unit.services.test_probe_identity import loaded, model_block


class TestTheIdentityCheck:
    def test_a_definition_stating_the_loaded_precision_is_clean(self):
        report = check_identity(model_block(load_dtype="bfloat16"), loaded(dtype="bfloat16"))
        assert report.ok and DTYPE_UNRECORDED not in report.warnings

    def test_a_different_precision_is_a_mismatch(self):
        report = check_identity(model_block(load_dtype="float16"), loaded(dtype="bfloat16"))
        assert not report.ok
        assert report.mismatches == [{"field": "load_dtype", "expected": "float16", "actual": "bfloat16"}]

    def test_a_definition_stating_no_precision_is_warned_never_assumed_float16(self):
        """⚠ Every pre-2026-10-03 probe WAS float16 — but the document does not say so, and the
        check may not supply a fact the definition did not record."""
        block = model_block()
        block.pop("load_dtype", None)
        report = check_identity(block, loaded(dtype="bfloat16"))
        assert report.ok
        assert DTYPE_UNRECORDED in report.warnings


    def test_a_stated_precision_this_server_cannot_report_is_warned_not_passed_silently(self):
        report = check_identity(model_block(load_dtype="bfloat16"), loaded(dtype=None))
        assert report.ok and DTYPE_UNVERIFIED in report.warnings


class TestQuantizationIsIdentityToo:
    """Review round 1, MED-2: Q4 and FP16 loads of one bfloat16 checkpoint share a precision and
    read different activations (~0.93 cosine per token)."""

    def test_a_different_quantization_at_the_same_precision_is_a_mismatch(self):
        report = check_identity(model_block(load_dtype="bfloat16", quantization="Q4"),
                                loaded(dtype="bfloat16", quantization="FP16"))
        assert [m["field"] for m in report.mismatches] == ["quantization"]

    def test_the_same_quantization_is_clean(self):
        report = check_identity(model_block(load_dtype="bfloat16", quantization="FP16"),
                                loaded(dtype="bfloat16", quantization="FP16"))
        assert report.ok

    def test_matched_needs_the_quantization_to_agree(self):
        definition = {"model": {"load_dtype": "bfloat16", "quantization": "Q4"}}
        assert dtype_comparison(definition, "bfloat16", "FP16")["matched"] is False
        assert dtype_comparison(definition, "bfloat16", "Q4")["matched"] is True
        assert dtype_comparison(definition, "bfloat16", None)["matched"] is None

    def test_loaded_identity_passes_the_rows_quantization(self):
        (call,) = _calls(probe_arm_bridge.loaded_identity, "LoadedIdentity")
        (kw,) = [k for k in call.keywords if k.arg == "quantization"]
        assert ast.unparse(kw.value) == (
            "str(getattr(row.quantization, 'value', row.quantization)) if row is not None else None"
        )

    def test_parity_across_a_quantization_change_is_an_error_not_a_pass(self):
        """Round 3, MED-A: the diagnostic route judged a Q4 probe on an FP16 load under the LOOSER
        floor. Re-scoring across quantizations is not a check of the probe."""
        from millm.services.probe_parity import ProbeParityEngine

        report = ProbeParityEngine(forward=lambda *a: None).run(
            probe=None, definition={"model": {"load_dtype": "bfloat16", "quantization": "Q4"},
                                    "test_vectors": {"vectors": [{"token_ids": [1]}]}},
            tolerance=0.05, loaded_dtype="bfloat16", loaded_quantization="FP16")
        assert report.error and "quantization change" in report.error
        assert report.passed is False


class TestTheParityFloor:
    def test_the_comparison(self):
        assert dtype_comparison({"model": {"load_dtype": "bfloat16"}}, "bfloat16")["matched"] is True
        assert dtype_comparison({"model": {"load_dtype": "float16"}}, "bfloat16")["matched"] is False
        assert dtype_comparison({"model": {}}, "bfloat16")["matched"] is None

    @pytest.mark.parametrize("matched,expected", [(True, 0.31), (False, 0.10), (None, 0.10)])
    def test_the_floor_is_keyed_on_whether_the_precisions_match(self, monkeypatch, matched, expected):
        """Two DIFFERENT floors so the branch is observable — they are equal by default until the
        matched one is measured, which would let a wrong branch pass unnoticed."""
        from millm.core.config import settings

        monkeypatch.setattr(settings, "PROBE_PARITY_SCORE_TOLERANCE", 0.10)
        monkeypatch.setattr(settings, "PROBE_PARITY_MATCHED_DTYPE_FLOOR", 0.31)
        report = ParityReport(tolerance=0.05, dtype={"matched": matched})
        assert report.score_tolerance == expected

    def test_the_report_carries_the_comparison(self):
        details = ParityReport(tolerance=0.05, dtype={"recorded": "float16", "loaded": "bfloat16",
                                                      "matched": False}).as_details()
        assert details["dtype"]["recorded"] == "float16"


def _calls(fn, name):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and (getattr(n.func, "id", None) == name or getattr(n.func, "attr", None) == name)]


class TestTheWiring:
    def test_loaded_identity_passes_the_loaded_precision(self):
        (call,) = _calls(probe_arm_bridge.loaded_identity, "LoadedIdentity")
        dtype = [kw for kw in call.keywords if kw.arg == "dtype"]
        assert dtype and "current.dtype" in ast.unparse(dtype[0].value)

    @pytest.mark.parametrize("fn_name,who", [("arm", "loaded"), ("check_parity", "identity")])
    def test_both_parity_entry_points_pass_the_loaded_precision_and_quantization(self, fn_name, who):
        """The PAYLOAD, exactly — a keyword present with the wrong expression would pass a
        presence check (round 3, MED-B)."""
        from millm.api.routes.management import probes

        fn = probe_arming.ProbeArmingService.arm if fn_name == "arm" else probes.check_parity
        (call,) = [c for c in _calls(fn, "run") if any(k.arg == "loaded_dtype" for k in c.keywords)]
        got = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert got["loaded_dtype"] == f"{who}.dtype"
        assert got["loaded_quantization"] == f"{who}.quantization"

