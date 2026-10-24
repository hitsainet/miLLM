"""`millm/ml/weights_io.py` on tiny REAL Llama and LFM2 models (FTASKS 3.4, FTID §3.3).

In-process CPU round trips compare with `torch.equal` (addendum D-27), never a tolerance.

FIXTURES THAT DISAGREE WITH THE DEFECT
* Untied models: transformers 5.15 still DECLARES `lm_head -> embed_tokens` on them, so a rule
  that trusted the declaration would drop a real tensor.
* A module whose tie target is registered BEFORE its source, so "first name" and "declared source"
  give different answers (on Llama they coincide).
* Two views of one buffer at different offsets, and two empty buffers: a pointer-only grouping
  merges both.
* Two models of one shape built from different seeds: an in-place copy that copied nothing would
  otherwise pass.

MUTATION CONTROLS (recorded in the C1 report; each turned this file red)
* group by `data_ptr()` alone                         -> the views / empty-buffer tests
* keep the first name instead of the declared source  -> the target-registered-first test
* trust the declaration without the storage check     -> the untied naming tests
* `dst.copy_(src.to(dst.device))`                     -> the no-device-temporary test
* `dst.data = src`                                    -> the address-stability tests
* skip the pre-write comparison in `copy_in_place`    -> the refused-before-write test
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers import AutoModelForCausalLM

from millm.ml import weights_io
from millm.ml.weights_io import (
    InPlaceMismatchError,
    SafetensorsHeaderError,
    canonical_state,
    compare_for_in_place,
    copy_in_place,
    decide_mode,
    read_header,
    snapshot_to_host,
    tie_groups,
)
from tests.unit.weights_fixtures import BUILDERS, save_pretrained_dir, tiny_llama

ARCHS = sorted(BUILDERS)
INPUT = torch.tensor([[1, 5, 9, 33, 200, 7]])


def _build(arch: str, **kwargs):
    return BUILDERS[arch](**kwargs)


def _file(directory: Path) -> Path:
    return directory / "model.safetensors"


def _logits(model) -> torch.Tensor:
    with torch.no_grad():
        return model(INPUT).logits


def _addresses(model) -> dict[str, int]:
    return {n: t.data_ptr() for n, t in model.state_dict(keep_vars=True).items()}


# --------------------------------------------------------------------------- tie groups


class TestTieGroupsOnRealModels:
    @pytest.mark.parametrize("arch", ARCHS)
    def test_a_tied_model_has_one_group_for_two_names(self, arch):
        model = _build(arch, tied=True)
        groups = [g for g in tie_groups(model) if len(g) > 1]
        assert groups == [("model.embed_tokens.weight", "lm_head.weight")]
        state = canonical_state(model)
        assert "model.embed_tokens.weight" in state and "lm_head.weight" not in state

    @pytest.mark.parametrize("arch", ARCHS)
    def test_an_untied_model_keeps_both_names_despite_the_declaration(self, arch):
        model = _build(arch, tied=False)
        assert model._tied_weights_keys, "precondition: transformers declares the tie anyway"
        assert all(len(g) == 1 for g in tie_groups(model))
        state = canonical_state(model)
        assert "lm_head.weight" in state and "model.embed_tokens.weight" in state

    @pytest.mark.parametrize("tied", [True, False])
    @pytest.mark.parametrize("arch", ARCHS)
    def test_canonical_names_equal_what_save_pretrained_writes(self, arch, tied, tmp_path):
        """The inbound/outbound rule: miLLM's names are transformers' names."""
        model = _build(arch, tied=tied)
        save_pretrained_dir(model, tmp_path)
        with safe_open(_file(tmp_path), framework="pt") as f:
            written = set(f.keys())
        assert set(canonical_state(model)) == written

    @pytest.mark.parametrize("arch", ARCHS)
    def test_non_persistent_buffers_are_not_in_the_state(self, arch):
        model = _build(arch)
        buffers = {n for n, _ in model.named_buffers()}
        assert any("inv_freq" in n for n in buffers), "precondition: a rotary buffer exists"
        assert not any("inv_freq" in n for n in canonical_state(model))


class _TargetFirst(nn.Module):
    """lm_head registered BEFORE its tie source, tied by assignment."""

    _tied_weights_keys = {"lm_head.weight": "embed.weight"}

    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.embed = nn.Embedding(8, 4)
        self.lm_head.weight = self.embed.weight


class _Views(nn.Module):
    def __init__(self):
        super().__init__()
        base = torch.arange(8.0)
        self.register_buffer("low", base[:4])
        self.register_buffer("high", base[4:])
        self.register_buffer("prefix", base[:2])
        self.register_buffer("empty_a", torch.empty(0))
        self.register_buffer("empty_b", torch.empty(0))


class TestTieGroupingRules:
    def test_the_declared_source_wins_over_the_first_name(self):
        model = _TargetFirst()
        assert list(model.state_dict())[0] == "lm_head.weight", "precondition"
        assert list(canonical_state(model)) == ["embed.weight"]

    def test_views_at_other_offsets_or_shapes_are_distinct_tensors(self):
        groups = tie_groups(_Views())
        assert ("low",) in groups and ("high",) in groups and ("prefix",) in groups

    def test_empty_tensors_are_never_grouped(self):
        groups = tie_groups(_Views())
        assert ("empty_a",) in groups and ("empty_b",) in groups

    def test_an_undeclared_tie_keeps_the_first_name(self):
        model = _TargetFirst()
        model._tied_weights_keys = None
        assert list(canonical_state(model)) == ["lm_head.weight"]


# --------------------------------------------------------------------------- snapshot


class TestSnapshotToHost:
    def test_the_snapshot_is_a_copy_not_a_view(self):
        model = tiny_llama()
        snap = snapshot_to_host(model)
        before = snap["model.norm.weight"].clone()
        with torch.no_grad():
            model.model.norm.weight.add_(1.0)
        assert torch.equal(snap["model.norm.weight"], before)

    def test_the_snapshot_is_contiguous_on_the_host_and_shares_nothing(self):
        snap = snapshot_to_host(tiny_llama(tied=True))
        pointers = [t.untyped_storage().data_ptr() for t in snap.values()]
        assert len(set(pointers)) == len(pointers)
        assert all(t.device.type == "cpu" and t.is_contiguous() for t in snap.values())

    @pytest.mark.parametrize("tied", [True, False])
    @pytest.mark.parametrize("arch", ARCHS)
    def test_a_snapshot_file_loads_with_plain_from_pretrained(self, arch, tied, tmp_path):
        """Outbound portability: what miLLM writes, transformers (and miForge) reads."""
        model = _build(arch, tied=tied, seed=3)
        model.config.save_pretrained(tmp_path)
        save_file(snapshot_to_host(model), str(_file(tmp_path)), metadata={"format": "pt"})

        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()

        for name, tensor in model.state_dict().items():
            assert torch.equal(reloaded.state_dict()[name], tensor), name
        if tied:
            assert (
                reloaded.lm_head.weight.data_ptr()
                == reloaded.model.embed_tokens.weight.data_ptr()
            ), "ties must survive the round trip"
        assert torch.equal(_logits(reloaded), _logits(model))


# --------------------------------------------------------------------------- header


class TestReadHeader:
    @pytest.mark.parametrize("arch", ARCHS)
    def test_agrees_with_safetensors_and_returns_the_metadata(self, arch, tmp_path):
        save_pretrained_dir(_build(arch, dtype=torch.bfloat16), tmp_path)
        header = read_header(_file(tmp_path))
        assert header.metadata == {"format": "pt"}
        with safe_open(_file(tmp_path), framework="pt") as f:
            expected = {
                k: (tuple(f.get_slice(k).get_shape()), f.get_slice(k).get_dtype()) for k in f.keys()
            }
        assert {k: (v.shape, v.dtype) for k, v in header.tensors.items()} == expected
        assert {v.dtype for v in header.tensors.values()} == {"BF16"}

    def test_reads_only_the_header_bytes(self, tmp_path, monkeypatch):
        save_pretrained_dir(tiny_llama(), tmp_path)
        raw = _file(tmp_path).read_bytes()
        header_end = 8 + int.from_bytes(raw[:8], "little")
        read = []

        real_open = open

        def spy_open(path, mode="r", *args, **kwargs):
            handle = real_open(path, mode, *args, **kwargs)
            real_read = handle.read

            def counting_read(n=-1):
                data = real_read(n)
                read.append(len(data))
                return data

            handle.read = counting_read
            return handle

        monkeypatch.setattr(weights_io, "open", spy_open, raising=False)
        read_header(_file(tmp_path))
        assert sum(read) == header_end < len(raw)

    def test_garbage_tensor_data_does_not_change_the_header(self, tmp_path):
        save_pretrained_dir(tiny_llama(), tmp_path)
        path = _file(tmp_path)
        before = read_header(path)
        raw = bytearray(path.read_bytes())
        start = 8 + int.from_bytes(raw[:8], "little")
        raw[start:] = b"\xff" * (len(raw) - start)
        path.write_bytes(bytes(raw))
        assert read_header(path) == before

    def test_a_truncated_file_is_refused(self, tmp_path):
        save_pretrained_dir(tiny_llama(), tmp_path)
        path = _file(tmp_path)
        path.write_bytes(path.read_bytes()[:-16])
        with pytest.raises(SafetensorsHeaderError, match="outside the file"):
            read_header(path)

    @pytest.mark.parametrize(
        "content",
        [b"", b"\x01\x00", (10**12).to_bytes(8, "little"), (4).to_bytes(8, "little") + b"nope"],
    )
    def test_malformed_files_are_refused(self, tmp_path, content):
        path = tmp_path / "bad.safetensors"
        path.write_bytes(content)
        with pytest.raises(SafetensorsHeaderError):
            read_header(path)

    def test_metadata_must_be_strings(self, tmp_path):
        doc = json.dumps({"__metadata__": {"format": 1}}).encode()
        path = tmp_path / "meta.safetensors"
        path.write_bytes(len(doc).to_bytes(8, "little") + doc)
        with pytest.raises(SafetensorsHeaderError, match="__metadata__"):
            read_header(path)


# --------------------------------------------------------------------------- mode decision


class TestCompareForInPlace:
    @pytest.mark.parametrize("tied", [True, False])
    @pytest.mark.parametrize("arch", ARCHS)
    def test_a_save_pretrained_directory_of_the_same_shape_is_in_place(self, arch, tied, tmp_path):
        save_pretrained_dir(_build(arch, tied=tied, seed=1), tmp_path)
        verdict = compare_for_in_place(_build(arch, tied=tied, seed=2), read_header(_file(tmp_path)))
        assert verdict.in_place and verdict.mismatches == ()

    def test_a_dtype_difference_is_refused_never_cast(self, tmp_path):
        save_pretrained_dir(tiny_llama(dtype=torch.bfloat16), tmp_path)
        model = tiny_llama(dtype=torch.float32)
        verdict = compare_for_in_place(model, read_header(_file(tmp_path)))
        assert verdict.reasons == ("dtype",)
        assert {m.name for m in verdict.mismatches} == set(canonical_state(model))
        assert {(m.expected, m.found) for m in verdict.mismatches} == {("F32", "BF16")}

    def test_a_shape_difference_is_refused(self, tmp_path):
        save_pretrained_dir(tiny_llama(vocab_size=512), tmp_path)
        verdict = compare_for_in_place(tiny_llama(), read_header(_file(tmp_path)))
        assert verdict.reasons == ("shape",)
        assert {m.name for m in verdict.mismatches} == {"model.embed_tokens.weight"}

    def test_a_tied_checkpoint_into_an_untied_model_is_missing_a_name(self, tmp_path):
        save_pretrained_dir(tiny_llama(tied=True), tmp_path)
        verdict = compare_for_in_place(tiny_llama(tied=False), read_header(_file(tmp_path)))
        assert [(m.reason, m.name) for m in verdict.mismatches] == [("missing", "lm_head.weight")]

    def test_an_untied_checkpoint_into_a_tied_model_has_an_unexpected_name(self, tmp_path):
        save_pretrained_dir(tiny_llama(tied=False), tmp_path)
        verdict = compare_for_in_place(tiny_llama(tied=True), read_header(_file(tmp_path)))
        assert [(m.reason, m.name) for m in verdict.mismatches] == [
            ("unexpected", "lm_head.weight")
        ]

    def test_another_architecture_lists_every_reason(self, tmp_path):
        save_pretrained_dir(BUILDERS["lfm2"](), tmp_path)
        verdict = compare_for_in_place(tiny_llama(), read_header(_file(tmp_path)))
        assert not verdict.in_place
        assert {"missing", "unexpected"} <= set(verdict.reasons)


class TestDecideMode:
    def _verdict(self, tmp_path, *, matching: bool):
        save_pretrained_dir(tiny_llama(vocab_size=256 if matching else 512), tmp_path)
        return compare_for_in_place(tiny_llama(), read_header(_file(tmp_path)))

    def test_in_place_only_when_every_condition_holds(self, tmp_path):
        decision = decide_mode(
            self._verdict(tmp_path, matching=True), engine_is_transformers=True,
            base_model_matches=True,
        )
        assert decision.mode == "in_place" and decision.reasons == ()

    @pytest.mark.parametrize(
        ("engine", "base", "expected"),
        [(False, True, ("engine",)), (True, False, ("base_model",)),
         (False, False, ("engine", "base_model"))],
    )
    def test_each_identity_condition_forces_a_full_reload(self, tmp_path, engine, base, expected):
        decision = decide_mode(
            self._verdict(tmp_path, matching=True), engine_is_transformers=engine,
            base_model_matches=base,
        )
        assert decision.mode == "full_reload" and decision.reasons == expected

    def test_every_failing_reason_is_reported_together(self, tmp_path):
        decision = decide_mode(
            self._verdict(tmp_path, matching=False), engine_is_transformers=False,
            base_model_matches=False,
        )
        assert decision.reasons == ("engine", "base_model", "shape")


# --------------------------------------------------------------------------- in-place copy


class TestCopyInPlace:
    @pytest.mark.parametrize("tied", [True, False])
    @pytest.mark.parametrize("arch", ARCHS)
    def test_a_save_pretrained_directory_copies_in_and_keeps_every_address(
        self, arch, tied, tmp_path
    ):
        """Inbound compatibility: a transformers-written checkpoint, tied key dropped."""
        source = _build(arch, tied=tied, seed=11)
        resident = _build(arch, tied=tied, seed=22)
        save_pretrained_dir(source, tmp_path)
        assert not torch.equal(_logits(source), _logits(resident)), "precondition: seeds differ"
        addresses = _addresses(resident)

        count = copy_in_place(resident, _file(tmp_path))

        assert count == len(canonical_state(resident))
        assert _addresses(resident) == addresses, "an in-place load must keep every address"
        for name, tensor in source.state_dict().items():
            assert torch.equal(resident.state_dict()[name], tensor), name
        if tied:
            assert resident.lm_head.weight.data_ptr() == resident.model.embed_tokens.weight.data_ptr()
        assert torch.equal(_logits(resident), _logits(source))

    @pytest.mark.parametrize("arch", ARCHS)
    def test_a_snapshot_written_by_millm_copies_back_exactly(self, arch, tmp_path):
        source = _build(arch, tied=True, seed=5, dtype=torch.bfloat16)
        resident = _build(arch, tied=True, seed=6, dtype=torch.bfloat16)
        save_file(snapshot_to_host(source), str(_file(tmp_path)), metadata={"format": "pt"})
        copy_in_place(resident, _file(tmp_path))
        for name, tensor in source.state_dict().items():
            assert torch.equal(resident.state_dict()[name], tensor), name

    def test_an_open_handle_is_accepted(self, tmp_path):
        source, resident = tiny_llama(seed=1), tiny_llama(seed=2)
        save_pretrained_dir(source, tmp_path)
        with safe_open(_file(tmp_path), framework="pt", device="cpu") as handle:
            copy_in_place(resident, handle)
        assert torch.equal(_logits(resident), _logits(source))

    @pytest.mark.parametrize(
        "variant",
        [dict(dtype=torch.bfloat16), dict(vocab_size=512), dict(tied=False)],
        ids=["dtype", "shape", "names"],
    )
    def test_a_mismatch_is_refused_before_the_first_write(self, tmp_path, variant):
        save_pretrained_dir(tiny_llama(seed=1, **variant), tmp_path)
        resident = tiny_llama(seed=2)
        before = {n: t.clone() for n, t in resident.state_dict().items()}

        with pytest.raises(InPlaceMismatchError):
            copy_in_place(resident, _file(tmp_path))

        for name, tensor in resident.state_dict().items():
            assert torch.equal(tensor, before[name]), f"{name} was written before the refusal"

    def test_no_device_temporary_is_made(self, tmp_path, monkeypatch):
        """The source goes straight into `copy_`; `src.to(dst.device)` would allocate on the card."""
        source, resident = tiny_llama(seed=1), tiny_llama(seed=2)
        save_pretrained_dir(source, tmp_path)
        calls = []
        real_to = torch.Tensor.to

        def spy_to(self, *args, **kwargs):
            calls.append((args, kwargs))
            return real_to(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", spy_to)
        copy_in_place(resident, _file(tmp_path))
        monkeypatch.undo()

        assert calls == [], f"copy_in_place moved tensors before copying: {calls[:3]}"
        assert torch.equal(_logits(resident), _logits(source))
