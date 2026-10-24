"""Inline and explicitly-unsteered per-request steering: apply, validate, restore (Feature 28,
FR-28.1, FR-28.2; FTASKS 3.x, 4.1).

Real `LoadedSAE`s in the real `AttachedSAEState`, two SAEs at two layers built from different
seeds, so "the other entry" is a different object with different values.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from millm.api.schemas.openai import InlineSteering
from millm.core.errors import InvalidFeatureIndexError, InvalidParameterError, SAENotAttachedError
from millm.services.inference_service import InferenceService, get_request_steering
from millm.services.sae_service import AttachedSAEState
from tests.unit.f28_fixtures import (
    SAE_A,
    SAE_B,
    build,
    chat,
    clean_state,  # noqa: F401 - fixture
    text,
)


def inline(features, sae_id=None) -> InlineSteering:
    return InlineSteering(sae_id=sae_id,
                          features=[{"index": i, "strength": s} for i, s in features])


@pytest.fixture
def two(clean_state):  # noqa: F811
    served = build([(SAE_A, 0, 11), (SAE_B, 1, 22)])
    # Live global steering on BOTH entries, as an operator or a circuit leaves it: the
    # restore must bring back these exact values and flags.
    served.sae(SAE_A, 0).set_steering_batch({1: 3.0, 2: -1.5})
    served.sae(SAE_A, 0).enable_steering(True)
    served.sae(SAE_B, 1).set_steering_batch({4: 7.0})
    served.sae(SAE_B, 1).enable_steering(True)
    yield served
    for handle in served.handles:
        handle.remove()


@pytest.fixture
def one(clean_state):  # noqa: F811
    served = build([(SAE_A, 0, 11)])
    yield served
    for handle in served.handles:
        handle.remove()


# ── 3.1 the dispatcher ──────────────────────────────────────────────────────────────────────


class TestDispatcher:
    async def _dispatch(self, svc, request, monkeypatch):
        calls = []

        def spy(name, ret):
            async def a(*args, **kwargs):
                calls.append((name, args, kwargs))
                return ret

            def s(*args, **kwargs):
                calls.append((name, args, kwargs))
                return ret
            return a if name == "profile" else s

        monkeypatch.setattr(svc, "_apply_inline_steering", spy("inline", {"layers": []}))
        monkeypatch.setattr(svc, "_apply_explicit_unsteered", spy("unsteered", {"layers": []}))
        monkeypatch.setattr(svc, "_apply_request_steering", spy("profile", {"values": {}}))
        result = await svc._dispatch_request_steering(request, request_id="rid")
        return calls, result

    @pytest.mark.parametrize("make", [chat, text])
    async def test_each_branch_is_chosen_for_its_input(self, one, make, monkeypatch):
        svc = one.svc
        steering = {"features": [{"index": 1, "strength": 2.0}]}
        calls, _ = await self._dispatch(svc, make(steering=steering), monkeypatch)
        assert [c[0] for c in calls] == ["inline"]
        assert calls[0][1][0].features[0].index == 1 and calls[0][1][1] == "rid"

        calls, _ = await self._dispatch(svc, make(steering={"features": []}), monkeypatch)
        assert [c[0] for c in calls] == ["unsteered"] and calls[0][1] == ("rid",)

        calls, _ = await self._dispatch(svc, make(profile="humor"), monkeypatch)
        assert [(c[0], c[1], c[2]) for c in calls] == [
            ("profile", ("humor", None), {"request_id": "rid"})]

        calls, _ = await self._dispatch(svc, make(steering_intensity=0.5), monkeypatch)
        assert [(c[0], c[1], c[2]) for c in calls] == [
            ("profile", (None, 0.5), {"request_id": "rid"})]

        calls, result = await self._dispatch(svc, make(), monkeypatch)
        assert calls == [] and result is None

    async def test_the_admission_epoch_is_recorded_first(self, one):
        AttachedSAEState().bump_steering_epoch("test")
        epoch = AttachedSAEState().steering_epoch
        await one.svc._dispatch_request_steering(chat(), request_id="r")
        record = get_request_steering()
        assert record.kind == "none" and record.epoch_at_admission == epoch


# ── 3.2 – 3.5 the inline apply ──────────────────────────────────────────────────────────────


class TestInlineApply:
    def test_target_runs_exactly_the_set_and_others_are_disabled(self, two):
        before_b = two.sae(SAE_B, 1).get_steering_values()
        saved = two.svc._apply_inline_steering(inline([(5, 8.0), (6, 0.0)], SAE_A), "r1")
        a, b = two.sae(SAE_A, 0), two.sae(SAE_B, 1)
        # Cleared before set: the live {1: 3.0, 2: -1.5} is gone, the zero is not applied.
        assert a.get_steering_values() == {5: 8.0}
        assert a.is_steering_enabled is True
        # T-79: DISABLED, not cleared and not suppressed.
        assert b.is_steering_enabled is False
        assert b.get_steering_values() == before_b
        assert b._suppressed is False and a._suppressed is False
        assert saved["kind"] == "inline" and saved["request_id"] == "r1"
        assert {(s["sae_id"], s["layer"]) for s in saved["layers"]} == {(SAE_A, 0), (SAE_B, 1)}

    def test_selection_is_by_sae_id_not_first_entry(self, two):
        two.svc._apply_inline_steering(inline([(3, 2.0)], SAE_B), "r")
        assert two.sae(SAE_B, 1).get_steering_values() == {3: 2.0}
        assert two.sae(SAE_B, 1).is_steering_enabled is True
        assert two.sae(SAE_A, 0).is_steering_enabled is False

    def test_strengths_are_clamped_and_recorded(self, one):
        one.svc._apply_inline_steering(inline([(3, 500.0), (4, -250.0), (5, 199.0)]), "r")
        assert one.sae(SAE_A, 0).get_steering_values() == {3: 200.0, 4: -200.0, 5: 199.0}
        record = get_request_steering()
        assert record.kind == "inline" and record.clamped == 2
        assert record.applied == {3: 200.0, 4: -200.0, 5: 199.0}
        assert (record.sae_id, record.layer) == (SAE_A, 0)

    def test_sole_entry_is_used_when_sae_id_is_omitted(self, one):
        one.svc._apply_inline_steering(inline([(2, 1.0)]), "r")
        assert one.sae(SAE_A, 0).get_steering_values() == {2: 1.0}

    def test_an_unattached_sae_id_is_refused_naming_it(self, two):
        before = two.state()
        with pytest.raises(SAENotAttachedError) as exc:
            two.svc._apply_inline_steering(inline([(1, 1.0)], "sae_gamma"), "r")
        assert "sae_gamma" in exc.value.message
        assert two.state() == before

    def test_no_sae_attached_with_a_non_empty_set_is_refused(self, clean_state):  # noqa: F811
        svc = build([]).svc
        with pytest.raises(SAENotAttachedError):
            svc._apply_inline_steering(inline([(1, 1.0)]), "r")

    def test_sae_id_omitted_with_two_entries_names_each(self, two):
        before = two.state()
        with pytest.raises(InvalidParameterError) as exc:
            two.svc._apply_inline_steering(inline([(1, 1.0)]), "r")
        assert f"({SAE_A}, layer 0)" in exc.value.message
        assert f"({SAE_B}, layer 1)" in exc.value.message
        assert exc.value.status_code == 400
        assert two.state() == before

    def test_one_sae_id_at_two_layers_is_refused_naming_each(self, clean_state):  # noqa: F811
        served = build([(SAE_A, 0, 11), (SAE_A, 1, 33)])
        try:
            with pytest.raises(InvalidParameterError) as exc:
                served.svc._apply_inline_steering(inline([(1, 1.0)], SAE_A), "r")
            assert "layer 0" in exc.value.message and "layer 1" in exc.value.message
        finally:
            for h in served.handles:
                h.remove()

    def test_an_index_past_d_sae_is_refused_and_nothing_changes(self, two):
        before = two.state()
        epoch = AttachedSAEState().steering_epoch
        with pytest.raises(InvalidFeatureIndexError) as exc:
            two.svc._apply_inline_steering(inline([(1, 1.0), (8, 1.0)], SAE_A), "r")
        assert "8" in exc.value.message and "[0, 8)" in exc.value.message
        assert exc.value.details["d_sae"] == 8 and exc.value.details["feature_idx"] == 8
        assert two.state() == before
        assert AttachedSAEState().steering_epoch == epoch

    def test_a_mutation_raising_mid_apply_restores_before_reraising(self, two, monkeypatch):
        """FTASKS 3.10: `set_steering_batch` raising after the clear already happened."""
        before = two.state()
        target = two.sae(SAE_A, 0)
        real = target.set_steering_batch
        calls = []

        def boom(values):  # the APPLY's write raises once; the restore's write must work
            calls.append(dict(values))
            if len(calls) == 1:
                raise RuntimeError("gpu hiccup")
            return real(values)

        monkeypatch.setattr(target, "set_steering_batch", boom)
        with pytest.raises(RuntimeError, match="gpu hiccup"):
            two.svc._apply_inline_steering(inline([(5, 8.0)], SAE_A), "r")
        monkeypatch.undo()
        assert calls == [{5: 8.0}, {1: 3.0, 2: -1.5}], "the rollback did not run"
        assert two.state() == before


# ── 3.6 explicit unsteered ──────────────────────────────────────────────────────────────────


class TestExplicitUnsteered:
    def test_every_entry_is_disabled_and_values_kept(self, two):
        before = two.state()
        saved = two.svc._apply_explicit_unsteered("r")
        assert [e.sae.is_steering_enabled for e in AttachedSAEState().entries()] == [False, False]
        assert [v for _i, _l, v, _e in two.state()] == [v for _i, _l, v, _e in before]
        assert saved["kind"] == "unsteered"
        assert get_request_steering().kind == "unsteered"

    def test_nothing_attached_is_accepted_and_returns_none(self, clean_state):  # noqa: F811
        assert build([]).svc._apply_explicit_unsteered("r") is None

    async def test_monitoring_still_captures_on_an_unsteered_generation(self, two):
        """T-80: DISABLED, not suppressed — monitoring and sensing still record."""
        b = two.sae(SAE_B, 1)
        b.enable_monitoring(True)
        assert b.get_last_feature_activations() is None
        from tests.unit.f28_fixtures import no_cuda

        with no_cuda():
            await two.svc.create_chat_completion(chat(steering={"features": []}))
        assert b.get_last_feature_activations() is not None, (
            "an explicitly unsteered generation silenced monitoring: suppressed, not disabled"
        )


# ── 3.7 restore, 3.11 epoch ─────────────────────────────────────────────────────────────────


class TestRestore:
    @pytest.mark.parametrize("apply", ["inline", "unsteered"])
    def test_restore_returns_every_entry_values_and_enabled_exactly(self, two, apply):
        two.sae(SAE_B, 1).enable_steering(False)  # a disabled entry must come back DISABLED
        before = two.state()
        saved = (two.svc._apply_inline_steering(inline([(5, 8.0)], SAE_A), "r")
                 if apply == "inline" else two.svc._apply_explicit_unsteered("r"))
        assert two.state() != before
        two.svc._restore_request_profile(saved)
        assert two.state() == before

    def test_restore_reaches_the_second_entry(self, two):
        """The branch is on `layers`: keyed on `circuit` an inline saved shape fell to the
        single-SAE branch, which restores only the FIRST entry."""
        saved = two.svc._apply_inline_steering(inline([(5, 8.0)], SAE_A), "r")
        assert two.sae(SAE_B, 1).is_steering_enabled is False
        two.svc._restore_request_profile(saved)
        assert two.sae(SAE_B, 1).is_steering_enabled is True

    def test_an_epoch_bump_mid_request_skips_the_restore(self, two):
        saved = two.svc._apply_inline_steering(inline([(5, 8.0)], SAE_A), "r")
        AttachedSAEState().bump_steering_epoch("operator")
        two.svc._restore_request_profile(saved)
        assert two.sae(SAE_A, 0).get_steering_values() == {5: 8.0}, (
            "the superseded restore ran and reverted an authoritative write"
        )

    @pytest.mark.parametrize("apply", ["inline", "unsteered"])
    def test_inline_and_unsteered_never_bump_the_epoch(self, two, apply):
        epoch = AttachedSAEState().steering_epoch
        saved = (two.svc._apply_inline_steering(inline([(5, 8.0)], SAE_A), "r")
                 if apply == "inline" else two.svc._apply_explicit_unsteered("r"))
        assert AttachedSAEState().steering_epoch == epoch
        two.svc._restore_request_profile(saved)
        assert AttachedSAEState().steering_epoch == epoch

    def test_finish_captures_before_restoring(self, two):
        saved = two.svc._apply_inline_steering(inline([(5, 8.0)], SAE_A), "r")
        snapshot = two.svc._finish_request_steering(saved)
        by_key = {(e.sae_id, e.layer): e for e in snapshot.entries}
        assert by_key[(SAE_A, 0)].applied == {5: 8.0} and by_key[(SAE_A, 0)].enabled
        assert by_key[(SAE_B, 1)].enabled is False
        assert two.sae(SAE_A, 0).get_steering_values() == {1: 3.0, 2: -1.5}, "restored after"


# ── 3.9 one dispatcher, one finisher ────────────────────────────────────────────────────────

#: The ONLY callers allowed to call `_restore_request_profile` directly, each with its reason and
#: the exact number of calls. Generation sites restore through `_finish_request_steering`.
RESTORE_CALLERS = {
    "_finish_request_steering": (1, "capture-then-restore, the non-streaming finish"),
    "stream_chat_completion": (2, "setup-error restore and the `finally` restore: the stream "
                                  "captures before its final chunk, and its `finally` runs after "
                                  "the stream has closed"),
    "_apply_request_circuit_steering": (1, "rollback of a failed circuit dial"),
    "_apply_inline_steering": (1, "rollback of a partial inline apply"),
    "_apply_explicit_unsteered": (1, "rollback of a partial unsteered apply"),
}


def _calls_to(method: str) -> dict[str, int]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(InferenceService)))
    found: dict[str, int] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == method and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"):
                found[fn.name] = found.get(fn.name, 0) + 1
    return found


def test_only_the_dispatcher_applies_and_only_the_finisher_restores():
    assert _calls_to("_apply_request_steering") == {"_dispatch_request_steering": 1}
    restores = _calls_to("_restore_request_profile")
    assert restores == {name: n for name, (n, _why) in RESTORE_CALLERS.items()}, restores
    finishers = _calls_to("_finish_request_steering")
    assert finishers == {"create_chat_completion": 1, "_create_batched_chat_completion": 1,
                         "create_text_completion": 1}, finishers


# ── 4.1 serial routing ──────────────────────────────────────────────────────────────────────


class TestSerialRouting:
    @pytest.mark.parametrize("make", [chat, text])
    @pytest.mark.parametrize("steering", [
        {"features": [{"index": 1, "strength": 2.0}]}, {"features": []},
    ])
    def test_a_steering_request_never_routes_to_the_batching_manager(self, one, make, steering,
                                                                     monkeypatch):
        svc = one.svc
        svc._cbm_backend = SimpleNamespace(
            is_running=True, sampling_params_match=lambda t, p: True,
        )
        monkeypatch.setattr(svc, "_use_cbm", lambda: True)
        assert InferenceService._has_steering_override(make(steering=steering)) is True
        assert svc._use_cbm_for_request(**svc._cbm_route_kwargs(make(steering=steering))) is False
        # Control: the same request without `steering` would go to the manager.
        assert svc._use_cbm_for_request(**svc._cbm_route_kwargs(make())) is True

    @pytest.mark.parametrize("make", [chat, text])
    def test_text_completions_carry_profile_and_dial_too(self, make):
        assert InferenceService._has_steering_override(make(profile="p")) is True
        assert InferenceService._has_steering_override(make(steering_intensity=0.2)) is True
        assert InferenceService._has_steering_override(make()) is False
