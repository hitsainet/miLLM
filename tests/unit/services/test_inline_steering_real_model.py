"""Inline steering on a TINY REAL Llama with a REAL hooked `LoadedSAE`, greedy (Feature 28,
FTASKS 3.8 – 4.5; US-2, US-3, US-5; BRD-04 acceptance 12, the CPU half).

⚠ THE INLINE SET AND THE EQUIVALENT PROFILE ARE SEPARATE LITERALS. If both read one variable the
equality test would pass whether or not the two apply paths agree.
"""

from __future__ import annotations

import pytest

from millm.core.errors import ProfileNotFoundError
from millm.services.inference_service import get_steering_report
from millm.services.sae_service import AttachedSAEState
from tests.unit.f25_fixtures import make_service, word_model, word_tokenizer
from tests.unit.f28_fixtures import (
    SAE_A,
    add_profile,
    attach,
    chat,
    clean_state,  # noqa: F401 - fixture
    db,  # noqa: F401 - fixture
    independent_hash,
    no_cuda,
    text,
)

#: The profile, stored at λ = 1 (the profile path computes clamp(float(v) * 1.0)).
PROFILE_STEERING = {1: 30.0, 4: -25.0}
#: The same setting written independently as an inline request.
INLINE = {"features": [{"index": 4, "strength": -25.0}, {"index": 1, "strength": 30.0}]}


async def _chat_text(svc, **over) -> str:
    with no_cuda():
        out = await svc.create_chat_completion(chat(**over))
    return out.choices[0].message.content


async def _completion_text(svc, **over) -> str:
    with no_cuda():
        out = await svc.create_text_completion(text(**over))
    return out.choices[0].text


@pytest.fixture
def world(clean_state):  # noqa: F811
    """A service; the detached baseline is measured before anything attaches."""
    model, tokenizer = word_model(), word_tokenizer()
    svc = make_service(model, tokenizer)
    handles: list = []
    yield svc, model, handles
    for h in handles:
        h.remove()


async def test_inline_equals_an_equivalent_saved_profile(world, db):  # noqa: F811
    svc, model, handles = world
    detached = await _chat_text(svc)
    served = attach(model, [(SAE_A, 0, 11)])
    handles.extend(served.handles)
    await add_profile(db, "equivalent", PROFILE_STEERING)

    by_profile = await _chat_text(svc, profile="equivalent")
    profile_header = get_steering_report().header
    by_inline = await _chat_text(svc, steering=INLINE)
    inline_header = get_steering_report().header

    assert by_profile != detached, "precondition: this steering changes the greedy output"
    assert by_inline == by_profile, "US-2: inline and the equivalent profile must agree"
    expected = independent_hash(SAE_A, {1: 30.0, 4: -25.0})
    assert profile_header.startswith('profile;name="equivalent";source=request;intensity="1.0";')
    assert inline_header == f'inline;sae="{SAE_A}";layer=0;features=2;hash="{expected}"'
    assert f'hash="{expected}"' in profile_header, "same applied set, same hash, any kind"
    # Restored: nothing steers between requests.
    sae = served.sae(SAE_A, 0)
    assert sae.get_steering_values() == {} and sae.is_steering_enabled is False


@pytest.mark.parametrize("call", [_chat_text, _completion_text])
async def test_explicit_unsteered_under_an_active_profile_equals_detached(world, db, call):  # noqa: F811
    """US-3 and US-5: `features: []` under an active profile is the detached answer, reported
    `none`; the next request without the field is steered again."""
    svc, model, handles = world
    detached = await call(svc)
    served = attach(model, [(SAE_A, 0, 11)])
    handles.extend(served.handles)
    await add_profile(db, "active", {1: 30.0, 4: -25.0}, active=True)
    sae = served.sae(SAE_A, 0)
    sae.set_steering_batch({1: 30.0, 4: -25.0})   # what activation wrote
    sae.enable_steering(True)

    unsteered = await call(svc, steering={"features": []})
    assert get_steering_report().header == "none"
    assert unsteered == detached

    steered_again = await call(svc)
    assert steered_again != detached, "the active profile must steer the next request again"
    assert get_steering_report().header.startswith('profile;name="active";source=active;')
    assert sae.is_steering_enabled is True and sae.get_steering_values() == {1: 30.0, 4: -25.0}


async def test_a_text_completion_applies_a_named_profile(world, db):  # noqa: F811
    svc, model, handles = world
    detached = await _completion_text(svc)
    served = attach(model, [(SAE_A, 0, 11)])
    handles.extend(served.handles)
    await add_profile(db, "named", PROFILE_STEERING)
    out = await _completion_text(svc, profile="named")
    assert out != detached
    assert get_steering_report().header.startswith('profile;name="named";source=request;')
    assert await _completion_text(svc, steering=INLINE) == out


async def test_a_text_completion_naming_a_missing_profile_is_404(world, db):  # noqa: F811
    svc, model, handles = world
    served = attach(model, [(SAE_A, 0, 11)])
    handles.extend(served.handles)
    with pytest.raises(ProfileNotFoundError) as exc:
        await _completion_text(svc, profile="nope")
    assert exc.value.status_code == 404


async def test_a_text_completion_with_no_field_runs_and_reports_live_steering(world, db):  # noqa: F811
    """FR-28.4.4: no steering field → live global steering, as before, and now reported."""
    svc, model, handles = world
    detached = await _completion_text(svc)
    served = attach(model, [(SAE_A, 0, 11)])
    handles.extend(served.handles)
    sae = served.sae(SAE_A, 0)
    sae.set_steering_batch({2: 40.0})
    sae.enable_steering(True)
    out = await _completion_text(svc)
    assert out != detached
    assert get_steering_report().header == (
        f'manual;sae="{SAE_A}";layer=0;features=1;hash="{independent_hash(SAE_A, {2: 40.0})}"')


async def test_a_multi_prompt_completion_has_one_steering_state(world, db):  # noqa: F811
    """FR-28.4.2: applied around EVERY prompt — both prompts match their single-prompt runs."""
    svc, model, handles = world
    served = attach(model, [(SAE_A, 0, 11)])
    handles.extend(served.handles)
    singles = [await _completion_text(svc, prompt=p, steering=INLINE) for p in ("w1 w2", "w5")]
    with no_cuda():
        out = await svc.create_text_completion(text(prompt=["w1 w2", "w5"], steering=INLINE))
    assert [c.text for c in out.choices] == singles
    assert get_steering_report().header.startswith("inline;")
    assert AttachedSAEState().entries()[0].sae.get_steering_values() == {}
