"""The steering-set hash and the `X-miLLM-Steering` grammar (Feature 28, FTDD §5.2 – §5.3).

⚠ THE TEST VECTORS ARE LITERALS COPIED OUT OF FTDD §5.3, NOT RECOMPUTED BY THE CODE UNDER TEST.
miDataworks 007 pins the same four (X-07). They were reproduced independently twice before being
pinned here (2026-10-07): once with a stand-alone Python one-liner (`struct` + `hashlib`, no
miLLM import) and once in Node (`DataView.setFloat64` + `crypto`), both byte-identical to the
FTDD. A test that computed its expectation through `canonical_set_form` would agree with any
change to it.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest

from millm.core.steering_state import (
    SteeringItem,
    applied_set,
    canonical_set_form,
    count_clamped,
    encode_name,
    format_intensity,
    serialize_steering_header,
    steering_set_hash,
)

LFM = "LiquidAI--LFM2.5-1.2B-Instruct-sae--layer_11"
GEMMA = "jbloom--gemma-2-2b-res-jb--layer_20--width_16k--average_l0_71"

# (id, sae, input features, canonical form, hash) — FTDD §5.3, verbatim.
VECTORS = [
    (
        "TV-1", LFM, [(1234, 8.0)],
        b"millm.steering-set/v1\nsae=LiquidAI--LFM2.5-1.2B-Instruct-sae--layer_11\n"
        b"1234:4020000000000000\n",
        "sha256:a4eae730e5105f422b93abeece7e03bda3c29b096c07aa9cee6f247e12844105",
    ),
    (
        "TV-2", LFM, [(1234, -8.0)],
        b"millm.steering-set/v1\nsae=LiquidAI--LFM2.5-1.2B-Instruct-sae--layer_11\n"
        b"1234:c020000000000000\n",
        "sha256:cc6c48faa720096e5c65fc1b2afab1b9b87659fb4465f20db421a2ed7fed78a7",
    ),
    (
        "TV-3", GEMMA, [(77, -2.5), (5, 0.1), (900, 0.0), (12, -0.0)],
        b"millm.steering-set/v1\nsae=jbloom--gemma-2-2b-res-jb--layer_20--width_16k--average_l0_71\n"
        b"5:3fb999999999999a\n77:c004000000000000\n",
        "sha256:b843912201c5c18f872976e35af288e5f484d81e31a03a1dc16c3b9dbd82e710",
    ),
    (
        "TV-4", GEMMA, [(3, 500.0)],
        b"millm.steering-set/v1\nsae=jbloom--gemma-2-2b-res-jb--layer_20--width_16k--average_l0_71\n"
        b"3:4069000000000000\n",
        "sha256:3f51779e6e33ee248acf6f520cb9475b4ef62f68275060bb69d2e228033b78df",
    ),
]


@pytest.mark.parametrize("vid,sae,pairs,form,digest", VECTORS, ids=[v[0] for v in VECTORS])
def test_published_vector_canonical_form_byte_for_byte(vid, sae, pairs, form, digest):
    assert canonical_set_form(sae, applied_set(pairs)) == form


@pytest.mark.parametrize("vid,sae,pairs,form,digest", VECTORS, ids=[v[0] for v in VECTORS])
def test_published_vector_hash_string_for_string(vid, sae, pairs, form, digest):
    assert steering_set_hash(sae, applied_set(pairs)) == digest


def test_p22_pair_differs_only_by_sign_and_has_two_hashes():
    """TV-1/TV-2: one feature axis (P-22), opposite strengths — two distinct hashes."""
    a = steering_set_hash(LFM, applied_set([(1234, 8.0)]))
    b = steering_set_hash(LFM, applied_set([(1234, -8.0)]))
    assert a != b


def test_hash_is_independent_of_input_order():
    forward = [(77, -2.5), (5, 0.1), (900, 0.0), (12, -0.0)]
    assert steering_set_hash(GEMMA, applied_set(forward)) == steering_set_hash(
        GEMMA, applied_set(list(reversed(forward)))
    )


def test_canonical_lines_are_in_ascending_numeric_not_lexical_order():
    """`10` sorts after `9` numerically and before it lexically."""
    form = canonical_set_form("s", {10: 1.0, 9: 1.0, 100: 1.0}).decode()
    assert form.splitlines()[2:] == [
        "9:3ff0000000000000", "10:3ff0000000000000", "100:3ff0000000000000",
    ]


def test_zeros_are_excluded_from_the_applied_set_and_the_hash():
    applied = applied_set([(1, 0.0), (2, -0.0), (3, 2.0)])
    assert applied == {3: 2.0}
    assert steering_set_hash("s", applied) == steering_set_hash("s", {3: 2.0})
    # A negative zero must not hash as `8000000000000000`.
    assert b"8000000000000000" not in canonical_set_form("s", {2: -0.0, 3: 2.0})


def test_clamp_changes_the_hash_and_is_counted():
    """TV-4: a client hashing the 500.0 it SENT gets a different hash; `clamped=1` says why."""
    sent = {3: 500.0}
    assert steering_set_hash(GEMMA, applied_set(sent.items())) != steering_set_hash(GEMMA, sent)
    assert applied_set(sent.items()) == {3: 200.0}
    assert count_clamped([(3, 500.0), (4, -201.0), (5, 200.0), (6, -200.0)]) == 2


def test_an_sae_id_with_a_line_break_is_refused():
    with pytest.raises(ValueError, match="line break"):
        canonical_set_form("a\n1:3ff0000000000000", {})


def test_a_non_finite_strength_cannot_be_hashed():
    with pytest.raises(ValueError):
        canonical_set_form("s", {1: float("nan")})


def test_canonical_form_has_no_spaces_and_ends_in_one_lf():
    form = canonical_set_form(LFM, {1: 1.5})
    assert b" " not in form and form.endswith(b"\n") and not form.endswith(b"\n\n")
    assert not form.startswith(b"\xef\xbb\xbf")


# --- header grammar -----------------------------------------------------------------------


def _inline(**kw):
    base = dict(kind="inline", sae=LFM, layer=11, features=1,
                hash="sha256:a4eae730e5105f422b93abeece7e03bda3c29b096c07aa9cee6f247e12844105")
    base.update(kw)
    return SteeringItem(**base)


def test_none_and_empty_serialise_as_none():
    assert serialize_steering_header([]) == "none"
    assert serialize_steering_header([SteeringItem(kind="none")]) == "none"
    assert serialize_steering_header([SteeringItem(kind="none", changed=True)]) == "none;changed"


def test_inline_matches_the_ftdd_example_exactly():
    assert serialize_steering_header([_inline()]) == (
        'inline;sae="LiquidAI--LFM2.5-1.2B-Instruct-sae--layer_11";layer=11;features=1;'
        'hash="sha256:a4eae730e5105f422b93abeece7e03bda3c29b096c07aa9cee6f247e12844105"'
    )


def test_profile_parameters_in_order_with_clamped_and_changed_bare():
    item = SteeringItem(kind="profile", name="humor", source="active", intensity=1.0,
                        sae="s", layer=3, features=12, hash="sha256:x", clamped=2, changed=True)
    assert serialize_steering_header([item]) == (
        'profile;name="humor";source=active;intensity="1.0";sae="s";layer=3;features=12;'
        'hash="sha256:x";clamped=2;changed'
    )


def test_clamped_zero_is_omitted():
    assert "clamped" not in serialize_steering_header([_inline(clamped=0)])
    assert serialize_steering_header([_inline(clamped=1)]).endswith(";clamped=1")


def test_circuit_item_and_composed():
    item = SteeringItem(kind="circuit", circuit_id="crc_124fd83d1f2a", intensity=0.5,
                        composed=True, changed=True)
    assert serialize_steering_header([item]) == (
        'circuit;id="crc_124fd83d1f2a";intensity="0.5";composed;changed'
    )


def test_unknown_with_reason():
    assert serialize_steering_header(
        [SteeringItem(kind="unknown", reason="claims_unreadable")]
    ) == "unknown;reason=claims_unreadable"


@pytest.mark.parametrize("sole", ["none", "unknown"])
def test_sole_kinds_combined_with_another_member_raise(sole):
    with pytest.raises(ValueError):
        serialize_steering_header([SteeringItem(kind=sole), _inline()])


def test_member_order_circuits_by_id_then_sae_items_by_layer_then_sae():
    items = [
        _inline(layer=12, sae="b"),
        SteeringItem(kind="circuit", circuit_id="crc_b", intensity=1.0),
        SteeringItem(kind="manual", sae="z", layer=3, features=1, hash="sha256:m"),
        _inline(layer=12, sae="a"),
        SteeringItem(kind="circuit", circuit_id="crc_a", intensity=1.0),
    ]
    header = serialize_steering_header(items)
    order = [m.split(";")[0] + ("" if "id=" not in m else m.split(";")[1]) for m in header.split(", ")]
    assert [m.split(";")[1] for m in header.split(", ")] == [
        'id="crc_a"', 'id="crc_b"', 'sae="z"', 'sae="a"', 'sae="b"'
    ], order


def test_format_intensity_is_shortest_round_trip():
    assert format_intensity(0.4375) == "0.4375"
    assert float(format_intensity(0.4375)) == 0.4375
    assert format_intensity(1) == "1.0"
    assert format_intensity(0.1) == "0.1"
    assert format_intensity(1e-05) == "1e-05"


def test_encode_name_percent_encodes_non_ascii_quote_backslash_and_percent():
    assert encode_name("humor") == "humor"
    assert encode_name('a"b\\c%d') == "a%22b%5Cc%25d"
    assert encode_name("café") == "caf%C3%A9"
    assert encode_name("tab\there") == "tab%09here"


def test_a_profile_name_cannot_inject_header_syntax():
    header = serialize_steering_header([SteeringItem(
        kind="profile", name='x", none;evil="1', source="request", intensity=1.0,
        sae="s", layer=1, features=1, hash="sha256:h",
    )])
    # The quote is percent-encoded, so the String cannot close early: the `, none` stays inside it.
    assert header.startswith('profile;name="x%22, none;evil=%221";source=request;')
    name_value = header[len('profile;name="'):header.index('";source=')]
    assert '"' not in name_value and "\\" not in name_value
    if _SFV is not None:
        http_sfv = importlib.import_module("http_sfv")
        parsed = http_sfv.List()
        parsed.parse(header.encode("ascii"))
        assert len(parsed) == 1 and str(parsed[0].value) == "profile"


# --- parsed by the consumer's RFC 8941 library (FTDD §3; 007 FTDD TQ7) ----------------------

_SFV = importlib.util.find_spec("http_sfv")


@pytest.mark.skipif(
    _SFV is None,
    reason="LOUD SKIP: http-sfv==0.9.9 (dev extra) is not installed, so the header grammar is "
    "NOT verified against an independent RFC 8941 parser in this run",
)
@pytest.mark.parametrize(
    "items,expected",
    [
        ([], [("none", {})]),
        ([SteeringItem(kind="none", changed=True)], [("none", {"changed": True})]),
        ([SteeringItem(kind="unknown", reason="read_failed")],
         [("unknown", {"reason": "read_failed"})]),
        ([_inline(clamped=1, changed=True)],
         [("inline", {"sae": LFM, "layer": 11, "features": 1,
                      "hash": "sha256:a4eae730e5105f422b93abeece7e03bda3c29b096c07aa9cee6f247e12844105",
                      "clamped": 1, "changed": True})]),
        ([SteeringItem(kind="profile", name='hü"mor', source="request", intensity=0.4375,
                       sae="s", layer=3, features=2, hash="sha256:h")],
         [("profile", {"name": "h%C3%BC%22mor", "source": "request", "intensity": "0.4375",
                       "sae": "s", "layer": 3, "features": 2, "hash": "sha256:h"})]),
        ([SteeringItem(kind="manual", sae='q"\\', layer=0, features=4, hash="sha256:m")],
         [("manual", {"sae": 'q"\\', "layer": 0, "features": 4, "hash": "sha256:m"})]),
        ([SteeringItem(kind="circuit", circuit_id="crc_1", intensity=0.5, composed=True),
          _inline(layer=2)],
         [("circuit", {"id": "crc_1", "intensity": "0.5", "composed": True}),
          ("inline", {"sae": LFM, "layer": 2, "features": 1,
                      "hash": "sha256:a4eae730e5105f422b93abeece7e03bda3c29b096c07aa9cee6f247e12844105"})]),
    ],
)
def test_every_kind_parses_with_http_sfv(items, expected):
    http_sfv = importlib.import_module("http_sfv")
    parsed = http_sfv.List()
    parsed.parse(serialize_steering_header(items).encode("ascii"))
    got = []
    for member in parsed:
        assert isinstance(member.value, http_sfv.Token), "the kind must be an RFC 8941 Token"
        got.append((str(member.value), dict(member.params)))
    assert got == expected
    # Booleans are real booleans and integers real integers, not strings.
    token_params = {"reason", "source"}
    for (_k, params), (_ek, eparams) in zip(got, expected):
        for key, value in eparams.items():
            if key in token_params:
                assert isinstance(params[key], http_sfv.Token), f"{key} must be a Token"
            else:
                assert type(params[key]) is type(value), key


def test_module_is_pure():
    """No torch, no DB, no services: a consumer can copy it (FTID §2 import rules)."""
    import ast
    import inspect

    import millm.core.steering_state as mod

    tree = ast.parse(inspect.getsource(mod))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    millm_imports = {m for m in imported if m.startswith("millm")}
    assert millm_imports == {"millm.core.steering_range"}
    assert not {m for m in imported if m.split(".")[0] in {"torch", "sqlalchemy", "fastapi"}}


def test_the_api_reference_publishes_exactly_these_vectors():
    """FTASKS 8.5 / X-07: the vectors consumers pin from the API reference are the ones pinned
    here. Skips LOUDLY where the manual is not shipped (the public mirror strips doc trees)."""
    from pathlib import Path

    page = Path(__file__).resolve().parents[3] / "manual/docs/api/openai-compatible.md"
    if not page.is_file():
        pytest.skip(f"LOUD SKIP: {page} is absent in this checkout (mirror view), so the published "
                    "vectors are NOT cross-checked against the pinned literals in this run")
    text = page.read_text()
    for vid, _sae, _pairs, form, digest in VECTORS:
        row = next(line for line in text.splitlines() if line.startswith(f"| {vid} |"))
        assert digest in row, f"{vid}: the published hash differs from the pinned one"
        assert form.decode().replace("\n", "\\n") in row, f"{vid}: published canonical form"
