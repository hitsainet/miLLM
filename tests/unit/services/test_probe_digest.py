"""Detector and decision digests, computed (FR-38.7 computed half; FTASKS 4.1).

* RFC 8785's number vectors (Appendix B) and UTF-16 key order;
* a threshold-only change moves the decision digest and never the detector digest (control M7,
  first half: put the threshold or `decision` into the detector projection — red);
* excluded keys (`name`, `evidence`, `provenance`, `test_vectors`, …) change neither; a weight
  changes the detector digest;
* the stored JSON round trip cannot change a digest (T-S1-14; the PostgreSQL half is in
  `tests/schema/test_probe_result_contract_migration.py`).

⚠ The case file that PROVES equality with miStudio's implementation arrives at R-039 (FTASKS 11.2).
"""

from __future__ import annotations

import copy
import json
import struct

import pytest

from millm.services.probe_digest import (
    _number,
    decision_digest,
    detector_digest,
    detector_payload,
    digests_of,
    jcs,
)
from tests.unit.score_fixtures import tiny_definition

#: RFC 8785 Appendix B: IEEE-754 bits → canonical text.
RFC8785_NUMBERS = [
    ("0000000000000000", "0"), ("8000000000000000", "0"), ("0000000000000001", "5e-324"),
    ("8000000000000001", "-5e-324"), ("7fefffffffffffff", "1.7976931348623157e+308"),
    ("ffefffffffffffff", "-1.7976931348623157e+308"), ("4340000000000000", "9007199254740992"),
    ("c340000000000000", "-9007199254740992"), ("4430000000000000", "295147905179352830000"),
    ("44b52d02c7e14af5", "9.999999999999997e+22"), ("44b52d02c7e14af6", "1e+23"),
    ("44b52d02c7e14af7", "1.0000000000000001e+23"), ("444b1ae4d6e2ef4e", "999999999999999700000"),
    ("444b1ae4d6e2ef4f", "999999999999999900000"), ("444b1ae4d6e2ef50", "1e+21"),
    ("3eb0c6f7a0b5ed8c", "9.999999999999997e-7"), ("3eb0c6f7a0b5ed8d", "0.000001"),
    ("41b3de4355555553", "333333333.3333332"), ("41b3de4355555554", "333333333.33333325"),
    ("41b3de4355555555", "333333333.3333333"), ("41b3de4355555556", "333333333.3333334"),
    ("41b3de4355555557", "333333333.33333343"), ("becbf647612f3696", "-0.0000033333333333333333"),
    ("43143ff3c1cb0959", "1424953923781206.2"),
]


@pytest.mark.parametrize("bits,text", RFC8785_NUMBERS)
def test_rfc8785_numbers(bits, text):
    assert _number(struct.unpack(">d", bytes.fromhex(bits))[0]) == text


def test_rfc8785_key_order_is_utf16():
    expected = '{"a":[1,0,"\u00e9\\n"],"\u20ac":1,"\U0001f600":2,"\ufb33":3}'
    assert jcs({"\u20ac": 1, "\U0001f600": 2, "\ufb33": 3, "a": [1.0, -0.0, "\u00e9\n"]}) == expected


def test_a_threshold_only_change_moves_the_decision_and_never_the_detector():
    a = tiny_definition(threshold=1.0)
    b = copy.deepcopy(a)
    b["decision"]["threshold"] = 2.0
    assert detector_digest(a) == detector_digest(b)
    assert decision_digest(a["decision"]) != decision_digest(b["decision"])


@pytest.mark.parametrize("key,value", [
    ("name", "renamed"), ("description", "other words"), ("concept", "other"),
    ("provenance", {"exported_at": "2026-10-23T00:00:00Z"}), ("evidence", None),
    ("behaviour", {"key": "b-other", "name": "other", "version": 3}), ("kind", "x"),
])
def test_excluded_keys_change_no_detector_digest(key, value):
    a = tiny_definition()
    b = copy.deepcopy(a)
    b[key] = value
    assert detector_digest(a) == detector_digest(b)


def test_test_vectors_are_excluded():
    a = tiny_definition()
    b = copy.deepcopy(a)
    b["test_vectors"]["vectors"][0]["token_ids"] = [9, 9, 9]
    assert detector_digest(a) == detector_digest(b)


def test_one_weight_changes_the_detector_digest():
    a = tiny_definition()
    b = copy.deepcopy(a)
    b["head"]["weights"][3] = 0.5
    assert detector_digest(a) != detector_digest(b)


def test_absent_and_defaulted_keys_digest_equal():
    """`head.bias` absent is its default 0.0; `-0.0` and `0.0` hash equal (JCS writes `0`)."""
    a = tiny_definition()
    b = copy.deepcopy(a)
    del b["head"]["bias"]
    c = copy.deepcopy(a)
    c["head"]["bias"] = -0.0
    assert detector_digest(a) == detector_digest(b) == detector_digest(c)


def test_the_projection_is_inclusive():
    payload = detector_payload(tiny_definition())
    assert set(payload) == {"model", "read", "scope", "basis", "head", "sae", "aggregation",
                            "render"}
    assert "decision" not in json.dumps(payload)


def test_a_json_round_trip_cannot_change_a_digest():
    doc = tiny_definition()
    doc["decision"]["target_fpr"] = 1e-2
    again = json.loads(json.dumps(doc))
    assert digests_of(doc) == digests_of(again)


def test_a_non_finite_number_has_no_digest():
    """R-039: miStudio's `document_identity` refuses the WHOLE document when any projected value
    (`decision` included) is non-finite, so neither digest exists — S1 kept the detector half."""
    doc = tiny_definition()
    doc["decision"]["threshold"] = float("nan")
    assert digests_of(doc) == (None, None)
    big = tiny_definition()
    big["aggregation"]["params"] = {"n": 2 ** 53}
    assert digests_of(big) == (None, None), "an integer JCS cannot write exactly is refused"
    big["aggregation"]["params"] = {"n": 2 ** 53 - 1}
    assert None not in digests_of(big)
    excluded = tiny_definition()
    excluded["provenance"] = {"x": float("nan")}
    assert None not in digests_of(excluded), "only projected values are checked"
