"""Exercise for DR-02 (`millm.steering-set/v1`) — 040 FR-040.7. DR-47 (logger `millm.rule_runner`) was
renamed by 041 at P8 and retired; its exercise left with it (041 ID-3). DR-26 (the Open WebUI filter's file
and valve) was retired in this repo on 2026-11-11: filter v1.5.0 is `miserver_dial_filter.py` with the valve
`miserver_base_url`.
"""

from __future__ import annotations

from pathlib import Path

from miserver.core.steering_state import applied_set, canonical_set_form, steering_set_hash
from tests.unit.core.test_steering_state import VECTORS

REPO = Path(__file__).resolve().parents[4]


def test_steering_set_digest_under_frozen_kind():
    """TV-1..TV-4 are literals copied from the 028 FTDD (and pinned by miDataworks): every stored
    and sent steering-set digest is computed under `millm.steering-set/v1`."""
    assert len(VECTORS) == 4
    for vid, sae, features, canonical, digest in VECTORS:
        assert canonical.startswith(b"millm.steering-set/v1\n"), vid
        assert canonical_set_form(sae, applied_set(features)) == canonical, vid
        assert steering_set_hash(sae, applied_set(features)) == digest, vid
