"""
Tests for the Feature 8 cluster columns on the Profile model
(source_kind / cluster_meta / intensity / sensing_enabled).
"""

import pytest
from sqlalchemy import select

from millm.db.models.profile import Profile


@pytest.fixture
def make_profile():
    def _make(**overrides):
        defaults = dict(
            id="prof_test0001",
            name="test-profile",
            steering={"10": 1.5},
        )
        defaults.update(overrides)
        return Profile(**defaults)

    return _make


async def test_defaults_are_manual_lambda1(test_session, make_profile):
    """Existing creation paths (no new kwargs) must behave exactly as before."""
    profile = make_profile()
    test_session.add(profile)
    await test_session.commit()

    row = (await test_session.execute(select(Profile))).scalar_one()
    assert row.source_kind == "manual"
    assert row.is_cluster is False
    assert row.cluster_meta is None
    assert row.intensity == 1.0
    assert row.sensing_enabled is False


async def test_cluster_row_round_trips_meta(test_session, make_profile):
    meta = {
        "kind": "mistudio.cluster-definition",
        "schema_version": "1",
        "display_token": "fear",
        "members": [{"feature_idx": 10, "strength": 1.5, "sign": 1}],
        "budget": {"B": 2.4, "intensity": 1.0, "intensity_range": [0.5, 1.5]},
        "warnings": ["Layer mismatch: definition L12, attached L6"],
    }
    profile = make_profile(
        id="prof_cluster01",
        name="fear cluster",
        source_kind="cluster",
        cluster_meta=meta,
        intensity=1.25,
        sensing_enabled=True,
    )
    test_session.add(profile)
    await test_session.commit()
    test_session.expire_all()

    row = (await test_session.execute(select(Profile))).scalar_one()
    assert row.is_cluster is True
    assert row.cluster_meta == meta
    assert row.intensity == 1.25
    assert row.sensing_enabled is True


async def test_response_schema_exposes_cluster_fields(test_session, make_profile):
    from millm.api.schemas.profile import ProfileResponse

    profile = make_profile(source_kind="cluster", intensity=0.8)
    test_session.add(profile)
    await test_session.commit()  # routes always serve flushed rows (defaults applied)

    resp = ProfileResponse.from_profile(profile)
    assert resp.source_kind == "cluster"
    assert resp.intensity == 0.8
    assert resp.sensing_enabled is False
