"""Sensing arms on the cluster's OWN SAE, never the first attached entry (028 FU-1).

Two attached SAEs; the active cluster is bound to the SECOND. Toggling sensing on must arm the
second entry's SAE. Before FU-1 the route armed `attached_sae` — the first entry, a different
basis — so the toggle read "armed" while the cluster's own layer was never sensed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from millm.api.routes.management import sensing as sensing_routes


def _two_entries():
    first = SimpleNamespace(sae=MagicMock(name="first_sae"), sae_id="sae-a", layer=10)
    second = SimpleNamespace(sae=MagicMock(name="second_sae"), sae_id="sae-b", layer=12)
    return first, second


@pytest.mark.asyncio
async def test_toggle_arms_the_clusters_own_sae():
    first, second = _two_entries()
    profile = SimpleNamespace(
        id="prof_x", name="calm", source_kind="cluster", sae_id="sae-b", layer=12,
        is_active=True, sensing_enabled=False,
    )
    service = MagicMock()
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.commit = AsyncMock()
    with patch.object(sensing_routes, "_sensing_service", return_value=service), \
         patch("millm.db.base.async_session_factory", return_value=session), \
         patch("millm.db.repositories.profile_repository.ProfileRepository") as Repo, \
         patch("millm.services.sae_service.AttachedSAEState") as State:
        Repo.return_value.get = AsyncMock(return_value=profile)
        State.return_value.entries.return_value = [first, second]
        State.return_value.attached_sae = first.sae
        await sensing_routes._toggle("prof_x", True)
    service.arm_for_profile.assert_called_once_with(profile, second.sae)


@pytest.mark.asyncio
async def test_toggle_arms_nothing_when_its_sae_is_not_attached():
    first, _ = _two_entries()
    profile = SimpleNamespace(
        id="prof_x", name="calm", source_kind="cluster", sae_id="sae-z", layer=3,
        is_active=True, sensing_enabled=False,
    )
    service = MagicMock()
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.commit = AsyncMock()
    with patch.object(sensing_routes, "_sensing_service", return_value=service), \
         patch("millm.db.base.async_session_factory", return_value=session), \
         patch("millm.db.repositories.profile_repository.ProfileRepository") as Repo, \
         patch("millm.services.sae_service.AttachedSAEState") as State:
        Repo.return_value.get = AsyncMock(return_value=profile)
        State.return_value.entries.return_value = [first]
        State.return_value.attached_sae = first.sae
        await sensing_routes._toggle("prof_x", True)
    service.arm_for_profile.assert_not_called()


@pytest.mark.asyncio
async def test_a_quorum_override_rearms_the_clusters_own_sae():
    from millm.api.schemas.sensing import SensingConfigRequest

    first, second = _two_entries()
    profile = SimpleNamespace(
        id="prof_x", name="calm", source_kind="cluster", sae_id="sae-b", layer=12,
        is_active=True, sensing_enabled=True, cluster_meta={"members": [1, 2]},
    )
    service = MagicMock()
    service.armed_profile_id = "prof_x"
    service.build_config.return_value = MagicMock(
        thresholds=MagicMock(tolist=lambda: [0.1, 0.2]), min_k=None
    )
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.commit = AsyncMock()
    with patch.object(sensing_routes, "_sensing_service", return_value=service), \
         patch("millm.db.base.async_session_factory", return_value=session), \
         patch("millm.db.repositories.profile_repository.ProfileRepository") as Repo, \
         patch("millm.services.sae_service.AttachedSAEState") as State:
        Repo.return_value.get = AsyncMock(return_value=profile)
        State.return_value.entries.return_value = [first, second]
        State.return_value.attached_sae = first.sae
        await sensing_routes.set_sensing_config("prof_x", SensingConfigRequest(min_k=1))
    service.arm_for_profile.assert_called_once_with(profile, second.sae)
