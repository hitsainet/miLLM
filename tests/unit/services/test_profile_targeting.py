"""The one rule for which attached SAE a profile steers (028 FU-1, suite tracker 1.2).

Fixtures hold TWO attached entries on purpose: with one, "the profile's own SAE" and "the first
SAE" are the same entry, and a test passes against the defect it exists to catch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from millm.core.errors import InvalidParameterError, SAENotAttachedError
from millm.services.profile_targeting import select_profile_entry


def _entry(sae_id: str, layer: int) -> SimpleNamespace:
    return SimpleNamespace(sae_id=sae_id, layer=layer, sae=SimpleNamespace(d_sae=16))


def _profile(sae_id=None, layer=None, name="p") -> SimpleNamespace:
    return SimpleNamespace(name=name, sae_id=sae_id, layer=layer)


FIRST = _entry("sae-a", 10)
SECOND = _entry("sae-b", 12)
BOTH = [FIRST, SECOND]


class TestABoundProfileSteersItsOwnSAE:
    def test_the_second_entry_when_the_profile_names_it(self):
        assert select_profile_entry(_profile("sae-b", 12), BOTH) is SECOND

    def test_by_sae_id_alone(self):
        assert select_profile_entry(_profile("sae-b"), BOTH) is SECOND

    def test_by_layer_alone(self):
        assert select_profile_entry(_profile(layer=12), BOTH) is SECOND

    def test_its_sae_not_attached_refuses_naming_both_sides(self):
        with pytest.raises(SAENotAttachedError) as err:
            select_profile_entry(_profile("sae-c", 3, name="humor"), BOTH)
        assert "humor" in err.value.message and "sae-c" in err.value.message
        assert err.value.details["attached"] == [
            {"sae_id": "sae-a", "layer": 10}, {"sae_id": "sae-b", "layer": 12},
        ]

    def test_right_sae_wrong_layer_refuses(self):
        with pytest.raises(SAENotAttachedError):
            select_profile_entry(_profile("sae-b", 10), BOTH)

    def test_one_sae_at_two_layers_and_no_layer_recorded_is_ambiguous(self):
        with pytest.raises(InvalidParameterError):
            select_profile_entry(_profile("sae-a"), [FIRST, _entry("sae-a", 11)])


class TestAnUnboundProfile:
    def test_the_only_attached_entry(self):
        assert select_profile_entry(_profile(), [SECOND]) is SECOND

    def test_several_attached_is_refused_never_the_first(self):
        with pytest.raises(InvalidParameterError) as err:
            select_profile_entry(_profile(), BOTH)
        assert len(err.value.details["attached"]) == 2


def test_nothing_attached_is_none_for_the_caller_to_handle():
    assert select_profile_entry(_profile("sae-a", 10), []) is None


class TestThePerRequestPathSteersTheProfilesOwnSAE:
    """`_apply_request_steering` with TWO real-shaped entries: the profile is bound to the second.
    Before FU-1 it wrote into the first entry's SAE (`attached_sae`)."""

    @staticmethod
    def _sae(values=None):
        from unittest.mock import MagicMock

        state = {"values": dict(values or {}), "enabled": False}
        sae = MagicMock()
        sae.d_sae = 16384
        sae.get_steering_values.side_effect = lambda: dict(state["values"])
        sae.set_steering_batch.side_effect = lambda v: state["values"].update(v)
        sae.clear_steering.side_effect = lambda: state.update(values={})
        sae.enable_steering.side_effect = lambda e: state.update(enabled=e)
        type(sae).is_steering_enabled = property(lambda self: state["enabled"])
        return sae, state

    async def _apply(self, profile, entries):
        from unittest.mock import AsyncMock, MagicMock, patch

        from millm.services.inference_service import InferenceService

        service = InferenceService(model_service=MagicMock())
        with patch("millm.services.sae_service.AttachedSAEState") as MockState, \
             patch("millm.db.base.async_session_factory"), \
             patch("millm.db.repositories.profile_repository.ProfileRepository") as MockRepo:
            MockState.return_value.entries.return_value = entries
            MockState.return_value.attached_sae = entries[0].sae
            MockState.return_value.steering_epoch = 0
            MockState.return_value.get.side_effect = lambda sid, lay: next(
                (e for e in entries if e.sae_id == sid and e.layer == int(lay)), None)
            MockRepo.return_value.get_by_name = AsyncMock(return_value=profile)
            saved = await service._apply_request_steering(profile.name, None)
            service._restore_request_profile(saved)
            return saved

    async def test_applies_to_and_restores_the_second_entry(self):
        first_sae, first = self._sae({1: 9.0})
        second_sae, second = self._sae({2: 4.0})
        entries = [SimpleNamespace(sae=first_sae, sae_id="sae-a", layer=10),
                   SimpleNamespace(sae=second_sae, sae_id="sae-b", layer=12)]
        profile = SimpleNamespace(name="calm", sae_id="sae-b", layer=12, steering={"7": 3.0},
                                  intensity=1.0, source_kind="manual", cluster_meta=None)
        saved = await self._apply(profile, entries)

        assert (saved["sae_id"], saved["layer"]) == ("sae-b", 12)
        # Applied on the second entry, then its pre-request values restored there.
        assert [c.args[0] for c in second_sae.set_steering_batch.call_args_list] == [
            {7: 3.0}, {2: 4.0},
        ]
        first_sae.set_steering_batch.assert_not_called()
        first_sae.clear_steering.assert_not_called()
        # Restored on the SAME entry: the second SAE is back to its pre-request values.
        assert second["values"] == {2: 4.0}
        assert first["values"] == {1: 9.0}

    async def test_its_sae_not_attached_refuses_and_touches_nothing(self):
        first_sae, first = self._sae({1: 9.0})
        entries = [SimpleNamespace(sae=first_sae, sae_id="sae-a", layer=10)]
        profile = SimpleNamespace(name="calm", sae_id="sae-b", layer=12, steering={"7": 3.0},
                                  intensity=1.0, source_kind="manual", cluster_meta=None)
        with pytest.raises(SAENotAttachedError):
            await self._apply(profile, entries)
        first_sae.set_steering_batch.assert_not_called()
        assert first["values"] == {1: 9.0}


class TestTheRealSAEServiceMethods:
    """The profile/cluster tests mock SAEService; these run its two FU-1 methods for real."""

    def _svc(self, entries):
        from unittest.mock import MagicMock

        from millm.services.sae_service import SAEService

        state = MagicMock()
        state.entries.return_value = entries
        return SimpleNamespace(
            _sae_state=state,
            _check_feature_idx=lambda idx, sae: SAEService._check_feature_idx(None, idx, sae),
        ), SAEService

    def test_profile_target_reads_the_registry_not_the_first_entry(self):
        from unittest.mock import MagicMock

        first = SimpleNamespace(sae=MagicMock(d_sae=16), sae_id="sae-a", layer=10)
        second = SimpleNamespace(sae=MagicMock(d_sae=16), sae_id="sae-b", layer=12)
        fake, SAEService = self._svc([first, second])
        assert SAEService.profile_target(fake, _profile("sae-b", 12)) is second

    def test_steer_entry_writes_only_that_entry_and_validates_against_it(self):
        from unittest.mock import MagicMock

        from millm.core.errors import InvalidFeatureIndexError

        entry = SimpleNamespace(sae=MagicMock(d_sae=16), sae_id="sae-b", layer=12)
        fake, SAEService = self._svc([entry])
        SAEService.steer_entry(fake, entry, {3: 1.5})
        entry.sae.clear_steering.assert_called_once_with()
        entry.sae.set_steering_batch.assert_called_once_with({3: 1.5})
        entry.sae.enable_steering.assert_called_once_with(True)
        fake._sae_state.bump_steering_epoch.assert_called_once()
        with pytest.raises(InvalidFeatureIndexError):
            SAEService.steer_entry(fake, entry, {16: 1.0})  # out of THIS SAE's range
        SAEService.steer_entry(fake, entry, None)
        entry.sae.enable_steering.assert_called_with(False)
