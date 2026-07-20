"""Unit tests for the multi-SAE AttachedSAEState registry (Feature 12).

Feature 12 generalizes the single-SAE ``AttachedSAEState`` to a registry keyed
by ``(sae_id, layer)`` so a cross-layer circuit can attach one SAE per layer.
These tests pin the registry contract: per-key set/clear, idempotent re-attach
(orphaned-hook removal), ``by_layer`` uniqueness, back-compat singular
properties (first entry), and the plural status accessor.
"""

from unittest.mock import MagicMock

import pytest

from millm.services.sae_service import (
    AttachedEntry,
    AttachedSAEState,
    AttachmentStatusSet,
)


@pytest.fixture(autouse=True)
def reset_registry():
    state = AttachedSAEState()
    state._entries.clear()
    yield
    state._entries.clear()


def _sae(mb: float = 64.0):
    sae = MagicMock()
    sae.estimate_memory_mb.return_value = mb
    sae.is_steering_enabled = False
    sae.is_monitoring_enabled = False
    sae.steering_apply_count = 0
    return sae


class TestRegistrySetClear:
    def test_set_creates_entry(self):
        state = AttachedSAEState()
        h = MagicMock()
        state.set(_sae(), "sae-a", 10, h)
        assert state.count == 1
        entry = state.get("sae-a", 10)
        assert isinstance(entry, AttachedEntry)
        assert entry.sae_id == "sae-a" and entry.layer == 10 and entry.hook_handle is h

    def test_layer_coerced_to_int(self):
        state = AttachedSAEState()
        state.set(_sae(), "sae-a", 10, MagicMock())
        # set/get/by_layer all coerce the layer to int, so an int-like str
        # resolves to the same stored entry (no accidental duplicate keys).
        assert state.get("sae-a", 10) is not None
        assert state.get("sae-a", "10") is not None  # coerced, same entry
        assert state.by_layer("10").sae_id == "sae-a"
        assert list(state._entries.keys()) == [("sae-a", 10)]  # stored as int

    def test_reattach_same_key_removes_old_hook(self):
        state = AttachedSAEState()
        old, new = MagicMock(), MagicMock()
        state.set(_sae(), "sae-a", 10, old)
        state.set(_sae(), "sae-a", 10, new)
        old.remove.assert_called_once()
        assert state.count == 1
        assert state.get("sae-a", 10).hook_handle is new

    def test_different_keys_coexist(self):
        state = AttachedSAEState()
        h1, h2 = MagicMock(), MagicMock()
        state.set(_sae(), "sae-a", 10, h1)
        state.set(_sae(), "sae-b", 13, h2)
        assert state.count == 2
        h1.remove.assert_not_called()

    def test_clear_all(self):
        state = AttachedSAEState()
        h1, h2 = MagicMock(), MagicMock()
        state.set(_sae(), "sae-a", 10, h1)
        state.set(_sae(), "sae-b", 13, h2)
        state.clear()
        assert state.count == 0 and not state.is_attached
        h1.remove.assert_called_once()
        h2.remove.assert_called_once()

    def test_clear_one_by_key(self):
        state = AttachedSAEState()
        h1, h2 = MagicMock(), MagicMock()
        state.set(_sae(), "sae-a", 10, h1)
        state.set(_sae(), "sae-b", 13, h2)
        state.clear(sae_id="sae-a", layer=10)
        assert state.count == 1
        assert state.get("sae-b", 13) is not None
        h1.remove.assert_called_once()
        h2.remove.assert_not_called()

    def test_clear_by_sae_id_only(self):
        """A sae_id attached on two layers: clearing by sae_id removes both."""
        state = AttachedSAEState()
        h1, h2, h3 = MagicMock(), MagicMock(), MagicMock()
        state.set(_sae(), "sae-a", 10, h1)
        state.set(_sae(), "sae-a", 11, h2)
        state.set(_sae(), "sae-b", 13, h3)
        state.clear(sae_id="sae-a")
        assert state.count == 1 and state.get("sae-b", 13) is not None
        h1.remove.assert_called_once()
        h2.remove.assert_called_once()

    def test_clear_swallows_hook_remove_error(self):
        state = AttachedSAEState()
        bad = MagicMock()
        bad.remove.side_effect = RuntimeError("boom")
        state.set(_sae(), "sae-a", 10, bad)
        state.clear()  # must not raise
        assert state.count == 0


class TestByLayerUniqueness:
    def test_unique_layer_returns_entry(self):
        state = AttachedSAEState()
        state.set(_sae(), "sae-a", 10, MagicMock())
        assert state.by_layer(10).sae_id == "sae-a"

    def test_absent_layer_returns_none(self):
        state = AttachedSAEState()
        state.set(_sae(), "sae-a", 10, MagicMock())
        assert state.by_layer(11) is None

    def test_ambiguous_layer_returns_none(self):
        """Two SAEs on the SAME layer → by_layer refuses to guess (returns
        None) so a caller never silently picks the wrong basis."""
        state = AttachedSAEState()
        state.set(_sae(), "sae-a", 10, MagicMock())
        state.set(_sae(), "sae-b", 10, MagicMock())
        assert state.by_layer(10) is None


class TestBackCompatSingular:
    def test_singular_props_reflect_first_entry(self):
        state = AttachedSAEState()
        s1 = _sae()
        state.set(s1, "sae-a", 10, MagicMock())
        state.set(_sae(), "sae-b", 13, MagicMock())
        assert state.attached_sae is s1
        assert state.attached_sae_id == "sae-a"
        assert state.attached_layer == 10
        assert state.is_attached is True

    def test_singular_props_none_when_empty(self):
        state = AttachedSAEState()
        assert state.attached_sae is None
        assert state.attached_sae_id is None
        assert state.attached_layer is None
        assert state.is_attached is False


class TestPluralStatus:
    def test_status_set_sums_memory_and_lists_entries(self):
        from millm.services.sae_service import SAEService

        state = AttachedSAEState()
        state.set(_sae(64.0), "sae-a", 10, MagicMock())
        state.set(_sae(64.0), "sae-b", 13, MagicMock())
        # get_attachment_status_set only touches the singleton, not the repo.
        svc = SAEService.__new__(SAEService)
        svc._sae_state = state
        status = svc.get_attachment_status_set()
        assert isinstance(status, AttachmentStatusSet)
        assert status.is_attached and status.count == 2
        assert status.total_memory_usage_mb == 128
        assert {e.sae_id for e in status.entries} == {"sae-a", "sae-b"}

    def test_status_set_empty(self):
        from millm.services.sae_service import SAEService

        svc = SAEService.__new__(SAEService)
        svc._sae_state = AttachedSAEState()
        status = svc.get_attachment_status_set()
        assert not status.is_attached and status.count == 0
        assert status.entries == [] and status.total_memory_usage_mb is None
