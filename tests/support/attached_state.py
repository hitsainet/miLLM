"""Shared test double for the AttachedSAEState singleton.

Three test modules independently reimplemented this stub, so a change to the
registry surface (Feature 12's multi-SAE generalisation) broke all three at
once. One definition, imported everywhere, keeps the double honest: it mirrors
the real ``by_layer`` / ``count`` / ``entries`` / ``get`` semantics, including
``by_layer`` returning None when a layer is ambiguous.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class FakeAttachedState:
    """Stands in for ``AttachedSAEState``.

    Args:
        d_sae: feature width of the primary attached SAE.
        sae_id / layer: identity of the primary attached SAE.
        attached: when False, nothing is attached at all.
        extra_entries: iterable of ``(sae_id, layer, d_sae)`` for multi-SAE
            cases (Feature 12 circuits attach one SAE per referenced layer).
    """

    def __init__(
        self,
        d_sae: int = 16384,
        sae_id: str = "sae_local",
        layer: int = 12,
        attached: bool = True,
        extra_entries=(),
    ) -> None:
        self.attached_sae = MagicMock(d_sae=d_sae) if attached else None
        self.attached_sae_id = sae_id if attached else None
        self.attached_layer = layer if attached else None
        self._entries = []
        if attached:
            self._entries.append(
                SimpleNamespace(sae=self.attached_sae, sae_id=sae_id, layer=layer)
            )
        for extra_sae_id, extra_layer, extra_d_sae in extra_entries:
            self._entries.append(
                SimpleNamespace(
                    sae=MagicMock(d_sae=extra_d_sae),
                    sae_id=extra_sae_id,
                    layer=extra_layer,
                )
            )

    @property
    def is_attached(self) -> bool:
        return bool(self._entries)

    @property
    def count(self) -> int:
        return len(self._entries)

    def entries(self):
        return list(self._entries)

    def by_layer(self, layer):
        """Unique entry on ``layer``; None when absent OR ambiguous."""
        matches = [e for e in self._entries if e.layer == int(layer)]
        return matches[0] if len(matches) == 1 else None

    def get(self, sae_id, layer):
        for e in self._entries:
            if e.sae_id == sae_id and e.layer == int(layer):
                return e
        return None


def patched_state(**kwargs):
    """Patch the singleton at BOTH import sites: ClusterService (compat
    assessment) and SAEService (the shared activation gate)."""

    @contextlib.contextmanager
    def _both():
        state = FakeAttachedState(**kwargs)
        with patch(
            "millm.services.cluster_service.AttachedSAEState", return_value=state
        ), patch("millm.services.sae_service.AttachedSAEState", return_value=state):
            yield state

    return _both()
