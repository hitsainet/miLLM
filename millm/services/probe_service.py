"""Importing and storing probe definitions (Feature 24, FR-24.1).

The RAW document is stored verbatim in `Probe.definition` so a re-export is lossless: pydantic's
parsed form would drop additive fields a newer miStudio emits, and a document that loses fields on
a round trip is not the document that was published.

The typed columns beside it are a PROJECTION for queries and the arm-time identity check. When they
could disagree, the definition wins — nothing here writes a projected column without writing the
definition in the same call.

⚠ `on_conflict` is **`rename|fail`**, matching circuits and clusters. The FPRD said
`rename|replace|refuse`, which matched nothing in this repository. `refuse` is `fail` renamed, and
`replace` was rejected on merit: overwriting a definition in place while that probe is ARMED would
change the detector underneath a running monitor, while every event before and after kept the same
`probe_id` — so the event history would describe two different detectors as one. Re-importing a
rebuilt definition goes disarm → delete → import.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from millm.api.schemas.probe import KIND, ProbeDefinitionV1
from millm.core.config import settings
from millm.core.errors import ValidationError
from millm.db.models.probe import Probe
from millm.db.repositories.probe_repository import ProbeRepository

logger = logging.getLogger(__name__)

MAX_NAME_DEDUPE_ATTEMPTS = 50
VALID_ORIGINS = ("file", "hub", "mcp")


class ProbeService:
    """Import, list and delete probes. Arming lives in `ProbeRuntimeService` (phase 4)."""

    def __init__(self, repository: ProbeRepository) -> None:
        self.repository = repository

    async def import_definition(
        self,
        payload: dict[str, Any],
        *,
        raw_bytes: int | None = None,
        on_conflict: str = "rename",
        origin: str = "file",
    ) -> Probe:
        """Validate and store a probe definition.

        Gates run in a deliberate order: size before parsing (a hostile payload should not be
        walked at all), kind before schema (so a circuit definition posted here gets
        `UNKNOWN_KIND` rather than forty confusing field errors), then the full contract.
        """
        if raw_bytes is not None and raw_bytes > settings.PROBE_MAX_IMPORT_BYTES:
            raise ValidationError(
                f"Probe definition exceeds the {settings.PROBE_MAX_IMPORT_BYTES} byte cap",
                details={
                    "bytes": raw_bytes,
                    "max_bytes": settings.PROBE_MAX_IMPORT_BYTES,
                    "code": "PAYLOAD_TOO_LARGE",
                },
            )

        if origin not in VALID_ORIGINS:
            raise ValidationError(
                f"Unknown origin {origin!r} — expected one of {', '.join(VALID_ORIGINS)}",
                details={"origin": origin},
            )

        kind = (payload or {}).get("kind")
        if kind != KIND:
            # Checked before the schema so posting a circuit definition here says what is wrong
            # rather than producing a wall of field errors about a document of another shape.
            raise ValidationError(
                f"Unknown kind {kind!r} — expected {KIND!r}",
                details={"kind": kind, "code": "UNKNOWN_KIND"},
            )

        try:
            definition = ProbeDefinitionV1.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError(
                "Probe definition does not conform to mistudio.probe-definition/v1",
                details={"errors": exc.errors(include_url=False)[:20]},
            ) from exc

        name = await self._dedupe_name(definition.name, on_conflict)

        probe = await self.repository.create(
            id=f"pr_{uuid.uuid4().hex[:12]}",
            name=name,
            definition=payload,  # RAW document — lossless re-export
            hf_id=definition.model.hf_id,
            revision=definition.model.revision,
            d_model=definition.model.d_model,
            n_layers=definition.model.n_layers,
            template_sha256=definition.model.chat_template_sha256,
            layer=definition.read.layer,
            rule=definition.aggregation.rule,
            streamable=definition.aggregation.streamable,
            scope=definition.scope,
            basis=definition.basis,
            sae_ref=payload.get("sae"),
            threshold=definition.decision.threshold,
            target_fpr=definition.decision.target_fpr,
            rung=definition.evidence.rung,
            definition_acknowledgement=(
                definition.evidence.acknowledgement.model_dump()
                if definition.evidence.acknowledgement is not None
                else None
            ),
            provenance={**(payload.get("provenance") or {}), "origin": origin},
            armed=False,
        )
        logger.info(
            "probe_imported id=%s name=%s model=%s layer=%s rule=%s rung=%s origin=%s",
            probe.id,
            probe.name,
            probe.hf_id,
            probe.layer,
            probe.rule,
            probe.rung,
            origin,
        )
        return probe

    async def delete(self, probe_id: str) -> bool:
        """Delete a probe and its events. Returns False when it did not exist.

        ⚠ Refuses while armed. Deleting an armed probe would leave a hook installed against a row
        that no longer exists, and the next request would score into nothing.
        """
        probe = await self.repository.get(probe_id)
        if probe is None:
            return False
        if probe.armed:
            raise ValidationError(
                f"Probe '{probe.name}' is armed; disarm it before deleting",
                details={"probe_id": probe_id, "armed": True},
            )
        await self.repository.delete(probe)
        logger.info("probe_deleted id=%s", probe_id)
        return True

    async def _dedupe_name(self, name: str, on_conflict: str) -> str:
        if on_conflict not in ("rename", "fail"):
            raise ValidationError(
                f"Unknown on_conflict {on_conflict!r} — expected 'rename' or 'fail'",
                details={"on_conflict": on_conflict},
            )
        existing = await self.repository.get_by_name(name)
        if existing is None:
            return name
        if on_conflict == "fail":
            raise ValidationError(
                f"A probe named '{name}' already exists", details={"name": name}
            )
        for n in range(2, MAX_NAME_DEDUPE_ATTEMPTS + 2):
            candidate = f"{name} ({n})"
            if await self.repository.get_by_name(candidate) is None:
                return candidate
        raise ValidationError(
            f"Could not find a free name for '{name}' after {MAX_NAME_DEDUPE_ATTEMPTS} attempts",
            details={"name": name},
        )
