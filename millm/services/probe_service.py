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

MOVING A BAR IS NOT REPLACING A DETECTOR.
A probe definition carries two kinds of fact. The DETECTOR is everything that determines what
number the probe produces: `head.weights`, `bias`, `norm_mean`, `norm_std`, `attention_query`,
`read.layer`, `read.hook_point`, `scope`, `basis`, the `sae` block and its `feature_indices`,
`aggregation.rule` and its `params`, `model`, and the `evidence` that says what the number is
evidence of. The BAR is everything that determines only where that number is cut:
`decision.threshold`, `target_fpr`, `realised_fpr`, `threshold_source`, `calibration`,
`windows` and `length_bands`.

`replace` was refused because it replaces the first kind, and the objection above stands exactly
as written. It turns on a specific property of `probe_events`: the row records a `score` whose
MEANING comes from the detector, and nothing on the row records which detector produced it.
Change the weights and event #1's `score = 2.9` and event #900's `score = 2.9` are measurements
of different quantities under one id, with nothing to tell them apart.

A moved bar is not that, for one concrete reason: the event row ALREADY records the bar it was
judged against, per verdict, at judgement time — including the length-band override — and nothing
joins an event back to `probes.threshold`. After a re-cut, event #1 still says it was judged at
2.8786 and event #900 says 2.4011; both are true and both remain comparable, because the score
beneath each was produced by the same weights at the same layer under the same scope with the
same rule. The score is the measurement; the bar is the line drawn across it.

THE RULE: a probe's identity is everything that determines its SCORE; its bar is everything that
only determines the CUT. The first may never change in place under a probe id. The second may,
through `POST /api/probes/{probe_id}/recalibrate`, which is `extra="forbid"` and therefore
structurally incapable of carrying a detector, refuses any cut it cannot match to
`provenance.probe_id`, never stores the incoming object as the definition, and refuses a
threshold with no budget and no named source. `on_conflict` remains `rename|fail`.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from pydantic import ValidationError as PydanticValidationError

from millm.api.schemas.probe import KIND, ProbeDefinitionV1
from millm.core.config import settings
from millm.core.errors import ValidationError
from millm.core.probe_labels import behaviour_of, behaviour_version_of
from millm.db.models.probe import Probe
from millm.db.repositories.probe_repository import ProbeRepository

logger = logging.getLogger(__name__)

MAX_NAME_DEDUPE_ATTEMPTS = 50
VALID_ORIGINS = ("file", "hub", "mcp")


class ProbeService:
    """Import, list and delete probes. Arming lives in `ProbeRuntimeService` (phase 4)."""

    def __init__(self, repository: ProbeRepository) -> None:
        self.repository = repository

    async def import_definition(self, payload: dict[str, Any], **kwargs: Any) -> Probe:
        """`_import_definition` plus its lifecycle record (038 FR-38.20): `import` on success, the
        refusal's CODE otherwise. Never any text of the payload."""
        from millm import telemetry

        try:
            probe = await self._import_definition(payload, **kwargs)
        except Exception as exc:
            telemetry.lifecycle_refused(None, "import", exc)
            raise
        telemetry.emit_lifecycle(probe, "import")
        await self._refresh_members()
        return probe

    async def _refresh_members(self) -> None:
        """039: the per-key member map (`set` blocks read it with no database read per request)
        follows every import and delete. Never raises."""
        session = getattr(self.repository, "session", None)
        if session is None:
            return
        from millm.services import probe_selection

        await probe_selection.refresh_member_counts(session)

    async def _check_catalogue_limit(self) -> None:
        """039 FR-39.15: `PROBE_MAX_IMPORTED` refuses at limit + 1, before anything is written."""
        from millm.core.config import settings
        from millm.core.errors import CatalogueLimitError

        import inspect

        count = getattr(self.repository, "count", None)
        if count is None or not inspect.iscoroutinefunction(count):
            return  # a stand-in repository with no count (unit tests of other gates)
        limit = int(settings.PROBE_MAX_IMPORTED)
        if await count() >= limit:
            raise CatalogueLimitError(
                f"{limit} probes are imported, the most this deployment allows; delete one first",
                details={"limit": "PROBE_MAX_IMPORTED", "value": limit, "source": "global"},
            )

    async def _import_definition(
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
                # ⚠ `exc.json()`, not `exc.errors()` (12.5a F4): a validator that raises
                # `ValueError` puts the exception OBJECT in `ctx.error`, which the response
                # cannot serialise, so a refused import answered 500 instead of 422. pydantic's
                # own JSON writes it as its message.
                # ⚠ `include_input=False` (12.5a R1): a model-level validator's `input` is the
                # WHOLE document — its test vectors' `messages` text and its weights — and
                # these details are logged (`api_error`) and sent in the 422 body. The `loc` and
                # `msg` name what is wrong; the caller already holds what it sent.
                details={"errors": json.loads(exc.json(include_url=False,
                                                       include_input=False))[:20]},
            ) from exc

        # 038 FR-38.7: both identity digests, computed ONCE here from the raw document (parsed
        # values, so the stored JSONB form cannot change them) and never per request. `None` for a
        # definition carrying a number JCS cannot write.
        from millm.services.probe_digest import IDENTITY_ALGORITHM, digests_of

        detector_digest, decision_digest = digests_of(payload)
        # R-039 (D38.28/D38.29): a carried `identity` is VERIFIED before anything is written — the
        # producer's digests are stored only when miLLM reproduces them from the same document.
        identity = definition.identity
        if identity is not None:
            self._verify_identity(identity.model_dump(), detector_digest, decision_digest)

        await self._check_catalogue_limit()
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
            # Feature 36 (D18, D23): a projection of `definition.behaviour.key`, read from the RAW
            # document — the hand mirror may not know the field yet (it is `extra="allow"`, so it
            # is kept either way), and the definition is the source of truth.
            behaviour_key=behaviour_of(payload)["key"] if behaviour_of(payload) else None,
            # 039 S1: and its version (NULL = uncatalogued, a version of its own for the selector).
            behaviour_version=behaviour_version_of(payload),
            detector_digest=detector_digest,
            detector_digest_source=(
                "producer" if identity is not None
                else "computed" if detector_digest else "unknown"
            ),
            identity_algorithm=IDENTITY_ALGORITHM if detector_digest else None,
            producer_detector_digest=identity.detector_digest if identity is not None else None,
            decision_digest=decision_digest,
            # miStudio's revision of the bar, when it stated one; never miLLM's threshold_revision.
            decision_revision=identity.decision_revision if identity is not None else None,
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

    @staticmethod
    def _verify_identity(carried: dict[str, Any], detector: Optional[str],
                         decision: Optional[str]) -> None:
        """Raise the 409 for a carried identity miLLM cannot reproduce (import and hub import)."""
        from millm.core.errors import (
            ProbeIdentityAlgorithmUnknownError,
            ProbeIdentityMismatchError,
        )
        from millm.services.probe_digest import identity_refusal

        refusal = identity_refusal(carried, detector, decision)
        if refusal is None:
            return
        details = {k: v for k, v in refusal.items() if k != "refused"}
        if refusal["refused"] == "identity_algorithm_unknown":
            raise ProbeIdentityAlgorithmUnknownError(
                f"identity.algorithm {refusal['algorithm']!r} is not one this server can verify "
                f"({', '.join(refusal['supported'])})",
                details=details,
            )
        raise ProbeIdentityMismatchError(
            f"identity.{refusal['field']} is not this document's own: carried "
            f"{refusal['carried']}, recomputed {refusal['recomputed']}",
            details=details,
        )

    async def delete(self, probe_id: str) -> bool:
        """Delete a probe and its events. Returns False when it did not exist.

        ⚠ Refuses while armed. Deleting an armed probe would leave a hook installed against a row
        that no longer exists, and the next request would score into nothing.
        """
        probe = await self.repository.get(probe_id)
        if probe is None:
            return False
        from millm import telemetry

        if probe.armed:
            refusal = ValidationError(
                f"Probe '{probe.name}' is armed; disarm it before deleting",
                details={"probe_id": probe_id, "armed": True},
            )
            telemetry.lifecycle_refused(probe, "delete", refusal)
            raise refusal
        await self.repository.delete(probe)
        logger.info("probe_deleted id=%s", probe_id)
        telemetry.emit_lifecycle(probe, "delete")
        await self._refresh_members()
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
