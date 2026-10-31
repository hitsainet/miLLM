"""Which attached SAE a profile steers: ITS OWN, never "the first one attached" (028 FU-1).

⚠ THE DEFECT (S3-09, 028 FPRD Open Question 1, suite tracker 1.2). Both profile paths — global
activation in `profile_service` and the per-request override in
`InferenceService._apply_request_steering` — applied a profile to `AttachedSAEState().attached_sae`,
the FIRST registry entry, ignoring the `sae_id` and `layer` the profile records. With two SAEs
attached, a profile authored on the second steered the first: feature indices meant for one basis
written into another, silently (028 only made it visible, through the header's `profile` item and
a `profile_sae_mismatch` log line).

The operator's decision (028 FTASKS FU-1): fix BOTH paths together — select the entry by the
profile's own `sae_id`/`layer`, and REFUSE when it is not attached. One rule, used by both paths,
so they cannot diverge again:

* A profile that names an SAE (and/or a layer) steers the one attached entry matching it. None
  attached → `SAENotAttachedError` naming what it needs and what is attached. Several (an SAE
  attached at more than one layer, profile has no layer) → `InvalidParameterError`, ambiguous.
* A profile that names neither (an unbound cluster import, a hand-built profile) can only mean
  "the SAE that is attached": served when exactly ONE entry is attached, refused as ambiguous when
  several are. It is never resolved by insertion order.
* Nothing attached → `None`; each caller keeps its existing "no SAE attached" behaviour.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from millm.core.errors import InvalidParameterError, SAENotAttachedError


def _attached(entries: Sequence[Any]) -> list[dict[str, Any]]:
    return [{"sae_id": e.sae_id, "layer": e.layer} for e in entries]


def select_profile_entry(profile: Any, entries: Sequence[Any]) -> Optional[Any]:
    """The ONE attached entry `profile` steers, `None` when nothing is attached, or a refusal."""
    entries = list(entries)
    if not entries:
        return None
    name = getattr(profile, "name", None)
    sae_id = getattr(profile, "sae_id", None)
    layer = getattr(profile, "layer", None)

    matches = [
        e for e in entries
        if (sae_id is None or e.sae_id == sae_id) and (layer is None or e.layer == int(layer))
    ]
    if not matches:
        wants = f"SAE '{sae_id}'" if sae_id is not None else "an SAE"
        if layer is not None:
            wants += f" at layer {int(layer)}"
        raise SAENotAttachedError(
            f"Profile '{name}' steers {wants}, which is not attached. Attached: "
            + ", ".join(f"({e.sae_id}, layer {e.layer})" for e in entries)
            + ". Attach it, or use a profile authored for an attached SAE.",
            details={"profile": name, "profile_sae_id": sae_id, "profile_layer": layer,
                     "attached": _attached(entries)},
        )
    if len(matches) > 1:
        what = (f"'{sae_id}' is attached at {len(matches)} layers" if sae_id is not None
                else f"{len(matches)} SAEs are attached and the profile names none")
        raise InvalidParameterError(
            f"Profile '{name}' is ambiguous: {what} — "
            + ", ".join(f"({e.sae_id}, layer {e.layer})" for e in matches)
            + ". Record the profile's sae_id and layer, or detach the others.",
            details={"profile": name, "profile_sae_id": sae_id, "profile_layer": layer,
                     "attached": _attached(matches)},
        )
    return matches[0]


def profile_sae_or_none(profile: Any, entries: Sequence[Any]) -> Optional[Any]:
    """The profile's own attached SAE, or None when it is not attached or ambiguous.

    For paths that only arm or observe (sensing): they act on the profile's SAE or on nothing,
    never on the first entry.
    """
    try:
        entry = select_profile_entry(profile, entries)
    except (SAENotAttachedError, InvalidParameterError):
        return None
    return entry.sae if entry is not None else None
