"""The order every surface carries a request's results in, and each behaviour's `set` block
(039 FR-39.10, FR-39.4, TD39-30; FTID §3.4, I-3).

⚠ ONE SORT, APPLIED ONCE. `order_results` runs inside `_probe_publish_results`, BEFORE 038's
`set_probe_results`: the header, the terminal chunk, the body field, the Batch line and the stored
rows all read that one ordered list, and no surface sorts again (the header's own `(name, window)`
sort is gone). A behaviour's probe and rule entries are therefore contiguous everywhere, which is
what an application blending one behaviour's results needs.

`result_sort_key(r) = (behaviour key, or "~" so results with none sort last; kind; name; window)`,
where a rule result's name is `rule:<rule_set_id>.<signal>`.

The `set` block (038 OQ-38.3 (a), names per D31) is filled here, never in 038's builder (I-3):
`armed_members` / `total_members` from the registry and the in-memory member map
(`probe_selection`, no database read per request), `complete`, `missing_probe_ids`,
`members_digest` over the armed members' detector digests in probe-id order, and
`attached_rule_sets` (S2).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Optional, Sequence

#: Sorts results with no behaviour after every behaviour key (`~` is above every key character).
NO_BEHAVIOUR = "~"


def _name_of(result: Any) -> str:
    kind = getattr(result, "kind", "probe")
    if kind == "rule":
        return f"rule:{getattr(result, 'rule_set_id', '')}.{getattr(result, 'signal', '')}"
    legacy = getattr(result, "_legacy_name", None)
    name = legacy if legacy is not None else getattr(result, "probe_name", None)
    return str(name if name is not None else getattr(result, "probe_id", ""))


def result_sort_key(result: Any) -> tuple[str, str, str, str]:
    behaviour = getattr(result, "behaviour", None)
    key = getattr(behaviour, "key", None) if behaviour is not None else None
    return (key or NO_BEHAVIOUR, str(getattr(result, "kind", "probe")), _name_of(result),
            str(getattr(result, "window", "")))


def _get(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def members_digest(digests_by_id: dict[str, Optional[str]]) -> str:
    """SHA-256 over the armed members' detector digests, in probe-id order (FTDD §5.3)."""
    ordered = [[pid, digests_by_id[pid] or "unknown"] for pid in sorted(digests_by_id)]
    return hashlib.sha256(
        json.dumps(ordered, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_set_block(key: str, version: Optional[int], armed: Sequence[Any],
                    attachments: Iterable[Any] = ()) -> dict[str, Any]:
    """The `set` block for one behaviour: what is armed against what is imported.

    `armed` are the registry's `ArmedProbe`s of this behaviour (and version). `total_members` is
    the member map's count for the same (model, key, version) group; a probe the map does not know
    (imported after its last refresh) still counts as a member, so `total` is never below `armed`.
    """
    from millm.services import probe_selection

    armed_ids = sorted({p.probe_id for p in armed})
    members: set[str] = set(armed_ids)
    for probe_id in armed_ids:
        group = probe_selection.members_of_group(probe_id)
        if group:
            members.update(group)
    missing = sorted(members - set(armed_ids))
    digests = {p.probe_id: getattr(getattr(p, "descriptor", None), "detector_digest", None)
               for p in armed}
    return {
        "armed_members": len(armed_ids),
        "total_members": len(members),
        "complete": not missing,
        "missing_probe_ids": missing,
        "members_digest": members_digest(digests),
        "attached_rule_sets": [
            {"rule_set_id": _get(a, "rule_set_id"), "rule_set_digest": _get(a, "rule_set_digest")}
            for a in attachments
        ],
    }


def set_blocks(probes: Sequence[Any],
               attachments_for: Any = None) -> dict[tuple[str, Optional[int]], dict[str, Any]]:
    """`{(key, version): set block}` for every behaviour among `probes` (a request's armed set)."""
    grouped: dict[tuple[str, Optional[int]], list[Any]] = {}
    for probe in probes:
        descriptor = getattr(probe, "descriptor", None)
        key = getattr(descriptor, "behaviour_key", None)
        if not isinstance(key, str) or not key:
            continue
        grouped.setdefault((key, getattr(descriptor, "behaviour_version", None)), []).append(probe)
    return {
        group: build_set_block(group[0], group[1], members,
                               attachments_for(group[0]) if attachments_for else ())
        for group, members in grouped.items()
    }


def order_results(results: Sequence[Any], probes: Sequence[Any],
                  attachments_for: Any = None) -> list[Any]:
    """The request's results in contract order, each behaviour's carrying its `set` block.

    Never raises to a serving path: a failure here returns the results unchanged, logged, because
    a monitor's results in another order are better than none.
    """
    try:
        from millm.api.schemas.probe_result import SetBlock

        blocks = set_blocks(probes, attachments_for)
        out = []
        for result in sorted(results, key=result_sort_key):
            behaviour = getattr(result, "behaviour", None)
            group = (getattr(behaviour, "key", None), getattr(behaviour, "version", None))
            block = blocks.get(group) if behaviour is not None else None
            if block is None and behaviour is not None:
                # A behaviour result whose version the descriptor did not state as an int.
                block = next((b for (k, _v), b in blocks.items() if k == group[0]), None)
            if block is not None:
                updated = result.model_copy(update={"set_": SetBlock(**block)})
                updated._legacy_name = getattr(result, "_legacy_name", None)
                result = updated
            out.append(result)
        return out
    except Exception as exc:  # noqa: BLE001 - an observer never fails the request
        import logging
        import os

        if os.environ.get("MILLM_STRICT_RESULTS") == "1":
            raise
        logging.getLogger(__name__).warning("probe_result_order_failed error=%s",
                                            type(exc).__name__)
        return list(results)
