"""What a probe's training was associated with, read off its definition.

⚠ A PROBE IS A BOUNDARY, AND THE TILE NAMED ONE SIDE OF IT. Until 2026-10-01 nothing in miLLM
said what a probe was fitted on at all: the tile read
`meta-llama/Llama-3.1-8B-Instruct · L11 · mean · dense residual · scope all`, which identifies the
READ POINT and says nothing about the concept. Two probes on this installation differed only in
their run, and the list could not tell them apart.

⚠ THE LABELS ARE THE CORPUS'S OWN STRINGS, NOT A VOCABULARY. `provenance.label_mapping` is
`{raw_value: "positive" | "negative" | "excluded"}` exactly as the operator entered it in miStudio
when building the training view — for the models-under-pressure corpus,
`{"<not behaviour>": "negative", "<behaviour>": "positive"}`. Nothing here translates, title-cases or
prettifies: a reader checking the tile against the corpus must find the same token. That is also
why this is not rendered server-side the way `rung_language` is — rung wording is a SHARED
vocabulary that must not drift between the two repos, and these strings are data.

⚠ THE DEFINITION WINS. `Probe.provenance` is a projection of `definition["provenance"]` plus an
`origin` key (`probe_service.py`), and the model's own docstring makes the document authoritative
when the two could disagree. This reads the document.
"""

from __future__ import annotations

from typing import Any


def label_mapping_of(definition: Any) -> dict[str, str]:
    """`{raw_label: side}` from a definition document, or `{}`.

    Defensive about shape on purpose: a definition arrives from a file or the Hub, and a
    malformed `provenance` must leave the tile blank rather than raise on a list route.
    """
    if not isinstance(definition, dict):
        return {}
    provenance = definition.get("provenance")
    if not isinstance(provenance, dict):
        return {}
    mapping = provenance.get("label_mapping")
    if not isinstance(mapping, dict):
        return {}
    return {str(k): str(v) for k, v in mapping.items()}


def concept_of(definition: Any) -> str | None:
    """The definition's own `concept` string, which miStudio writes from the same mapping.

    Carried through rather than recomputed: when the exporting build stated what positive
    means, that sentence is what the two repos should both show.
    """
    if not isinstance(definition, dict):
        return None
    concept = definition.get("concept")
    return concept if isinstance(concept, str) and concept.strip() else None


def behaviour_of(definition: Any) -> dict[str, Any] | None:
    """`{key, name, version}` from `definition.behaviour` (Feature 36, D18/D23), or `None`.

    The miStudio behaviour catalogue's additive optional object. Read from the document, so it is
    present even before the hand mirror learns the field. `None` when absent or malformed — the
    timeline then groups by the probe's own name.
    """
    if not isinstance(definition, dict):
        return None
    behaviour = definition.get("behaviour")
    if not isinstance(behaviour, dict):
        return None
    key = behaviour.get("key")
    if not isinstance(key, str) or not key.strip():
        return None
    name = behaviour.get("name")
    version = behaviour.get("version")
    return {
        "key": key[:64],
        "name": name if isinstance(name, str) and name.strip() else None,
        "version": version if isinstance(version, (int, str)) else None,
    }


def behaviour_version_of(definition: Any) -> int | None:
    """`definition.behaviour.version` as a positive integer, or `None` (039 S1).

    `None` means UNCATALOGUED (the producer's `version: null`, D25) or no behaviour at all; the
    selector treats an uncatalogued key's probes as one version of their own (OQ-39.3 (a)).
    Projected onto `probes.behaviour_version` at import, and by migration 034 for older rows.
    """
    behaviour = behaviour_of(definition)
    if behaviour is None:
        return None
    version = behaviour.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        return None
    return version


def source_probe_id_of(definition: Any) -> str | None:
    """`definition.provenance.probe_id`: the producer's (miStudio's) id for this probe, or None."""
    if not isinstance(definition, dict):
        return None
    provenance = definition.get("provenance")
    if not isinstance(provenance, dict):
        return None
    value = provenance.get("probe_id")
    return value if isinstance(value, str) and value else None
