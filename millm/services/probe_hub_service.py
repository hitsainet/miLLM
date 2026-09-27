"""Browsing and fetching probe definitions from HuggingFace (Feature 24, FR-24.2).

Everything interesting is inherited. `ClusterHubService` already carries the anonymous read-only
Hub access, the TTL cache, the circuit breaker that turns a flapping Hub into `HUB_UNAVAILABLE`
rather than a stalled request, the manifest-first listing with a glob fallback, and the traversal
guards on filenames. None of that is cluster-specific — only the tag, the suffix, the cache
directory and the contract model are, and those are the four knobs this subclass sets.

Copying the breaker and cache instead would have been a second implementation of the part most
likely to be subtly wrong, and the two would drift the first time either was fixed.
"""

from __future__ import annotations

from typing import Any

from millm.api.schemas.probe import ProbeDefinitionV1
from millm.services.cluster_hub_service import ClusterHubService


class ProbeHubService(ClusterHubService):
    """Anonymous, read-only Hub access for probe definitions."""

    HUB_TAG_SETTING = "PROBE_HUB_TAG"
    DEFINITION_SUFFIX = ".probe.json"
    CACHE_SUBDIR = "probes"
    DEFINITION_MODEL: Any = ProbeDefinitionV1
    ARTIFACT_NAME = "probe"

    def __init__(self, cache_ttl_s: int | None = None) -> None:
        from millm.core.config import settings

        super().__init__(
            cache_ttl_s if cache_ttl_s is not None else settings.PROBE_HUB_CACHE_TTL_S
        )
