"""
Hugging Face cluster-pack consumption (Feature 8, consume-only).

Browses public repos tagged with the community convention
(`mistudio-cluster-definition`), lists their definitions (manifest.jsonl
preferred, loose *.cluster.json fallback), and fetches single definition
files anonymously. All Hub calls run in a thread, behind a DEDICATED
cluster-hub circuit breaker (user-typed repo ids must never block model/SAE
downloads), with a bounded short-TTL listing cache.
"""

import asyncio
import json
import os
import time
from typing import Any

import structlog
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from millm.api.schemas.cluster import (
    MAX_IMPORT_BYTES,
    ClusterDefinitionV1,
    HubDefinitionRef,
    HubRepoInfo,
)
from millm.core.config import settings
from millm.core.errors import MiLLMError, ValidationError
from millm.core.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
)

_NOT_FOUND_ERRORS = (
    RepositoryNotFoundError,
    EntryNotFoundError,
    RevisionNotFoundError,
    GatedRepoError,
)

# Dedicated breaker: hub BROWSE failures must never open the shared
# huggingface circuit and block model/SAE downloads (review find). Not-found
# errors are EXCLUDED from failure counting at the breaker itself (round-2
# find: converting them in the caller was too late — the breaker had already
# recorded the failure), so typos genuinely don't count as service failures.
cluster_hub_circuit = CircuitBreaker(
    name="cluster_hub",
    config=CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        success_threshold=1,
        excluded_exceptions=_NOT_FOUND_ERRORS,
    ),
)

logger = structlog.get_logger()


class HubUnavailableError(MiLLMError):
    """Hub temporarily unreachable (circuit open or network failure)."""

    code = "HUB_UNAVAILABLE"
    status_code = 503


MAX_SEARCH_LIMIT = 50
MAX_LISTED_DEFINITIONS = 200
DEFINITION_SUFFIX = ".cluster.json"
MANIFEST_NAME = "manifest.jsonl"


@cluster_hub_circuit
def _list_models_sync(tag: str, query: str | None, base_model: str | None, limit: int):
    api = HfApi()
    filters = [tag]
    if base_model:
        filters.append(f"base_model:{base_model}")
    return list(api.list_models(filter=filters, search=query, limit=limit))


@cluster_hub_circuit
def _list_repo_files_sync(repo_id: str, revision: str | None) -> list[str]:
    api = HfApi()
    return list(api.list_repo_files(repo_id, revision=revision))


@cluster_hub_circuit
def _download_file_sync(repo_id: str, filename: str, revision: str | None,
                        cache_dir: str) -> str:
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        token=None,  # anonymous — public packs only
    )


class ClusterHubService:
    """Anonymous, read-only Hub access for cluster packs."""

    def __init__(self, cache_ttl_s: int | None = None) -> None:
        self._ttl = cache_ttl_s if cache_ttl_s is not None \
            else settings.CLUSTER_HUB_CACHE_TTL_S
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_dir = os.path.join(settings.SAE_CACHE_DIR, "clusters")

    # ── Browse ───────────────────────────────────────────────────────────

    async def search(
        self,
        query: str | None = None,
        base_model: str | None = None,
        limit: int = 30,
    ) -> list[HubRepoInfo]:
        """List repos tagged with the cluster-pack convention."""
        limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))
        key = f"search:{query}:{base_model}:{limit}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        try:
            models = await asyncio.to_thread(
                _list_models_sync, settings.CLUSTER_HUB_TAG, query, base_model, limit
            )
        except CircuitOpenError as e:
            raise HubUnavailableError(
                "Hugging Face is temporarily unreachable — retry shortly",
                details={"circuit": "cluster_hub"},
            ) from e
        result = [
            HubRepoInfo(
                repo_id=m.id,
                likes=getattr(m, "likes", 0) or 0,
                downloads=getattr(m, "downloads", 0) or 0,
                last_modified=getattr(m, "last_modified", None),
                tags=list(getattr(m, "tags", []) or []),
            )
            for m in models
        ]
        self._cache_put(key, result)
        return result

    async def list_definitions(
        self, repo_id: str, revision: str | None = None
    ) -> list[HubDefinitionRef]:
        """
        List a repo's definitions. Prefers manifest.jsonl (one JSON object per
        line: {file, name, member_count, base_model}); falls back to listing
        loose *.cluster.json files (capped).
        """
        key = f"defs:{repo_id}:{revision}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        try:
            files = await asyncio.to_thread(_list_repo_files_sync, repo_id, revision)
        except CircuitOpenError as e:
            raise HubUnavailableError(
                "Hugging Face is temporarily unreachable — retry shortly",
                details={"circuit": "cluster_hub"},
            ) from e
        except _NOT_FOUND_ERRORS as e:
            raise ValidationError(
                f"Hub repo not found or not accessible: {repo_id}",
                details={"repo_id": repo_id, "reason": type(e).__name__},
            ) from e

        refs: list[HubDefinitionRef]
        if MANIFEST_NAME in files:
            refs = await self._parse_manifest(repo_id, revision)
        else:
            refs = [
                HubDefinitionRef(file=f)
                for f in files
                if f.endswith(DEFINITION_SUFFIX)
            ][:MAX_LISTED_DEFINITIONS]

        self._cache_put(key, refs)
        return refs

    async def fetch_definition(
        self, repo_id: str, filename: str, revision: str | None = None
    ) -> tuple[ClusterDefinitionV1, dict[str, Any], dict[str, Any]]:
        """
        Download and validate ONE definition file.

        Returns (definition, raw_payload, hub_ref). The raw payload travels to
        storage so unknown additive fields survive re-export (lossless
        contract), exactly like file imports.
        """
        if not filename.endswith(DEFINITION_SUFFIX):
            raise ValidationError(
                f"Only {DEFINITION_SUFFIX} files can be imported from the Hub",
                details={"filename": filename},
            )
        if ".." in filename or filename.startswith("/"):
            raise ValidationError(
                "Invalid filename", details={"filename": filename}
            )

        try:
            path = await asyncio.to_thread(
                _download_file_sync, repo_id, filename, revision, self._cache_dir
            )
        except CircuitOpenError as e:
            raise HubUnavailableError(
                "Hugging Face is temporarily unreachable — retry shortly",
                details={"circuit": "cluster_hub"},
            ) from e
        except _NOT_FOUND_ERRORS as e:
            raise ValidationError(
                f"Definition not found on the Hub: {repo_id}/{filename}",
                details={"repo_id": repo_id, "filename": filename,
                         "reason": type(e).__name__},
            ) from e
        size = os.path.getsize(path)
        if size > MAX_IMPORT_BYTES:
            raise ValidationError(
                f"Definition file exceeds {MAX_IMPORT_BYTES} bytes",
                details={"filename": filename, "size": size},
            )
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        definition = ClusterDefinitionV1.model_validate(payload)
        hub_ref = {"repo_id": repo_id, "revision": revision or "main",
                   "path": filename}
        logger.info("cluster_hub_fetched", repo_id=repo_id, filename=filename)
        return definition, payload, hub_ref

    # ── Internals ────────────────────────────────────────────────────────

    async def _parse_manifest(
        self, repo_id: str, revision: str | None
    ) -> list[HubDefinitionRef]:
        path = await asyncio.to_thread(
            _download_file_sync, repo_id, MANIFEST_NAME, revision, self._cache_dir
        )
        refs: list[HubDefinitionRef] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if (isinstance(row, dict)
                            and isinstance(row.get("file"), str)
                            and row["file"].endswith(DEFINITION_SUFFIX)):
                        refs.append(HubDefinitionRef(
                            file=row["file"],
                            name=row.get("name"),
                            member_count=row.get("member_count"),
                            base_model=row.get("base_model"),
                        ))
                except (json.JSONDecodeError, TypeError):
                    continue  # skip malformed manifest lines, keep the rest
                if len(refs) >= MAX_LISTED_DEFINITIONS:
                    break
        return refs

    def _cache_get(self, key: str) -> Any | None:
        hit = self._cache.get(key)
        if hit and (time.monotonic() - hit[0]) < self._ttl:
            return hit[1]
        return None

    MAX_CACHE_ENTRIES = 64

    def _cache_put(self, key: str, value: Any) -> None:
        now = time.monotonic()
        # Evict expired entries; the cache must not grow for the process
        # lifetime under varied search strings (review find).
        expired = [k for k, (ts, _) in self._cache.items() if now - ts >= self._ttl]
        for k in expired:
            del self._cache[k]
        if len(self._cache) >= self.MAX_CACHE_ENTRIES:
            oldest = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest]
        self._cache[key] = (now, value)
