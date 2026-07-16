"""
Hugging Face cluster-pack consumption (Feature 8, consume-only).

Browses public repos tagged with the community convention
(`mistudio-cluster-definition`), lists their definitions (manifest.jsonl
preferred, loose *.cluster.json fallback), and fetches single definition
files anonymously. All Hub calls run in a thread, behind the shared
huggingface circuit breaker, with a short-TTL listing cache.
"""

import asyncio
import json
import os
import time
from typing import Any

import structlog
from huggingface_hub import HfApi, hf_hub_download

from millm.api.schemas.cluster import (
    MAX_IMPORT_BYTES,
    ClusterDefinitionV1,
    HubDefinitionRef,
    HubRepoInfo,
)
from millm.core.config import settings
from millm.core.errors import ValidationError
from millm.core.resilience import huggingface_circuit

logger = structlog.get_logger()

MAX_SEARCH_LIMIT = 50
MAX_LISTED_DEFINITIONS = 200
DEFINITION_SUFFIX = ".cluster.json"
MANIFEST_NAME = "manifest.jsonl"


@huggingface_circuit
def _list_models_sync(tag: str, query: str | None, base_model: str | None, limit: int):
    api = HfApi()
    filters = [tag]
    if base_model:
        filters.append(f"base_model:{base_model}")
    return list(api.list_models(filter=filters, search=query, limit=limit))


@huggingface_circuit
def _list_repo_files_sync(repo_id: str, revision: str | None) -> list[str]:
    api = HfApi()
    return list(api.list_repo_files(repo_id, revision=revision))


@huggingface_circuit
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

        models = await asyncio.to_thread(
            _list_models_sync, settings.CLUSTER_HUB_TAG, query, base_model, limit
        )
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

        files = await asyncio.to_thread(_list_repo_files_sync, repo_id, revision)

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
    ) -> tuple[ClusterDefinitionV1, dict[str, Any]]:
        """
        Download and validate ONE definition file.

        Returns the validated definition plus a hub_ref provenance dict
        {repo_id, revision, path}.
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

        path = await asyncio.to_thread(
            _download_file_sync, repo_id, filename, revision, self._cache_dir
        )
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
        return definition, hub_ref

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
                    if isinstance(row, dict) and row.get("file", "").endswith(
                        DEFINITION_SUFFIX
                    ):
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

    def _cache_put(self, key: str, value: Any) -> None:
        self._cache[key] = (time.monotonic(), value)
