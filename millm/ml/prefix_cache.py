"""
Prefix caching for repeated system prompts.

Caches the KV states (past_key_values) for system prompt prefixes
so that subsequent requests with the same system prompt can skip
re-computing those tokens. This yields significant speedups when
the same system prompt is used repeatedly (common with OpenAI API).

Cache invalidation:
- When SAE steering values change (steering delta affects hidden states)
- When a different model is loaded
- LRU eviction when max_entries is exceeded

Steering compatibility:
- Cache entries are tagged with the current steering delta hash
- If steering changes, cached prefixes are automatically invalidated
"""

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached prefix KV state."""

    prompt_hash: str
    steering_hash: str
    past_key_values: Any  # DynamicCache or tuple of tuples
    prompt_token_count: int
    hit_count: int = 0


class PrefixCache:
    """
    LRU cache for system prompt KV states.

    Thread safety: This cache is accessed only within the inference
    service's request queue (semaphore-guarded), so concurrent access
    is not an issue.

    Usage:
        cache = PrefixCache(max_entries=5)

        # Check for cached prefix
        entry = cache.get(system_prompt, steering_hash)
        if entry:
            # Use cached past_key_values
            outputs = model.generate(past_key_values=entry.past_key_values, ...)
        else:
            # Compute prefix and cache it
            prefix_kv = model(system_tokens, use_cache=True).past_key_values
            cache.put(system_prompt, steering_hash, prefix_kv, token_count)
    """

    def __init__(self, max_entries: int = 5, enabled: bool = True) -> None:
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._enabled = enabled
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "size": len(self._cache),
            "max_entries": self._max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(1, self._hits + self._misses) * 100, 1),
        }

    def get(self, prompt_text: str, steering_hash: str = "") -> Optional[CacheEntry]:
        """
        Look up a cached prefix.

        Args:
            prompt_text: The system prompt text to look up.
            steering_hash: Hash of current steering state.

        Returns:
            CacheEntry if found and valid, None otherwise.
        """
        if not self._enabled:
            return None

        key = self._make_key(prompt_text, steering_hash)

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry = self._cache[key]
            entry.hit_count += 1
            self._hits += 1
            logger.debug(
                "prefix_cache_hit",
                prompt_tokens=entry.prompt_token_count,
                hit_count=entry.hit_count,
            )
            return entry

        self._misses += 1
        return None

    def put(
        self,
        prompt_text: str,
        steering_hash: str,
        past_key_values: Any,
        prompt_token_count: int,
    ) -> None:
        """
        Store a prefix KV state in the cache.

        Args:
            prompt_text: The system prompt text.
            steering_hash: Hash of current steering state.
            past_key_values: The computed KV cache to store.
            prompt_token_count: Number of tokens in the prefix.
        """
        if not self._enabled:
            return

        key = self._make_key(prompt_text, steering_hash)

        # Evict oldest if at capacity
        while len(self._cache) >= self._max_entries:
            evicted_key, evicted_entry = self._cache.popitem(last=False)
            # Free GPU memory from evicted entry
            del evicted_entry.past_key_values
            logger.debug("prefix_cache_evicted", key=evicted_key[:16])

        self._cache[key] = CacheEntry(
            prompt_hash=self._hash_text(prompt_text),
            steering_hash=steering_hash,
            past_key_values=past_key_values,
            prompt_token_count=prompt_token_count,
        )
        logger.debug(
            "prefix_cache_stored",
            prompt_tokens=prompt_token_count,
            cache_size=len(self._cache),
        )

    def invalidate_steering(self, old_steering_hash: str) -> int:
        """
        Invalidate all entries with a specific steering hash.

        Called when steering values change.

        Args:
            old_steering_hash: The steering hash to invalidate.

        Returns:
            Number of entries invalidated.
        """
        keys_to_remove = [
            k for k, v in self._cache.items() if v.steering_hash == old_steering_hash
        ]
        for key in keys_to_remove:
            entry = self._cache.pop(key)
            del entry.past_key_values

        if keys_to_remove:
            logger.info("prefix_cache_invalidated", count=len(keys_to_remove))
        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cached entries and free GPU memory."""
        for entry in self._cache.values():
            del entry.past_key_values
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("prefix_cache_cleared")

    @staticmethod
    def get_steering_hash() -> str:
        """
        Get a hash of the current steering state.

        Returns empty string if no SAE is attached or steering is disabled.
        """
        try:
            from millm.services.sae_service import AttachedSAEState

            state = AttachedSAEState()
            if not state.is_attached or state.attached_sae is None:
                return ""

            sae = state.attached_sae
            if not sae.is_steering_enabled or sae.steering_delta is None:
                return ""

            # Hash the steering delta tensor
            delta_bytes = sae.steering_delta.cpu().numpy().tobytes()
            return hashlib.md5(delta_bytes).hexdigest()[:12]
        except Exception:
            return ""

    @staticmethod
    def _hash_text(text: str) -> str:
        """Create a short hash of text."""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    @staticmethod
    def _make_key(prompt_text: str, steering_hash: str) -> str:
        """Create cache key from prompt and steering state."""
        prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:16]
        return f"{prompt_hash}:{steering_hash}"
