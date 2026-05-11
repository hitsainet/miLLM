"""Unit tests for PrefixCache."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from millm.ml.prefix_cache import CacheEntry, PrefixCache


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cache():
    """Small cache (3 entries) for eviction testing."""
    return PrefixCache(max_entries=3, enabled=True)


@pytest.fixture
def disabled_cache():
    return PrefixCache(max_entries=5, enabled=False)


def _make_kv(seed: int = 0) -> torch.Tensor:
    """Tiny tensor to stand in for past_key_values."""
    torch.manual_seed(seed)
    return torch.randn(2, 4)


# =============================================================================
# Tests: enabled flag
# =============================================================================


class TestPrefixCacheEnabled:
    def test_enabled_by_default_on_construction(self):
        c = PrefixCache()
        assert c.enabled is True

    def test_disabled_get_always_returns_none(self, disabled_cache):
        disabled_cache.put("sys", "", _make_kv(), 5)
        # put is a no-op when disabled, so get should also return None
        assert disabled_cache.get("sys", "") is None

    def test_disabled_put_does_not_store(self, disabled_cache):
        disabled_cache.put("prompt", "hash", _make_kv(), 10)
        assert disabled_cache.size == 0


# =============================================================================
# Tests: put and get
# =============================================================================


class TestPrefixCacheGetPut:
    def test_get_returns_none_on_miss(self, cache):
        assert cache.get("unknown prompt", "hash") is None

    def test_put_then_get_returns_entry(self, cache):
        kv = _make_kv()
        cache.put("system: you are helpful", "abc", kv, 8)
        entry = cache.get("system: you are helpful", "abc")
        assert entry is not None
        assert entry.prompt_token_count == 8
        assert entry.steering_hash == "abc"

    def test_wrong_steering_hash_is_miss(self, cache):
        cache.put("prompt", "hash_a", _make_kv(), 5)
        assert cache.get("prompt", "hash_b") is None

    def test_hit_increments_hit_count(self, cache):
        cache.put("p", "h", _make_kv(), 3)
        entry = cache.get("p", "h")
        assert entry.hit_count == 1
        cache.get("p", "h")
        assert cache.get("p", "h").hit_count == 3

    def test_stats_track_hits_and_misses(self, cache):
        cache.put("p", "h", _make_kv(), 3)
        cache.get("p", "h")   # hit
        cache.get("p", "x")   # miss
        cache.get("q", "h")   # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["size"] == 1


# =============================================================================
# Tests: LRU eviction
# =============================================================================


class TestPrefixCacheEviction:
    def test_evicts_oldest_when_full(self, cache):
        # Fill cache (max_entries=3)
        cache.put("A", "h", _make_kv(0), 1)
        cache.put("B", "h", _make_kv(1), 1)
        cache.put("C", "h", _make_kv(2), 1)
        assert cache.size == 3
        # Add one more — "A" should be evicted (LRU)
        cache.put("D", "h", _make_kv(3), 1)
        assert cache.size == 3
        assert cache.get("A", "h") is None
        assert cache.get("D", "h") is not None

    def test_access_refreshes_lru_order(self, cache):
        cache.put("A", "h", _make_kv(0), 1)
        cache.put("B", "h", _make_kv(1), 1)
        cache.put("C", "h", _make_kv(2), 1)
        # Touch "A" — it becomes most-recently-used
        cache.get("A", "h")
        # Adding "D" should now evict "B" (the new LRU)
        cache.put("D", "h", _make_kv(3), 1)
        assert cache.get("A", "h") is not None
        assert cache.get("B", "h") is None
        assert cache.get("D", "h") is not None


# =============================================================================
# Tests: invalidate_steering
# =============================================================================


class TestPrefixCacheInvalidateSteering:
    def test_removes_entries_matching_hash(self, cache):
        cache.put("P1", "steer_a", _make_kv(0), 5)
        cache.put("P2", "steer_a", _make_kv(1), 6)
        cache.put("P3", "steer_b", _make_kv(2), 7)
        removed = cache.invalidate_steering("steer_a")
        assert removed == 2
        assert cache.size == 1
        assert cache.get("P3", "steer_b") is not None

    def test_invalidate_nonexistent_hash_removes_nothing(self, cache):
        cache.put("P", "real_hash", _make_kv(), 4)
        removed = cache.invalidate_steering("phantom_hash")
        assert removed == 0
        assert cache.size == 1

    def test_invalidate_empty_hash_targets_no_steering_entries(self, cache):
        cache.put("P", "", _make_kv(0), 3)   # no-steering entry
        cache.put("Q", "s1", _make_kv(1), 4)  # steered entry
        removed = cache.invalidate_steering("")
        assert removed == 1
        assert cache.get("P", "") is None
        assert cache.get("Q", "s1") is not None


# =============================================================================
# Tests: clear
# =============================================================================


class TestPrefixCacheClear:
    def test_clear_removes_all_entries(self, cache):
        cache.put("A", "h", _make_kv(0), 1)
        cache.put("B", "h", _make_kv(1), 1)
        cache.clear()
        assert cache.size == 0
        assert cache.get("A", "h") is None

    def test_clear_resets_hit_miss_stats(self, cache):
        cache.put("A", "h", _make_kv(), 1)
        cache.get("A", "h")
        cache.get("B", "h")
        cache.clear()
        stats = cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0


# =============================================================================
# Tests: get_steering_hash
# =============================================================================


class TestGetSteeringHash:
    def test_returns_empty_string_when_no_sae(self):
        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            MockState.return_value.is_attached = False
            result = PrefixCache.get_steering_hash()
        assert result == ""

    def test_returns_empty_string_when_steering_disabled(self):
        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            state = MockState.return_value
            state.is_attached = True
            sae = MagicMock()
            sae.is_steering_enabled = False
            sae.steering_delta = None
            state.attached_sae = sae
            result = PrefixCache.get_steering_hash()
        assert result == ""

    def test_returns_hash_string_when_steering_active(self):
        delta = torch.tensor([1.0, 2.0, 3.0])
        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            state = MockState.return_value
            state.is_attached = True
            sae = MagicMock()
            sae.is_steering_enabled = True
            sae.steering_delta = delta
            state.attached_sae = sae
            result = PrefixCache.get_steering_hash()
        assert isinstance(result, str)
        assert len(result) == 12  # MD5 truncated to 12 chars

    def test_same_delta_produces_same_hash(self):
        delta = torch.tensor([1.0, 2.0])
        with patch("millm.services.sae_service.AttachedSAEState") as MockState:
            state = MockState.return_value
            state.is_attached = True
            sae = MagicMock()
            sae.is_steering_enabled = True
            sae.steering_delta = delta
            state.attached_sae = sae
            h1 = PrefixCache.get_steering_hash()
            h2 = PrefixCache.get_steering_hash()
        assert h1 == h2

    def test_different_deltas_produce_different_hashes(self):
        delta_a = torch.tensor([1.0, 0.0])
        delta_b = torch.tensor([0.0, 1.0])
        results = []
        for delta in (delta_a, delta_b):
            with patch("millm.services.sae_service.AttachedSAEState") as MockState:
                state = MockState.return_value
                state.is_attached = True
                sae = MagicMock()
                sae.is_steering_enabled = True
                sae.steering_delta = delta
                state.attached_sae = sae
                results.append(PrefixCache.get_steering_hash())
        assert results[0] != results[1]

    def test_returns_empty_on_exception(self):
        # Simulate an unexpected error inside get_steering_hash's import chain
        with patch("millm.services.sae_service.AttachedSAEState", side_effect=RuntimeError):
            result = PrefixCache.get_steering_hash()
        assert result == ""
