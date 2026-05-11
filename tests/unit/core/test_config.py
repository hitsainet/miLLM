"""Unit tests for Settings / config.py."""

import os
from unittest.mock import patch

import pytest

from millm.core.config import Settings


# =============================================================================
# Tests: defaults
# =============================================================================


class TestSettingsDefaults:
    """Verify documented defaults so regressions are caught immediately."""

    def test_database_url_has_localhost_default(self):
        s = Settings()
        assert "localhost" in s.DATABASE_URL

    def test_hf_token_defaults_to_none(self):
        assert Settings().HF_TOKEN is None

    def test_debug_defaults_false(self):
        assert Settings().DEBUG is False

    def test_cors_origins_defaults_to_wildcard(self):
        assert Settings().CORS_ORIGINS == "*"

    def test_cors_origins_list_wildcard(self):
        assert Settings().cors_origins_list == ["*"]

    def test_max_concurrent_requests_default(self):
        assert Settings().MAX_CONCURRENT_REQUESTS == 2

    def test_max_pending_requests_default(self):
        assert Settings().MAX_PENDING_REQUESTS == 10

    def test_torch_compile_default_is_none(self):
        # None means auto-detect (not False — we changed this)
        assert Settings().TORCH_COMPILE is None

    def test_enable_prefix_cache_default_true(self):
        assert Settings().ENABLE_PREFIX_CACHE is True

    def test_prefix_cache_max_entries_default(self):
        assert Settings().PREFIX_CACHE_MAX_ENTRIES == 5

    def test_cbm_disabled_by_default(self):
        assert Settings().ENABLE_CONTINUOUS_BATCHING is False

    def test_cbm_force_serial_monitoring_default_false(self):
        assert Settings().CBM_FORCE_SERIAL_MONITORING is False

    def test_speculative_model_defaults_to_none(self):
        assert Settings().SPECULATIVE_MODEL is None

    def test_kv_cache_mode_default_dynamic(self):
        assert Settings().KV_CACHE_MODE == "dynamic"


# =============================================================================
# Tests: cors_origins_list property
# =============================================================================


class TestCorsOriginsList:
    def test_wildcard_returns_list_with_star(self):
        s = Settings(CORS_ORIGINS="*")
        assert s.cors_origins_list == ["*"]

    def test_single_origin_parsed(self):
        s = Settings(CORS_ORIGINS="http://localhost:3000")
        assert s.cors_origins_list == ["http://localhost:3000"]

    def test_multiple_origins_split_by_comma(self):
        s = Settings(CORS_ORIGINS="http://a.com,http://b.com")
        assert s.cors_origins_list == ["http://a.com", "http://b.com"]

    def test_whitespace_around_commas_stripped(self):
        s = Settings(CORS_ORIGINS="http://a.com , http://b.com")
        assert s.cors_origins_list == ["http://a.com", "http://b.com"]


# =============================================================================
# Tests: environment variable override
# =============================================================================


class TestSettingsEnvOverride:
    def test_hf_token_read_from_env(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test123"}, clear=False):
            s = Settings()
            assert s.HF_TOKEN == "hf_test123"

    def test_debug_read_from_env(self):
        with patch.dict(os.environ, {"DEBUG": "true"}, clear=False):
            s = Settings()
            assert s.DEBUG is True

    def test_torch_compile_false_from_env(self):
        with patch.dict(os.environ, {"TORCH_COMPILE": "false"}, clear=False):
            s = Settings()
            assert s.TORCH_COMPILE is False

    def test_torch_compile_true_from_env(self):
        with patch.dict(os.environ, {"TORCH_COMPILE": "true"}, clear=False):
            s = Settings()
            assert s.TORCH_COMPILE is True

    def test_torch_compile_none_when_not_set(self):
        env = {k: v for k, v in os.environ.items() if k != "TORCH_COMPILE"}
        with patch.dict(os.environ, env, clear=True):
            s = Settings()
            assert s.TORCH_COMPILE is None

    def test_max_concurrent_requests_from_env(self):
        with patch.dict(os.environ, {"MAX_CONCURRENT_REQUESTS": "4"}, clear=False):
            s = Settings()
            assert s.MAX_CONCURRENT_REQUESTS == 4

    def test_cbm_force_serial_monitoring_from_env(self):
        with patch.dict(os.environ, {"CBM_FORCE_SERIAL_MONITORING": "true"}, clear=False):
            s = Settings()
            assert s.CBM_FORCE_SERIAL_MONITORING is True

    def test_cors_origins_from_env(self):
        with patch.dict(
            os.environ, {"CORS_ORIGINS": "http://app.local:8080"}, clear=False
        ):
            s = Settings()
            assert s.cors_origins_list == ["http://app.local:8080"]


# =============================================================================
# Tests: LOG_FORMAT literal constraint
# =============================================================================


class TestLogFormatConstraint:
    def test_json_format_accepted(self):
        s = Settings(LOG_FORMAT="json")
        assert s.LOG_FORMAT == "json"

    def test_console_format_accepted(self):
        s = Settings(LOG_FORMAT="console")
        assert s.LOG_FORMAT == "console"

    def test_invalid_format_rejected(self):
        with pytest.raises(Exception):
            Settings(LOG_FORMAT="invalid")
