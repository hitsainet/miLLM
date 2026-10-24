"""GGUF_TENSOR_SPLIT is parsed once, and a malformed value fails at startup.

A typo that parsed as "unset" would split by free memory while the operator
believed their split was in force.

MUTATION CONTROLS (mutate.py, 2026-09-14; restored and sha256-verified):
  M25 drop the Settings validator -> test_a_malformed_value_fails_at_startup
"""

import pytest
from pydantic import ValidationError

from millm.core.config import Settings, parse_gguf_tensor_split


class TestParse:
    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_unset(self, value):
        assert parse_gguf_tensor_split(value) is None

    def test_proportions(self):
        assert parse_gguf_tensor_split("3, 1") == [3.0, 1.0]
        assert parse_gguf_tensor_split("0,1") == [0.0, 1.0]

    @pytest.mark.parametrize("value", ["a,b", "1,-1", "0,0", "nan,1", "inf,1", "1,,2"])
    def test_refused(self, value):
        with pytest.raises(ValueError):
            parse_gguf_tensor_split(value)


class TestSettings:
    def test_defaults_to_unset(self):
        assert Settings().GGUF_TENSOR_SPLIT == ""

    def test_a_malformed_value_fails_at_startup(self):
        with pytest.raises(ValidationError):
            Settings(GGUF_TENSOR_SPLIT="three,one")

    def test_a_valid_value_is_kept_as_written(self):
        assert Settings(GGUF_TENSOR_SPLIT="3,1").GGUF_TENSOR_SPLIT == "3,1"
