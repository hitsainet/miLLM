"""Unit tests for memory utilities."""

from unittest.mock import MagicMock, patch

import pytest

from millm.ml.memory_utils import (
    estimate_memory_mb,
    format_memory_mb,
    get_available_memory_mb,
    get_device_count,
    get_device_name,
    get_total_memory_mb,
    get_used_memory_mb,
    is_cuda_available,
    parse_params,
    verify_memory_available,
)


class TestParseParams:
    """Tests for parse_params function."""

    def test_parses_billions(self):
        """Test parsing billion parameter strings."""
        assert parse_params("7B") == 7_000_000_000
        assert parse_params("2.5B") == 2_500_000_000
        assert parse_params("70B") == 70_000_000_000

    def test_parses_millions(self):
        """Test parsing million parameter strings."""
        assert parse_params("350M") == 350_000_000
        assert parse_params("1.5M") == 1_500_000

    def test_parses_trillions(self):
        """Test parsing trillion parameter strings."""
        assert parse_params("1T") == 1_000_000_000_000
        assert parse_params("1.5T") == 1_500_000_000_000

    def test_parses_thousands(self):
        """Test parsing thousand parameter strings."""
        assert parse_params("100K") == 100_000

    def test_handles_lowercase(self):
        """Test that lowercase suffixes work."""
        assert parse_params("7b") == 7_000_000_000
        assert parse_params("350m") == 350_000_000

    def test_handles_whitespace(self):
        """Test that whitespace is handled."""
        assert parse_params(" 7B ") == 7_000_000_000
        assert parse_params("7 B") == 7_000_000_000

    def test_returns_zero_for_none(self):
        """Test that None returns 0."""
        assert parse_params(None) == 0

    def test_returns_zero_for_unknown(self):
        """Test that 'unknown' returns 0."""
        assert parse_params("unknown") == 0
        assert parse_params("UNKNOWN") == 0

    def test_returns_zero_for_invalid(self):
        """Test that invalid strings return 0."""
        assert parse_params("") == 0
        assert parse_params("abc") == 0
        assert parse_params("B7") == 0


class TestEstimateMemoryMb:
    """Tests for estimate_memory_mb function."""

    def test_estimates_fp16(self):
        """Test FP16 memory estimation."""
        # 7B params * 2 bytes * 1.2 overhead = 16.8 GB = ~16800 MB
        result = estimate_memory_mb("7B", "FP16")
        assert 16000 < result < 17000

    def test_estimates_q8(self):
        """Test Q8 memory estimation."""
        # 7B params * 1 byte * 1.2 overhead = 8.4 GB = ~8400 MB
        result = estimate_memory_mb("7B", "Q8")
        assert 8000 < result < 9000

    def test_estimates_q4(self):
        """Test Q4 memory estimation."""
        # 7B params * 0.5 bytes * 1.2 overhead = 4.2 GB = ~4200 MB
        result = estimate_memory_mb("7B", "Q4")
        assert 4000 < result < 4500

    def test_returns_zero_for_unknown_params(self):
        """Test that unknown params returns 0."""
        assert estimate_memory_mb("unknown", "Q4") == 0
        assert estimate_memory_mb(None, "Q4") == 0

    def test_defaults_to_fp16_for_unknown_quantization(self):
        """Test that unknown quantization defaults to FP16."""
        result_unknown = estimate_memory_mb("7B", "UNKNOWN")
        result_fp16 = estimate_memory_mb("7B", "FP16")
        assert result_unknown == result_fp16


class TestGetAvailableMemoryMb:
    """Tests for get_available_memory_mb function."""

    @patch("millm.ml.memory_utils.torch")
    def test_returns_available_memory(self, mock_torch):
        """Test that available memory is returned correctly."""
        # Mock 8 GB free, 16 GB total
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (8 * 1024**3, 16 * 1024**3)

        result = get_available_memory_mb()

        assert result == 8 * 1024  # 8 GB in MB

    @patch("millm.ml.memory_utils.torch")
    def test_returns_zero_when_cuda_unavailable(self, mock_torch):
        """Test that 0 is returned when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False

        result = get_available_memory_mb()

        assert result == 0


class TestGetTotalMemoryMb:
    """Tests for get_total_memory_mb function."""

    @patch("millm.ml.memory_utils.torch")
    def test_returns_total_memory(self, mock_torch):
        """Test that total memory is returned correctly."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (8 * 1024**3, 16 * 1024**3)

        result = get_total_memory_mb()

        assert result == 16 * 1024  # 16 GB in MB

    @patch("millm.ml.memory_utils.torch")
    def test_returns_zero_when_cuda_unavailable(self, mock_torch):
        """Test that 0 is returned when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False

        result = get_total_memory_mb()

        assert result == 0


class TestGetUsedMemoryMb:
    """Tests for get_used_memory_mb function."""

    @patch("millm.ml.memory_utils.torch")
    def test_returns_used_memory(self, mock_torch):
        """Test that used memory is returned correctly."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 4 * 1024**3  # 4 GB

        result = get_used_memory_mb()

        assert result == 4 * 1024  # 4 GB in MB


class TestVerifyMemoryAvailable:
    """Tests for verify_memory_available function."""

    @patch("millm.ml.memory_utils.get_available_memory_mb")
    def test_returns_true_when_enough_memory(self, mock_get_available):
        """Test that True is returned when enough memory is available."""
        mock_get_available.return_value = 8000  # 8 GB available

        is_available, available = verify_memory_available(4000)  # Need 4 GB

        assert is_available is True
        assert available == 8000

    @patch("millm.ml.memory_utils.get_available_memory_mb")
    def test_returns_false_when_not_enough_memory(self, mock_get_available):
        """Test that False is returned when not enough memory."""
        mock_get_available.return_value = 4000  # 4 GB available

        is_available, available = verify_memory_available(8000)  # Need 8 GB

        assert is_available is False
        assert available == 4000


class TestIsCudaAvailable:
    """Tests for is_cuda_available function."""

    @patch("millm.ml.memory_utils.torch")
    def test_returns_true_when_available(self, mock_torch):
        """Test that True is returned when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True

        assert is_cuda_available() is True

    @patch("millm.ml.memory_utils.torch")
    def test_returns_false_when_not_available(self, mock_torch):
        """Test that False is returned when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False

        assert is_cuda_available() is False


class TestGetDeviceCount:
    """Tests for get_device_count function."""

    @patch("millm.ml.memory_utils.torch")
    def test_returns_device_count(self, mock_torch):
        """Test that device count is returned."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2

        assert get_device_count() == 2

    @patch("millm.ml.memory_utils.torch")
    def test_returns_zero_when_cuda_unavailable(self, mock_torch):
        """Test that 0 is returned when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False

        assert get_device_count() == 0


class TestGetDeviceName:
    """Tests for get_device_name function."""

    @patch("millm.ml.memory_utils.torch")
    def test_returns_device_name(self, mock_torch):
        """Test that device name is returned."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"

        assert get_device_name() == "NVIDIA RTX 4090"

    @patch("millm.ml.memory_utils.torch")
    def test_returns_no_gpu_when_cuda_unavailable(self, mock_torch):
        """Test that 'No GPU' is returned when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False

        assert get_device_name() == "No GPU"


class TestFormatMemoryMb:
    """Tests for format_memory_mb function."""

    def test_formats_as_gb_when_large(self):
        """Test that large values are formatted as GB."""
        assert format_memory_mb(8192) == "8.0 GB"
        assert format_memory_mb(16384) == "16.0 GB"

    def test_formats_as_mb_when_small(self):
        """Test that small values are formatted as MB."""
        assert format_memory_mb(512) == "512 MB"
        assert format_memory_mb(256) == "256 MB"

    def test_handles_boundary(self):
        """Test the GB/MB boundary."""
        assert format_memory_mb(1024) == "1.0 GB"
        assert format_memory_mb(1023) == "1023 MB"
