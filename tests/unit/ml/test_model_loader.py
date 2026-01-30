"""Unit tests for model loader."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from millm.core.errors import InsufficientMemoryError, ModelLoadError
from millm.ml.model_loader import (
    LoadedModel,
    LoadedModelState,
    ModelLoadContext,
    ModelLoader,
)


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock()
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    return tokenizer


@pytest.fixture
def loaded_model(mock_model, mock_tokenizer):
    """Create a LoadedModel instance."""
    return LoadedModel(
        model_id=1,
        model=mock_model,
        tokenizer=mock_tokenizer,
        loaded_at=datetime.utcnow(),
        memory_used_mb=4000,
    )


class TestLoadedModel:
    """Tests for LoadedModel dataclass."""

    def test_creates_loaded_model(self, mock_model, mock_tokenizer):
        """Test that LoadedModel is created correctly."""
        loaded = LoadedModel(
            model_id=1,
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded_at=datetime.utcnow(),
            memory_used_mb=4000,
        )

        assert loaded.model_id == 1
        assert loaded.model is mock_model
        assert loaded.tokenizer is mock_tokenizer
        assert loaded.memory_used_mb == 4000


class TestLoadedModelState:
    """Tests for LoadedModelState singleton."""

    def test_is_singleton(self):
        """Test that LoadedModelState is a singleton."""
        state1 = LoadedModelState()
        state2 = LoadedModelState()

        assert state1 is state2

    def test_initial_state_is_not_loaded(self):
        """Test that initial state has no model loaded."""
        state = LoadedModelState()
        state._loaded = None  # Reset for test

        assert state.is_loaded is False
        assert state.current is None
        assert state.loaded_model_id is None

    def test_set_model(self, loaded_model):
        """Test setting a loaded model."""
        state = LoadedModelState()
        state.set(loaded_model)

        assert state.is_loaded is True
        assert state.current is loaded_model
        assert state.loaded_model_id == 1

    def test_clear_model(self, loaded_model):
        """Test clearing a loaded model."""
        state = LoadedModelState()
        state.set(loaded_model)

        with patch("millm.ml.model_loader.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            state.clear()

        assert state.is_loaded is False
        assert state.current is None


class TestModelLoadContext:
    """Tests for ModelLoadContext context manager."""

    def test_enters_and_exits_cleanly(self):
        """Test that context manager works without errors."""
        with ModelLoadContext(model_id=1) as ctx:
            assert ctx.model_id == 1
            assert ctx.model is None
            assert ctx.tokenizer is None

    def test_cleans_up_on_exception(self):
        """Test that cleanup happens on exception."""
        with patch("millm.ml.model_loader.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True

            with pytest.raises(ValueError):
                with ModelLoadContext(model_id=1) as ctx:
                    ctx.model = MagicMock()
                    ctx.tokenizer = MagicMock()
                    raise ValueError("Test error")

            mock_torch.cuda.empty_cache.assert_called()

    @patch("millm.ml.model_loader.AutoModelForCausalLM")
    @patch("millm.ml.model_loader.AutoTokenizer")
    @patch("millm.ml.model_loader.torch")
    def test_load_creates_loaded_model(
        self, mock_torch, mock_auto_tokenizer, mock_auto_model
    ):
        """Test that load creates a LoadedModel."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 4 * 1024 * 1024 * 1024
        mock_torch.float16 = "float16"

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        with ModelLoadContext(model_id=1) as ctx:
            loaded = ctx.load(
                cache_path="/data/models/test",
                quantization="FP16",
            )

        assert loaded.model_id == 1
        assert loaded.model is mock_model
        assert loaded.tokenizer is mock_tokenizer


class TestModelLoader:
    """Tests for ModelLoader class."""

    @pytest.fixture
    def loader(self):
        """Create a ModelLoader instance."""
        loader = ModelLoader()
        loader.state._loaded = None  # Reset state
        return loader

    def test_initial_state_not_loaded(self, loader):
        """Test that initial state has no model loaded."""
        assert loader.is_loaded is False
        assert loader.loaded_model_id is None

    @patch("millm.ml.model_loader.get_available_memory_mb")
    @patch("millm.ml.model_loader.torch")
    def test_load_checks_memory(self, mock_torch, mock_get_memory, loader):
        """Test that load checks memory availability."""
        mock_torch.cuda.is_available.return_value = True
        mock_get_memory.return_value = 4000  # 4 GB available

        with pytest.raises(InsufficientMemoryError) as exc_info:
            loader.load(
                model_id=1,
                cache_path="/data/models/test",
                quantization="FP16",
                estimated_memory_mb=8000,  # Need 8 GB
            )

        assert exc_info.value.details["available_mb"] == 4000
        assert exc_info.value.details["required_mb"] == 8000

    @patch("millm.ml.model_loader.torch")
    def test_load_requires_cuda(self, mock_torch, loader):
        """Test that load raises error when CUDA not available."""
        mock_torch.cuda.is_available.return_value = False

        with pytest.raises(ModelLoadError) as exc_info:
            loader.load(
                model_id=1,
                cache_path="/data/models/test",
                quantization="FP16",
                estimated_memory_mb=4000,
            )

        assert "CUDA is not available" in str(exc_info.value.message)

    def test_unload_returns_false_when_not_loaded(self, loader):
        """Test that unload returns False when no model is loaded."""
        result = loader.unload()

        assert result is False

    def test_unload_returns_true_when_loaded(self, loader, loaded_model):
        """Test that unload returns True when a model is unloaded."""
        loader.state.set(loaded_model)

        with patch("millm.ml.model_loader.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            result = loader.unload()

        assert result is True
        assert loader.is_loaded is False

    def test_get_memory_usage_when_loaded(self, loader, loaded_model):
        """Test getting memory usage when model is loaded."""
        loader.state.set(loaded_model)

        result = loader.get_memory_usage()

        assert result == 4000

    def test_get_memory_usage_when_not_loaded(self, loader):
        """Test getting memory usage when no model is loaded."""
        result = loader.get_memory_usage()

        assert result == 0
