"""
Unit tests for ModelRepository.
"""

import pytest

from millm.db.models.model import Model, ModelSource, ModelStatus, QuantizationType
from millm.db.repositories.model_repository import ModelRepository


@pytest.fixture
def repository(test_session) -> ModelRepository:
    """Create a ModelRepository with test session."""
    return ModelRepository(test_session)


@pytest.fixture
def sample_model_data() -> dict:
    """Sample model data for creating test models."""
    return {
        "name": "gemma-2-2b",
        "source": ModelSource.HUGGINGFACE,
        "repo_id": "google/gemma-2-2b",
        "quantization": QuantizationType.Q4,
        "cache_path": "/data/models/huggingface/google--gemma-2-2b--Q4",
        "params": "2.5B",
        "architecture": "causal-lm",
        "disk_size_mb": 1500,
        "estimated_memory_mb": 2000,
        "trust_remote_code": False,
        "status": ModelStatus.READY,
    }


class TestModelRepositoryCreate:
    """Tests for ModelRepository.create()."""

    async def test_create_returns_model_with_id(self, repository, sample_model_data):
        """Create should return a model with a populated id."""
        model = await repository.create(**sample_model_data)

        assert model.id is not None
        assert model.id > 0

    async def test_create_sets_all_fields(self, repository, sample_model_data):
        """Create should set all provided fields."""
        model = await repository.create(**sample_model_data)

        assert model.name == sample_model_data["name"]
        assert model.source == sample_model_data["source"]
        assert model.repo_id == sample_model_data["repo_id"]
        assert model.quantization == sample_model_data["quantization"]
        assert model.cache_path == sample_model_data["cache_path"]

    async def test_create_sets_timestamps(self, repository, sample_model_data):
        """Create should set created_at and updated_at timestamps."""
        model = await repository.create(**sample_model_data)

        assert model.created_at is not None
        assert model.updated_at is not None

    async def test_create_defaults_status_to_ready(self, repository):
        """Create should default status to READY if not specified."""
        data = {
            "name": "test-model",
            "source": ModelSource.LOCAL,
            "local_path": "/models/test",
            "quantization": QuantizationType.FP16,
            "cache_path": "/data/models/local/test",
        }

        model = await repository.create(**data)

        assert model.status == ModelStatus.READY


class TestModelRepositoryGetById:
    """Tests for ModelRepository.get_by_id()."""

    async def test_get_by_id_returns_existing_model(self, repository, sample_model_data):
        """get_by_id should return the model if it exists."""
        created = await repository.create(**sample_model_data)

        found = await repository.get_by_id(created.id)

        assert found is not None
        assert found.id == created.id
        assert found.name == created.name

    async def test_get_by_id_returns_none_for_nonexistent(self, repository):
        """get_by_id should return None for non-existent id."""
        found = await repository.get_by_id(99999)

        assert found is None


class TestModelRepositoryGetAll:
    """Tests for ModelRepository.get_all()."""

    async def test_get_all_returns_empty_list_when_no_models(self, repository):
        """get_all should return empty list when no models exist."""
        models = await repository.get_all()

        assert models == []

    async def test_get_all_returns_all_models(self, repository, sample_model_data):
        """get_all should return all models."""
        model1 = await repository.create(**sample_model_data)

        data2 = sample_model_data.copy()
        data2["repo_id"] = "other/model"
        data2["name"] = "other-model"
        data2["cache_path"] = "/data/models/other"
        model2 = await repository.create(**data2)

        models = await repository.get_all()

        assert len(models) == 2
        model_ids = {m.id for m in models}
        assert model1.id in model_ids
        assert model2.id in model_ids


class TestModelRepositoryFindByRepoQuantization:
    """Tests for ModelRepository.find_by_repo_quantization()."""

    async def test_find_by_repo_quantization_returns_matching_model(
        self, repository, sample_model_data
    ):
        """find_by_repo_quantization should return matching model."""
        created = await repository.create(**sample_model_data)

        found = await repository.find_by_repo_quantization(
            repo_id="google/gemma-2-2b",
            quantization=QuantizationType.Q4,
        )

        assert found is not None
        assert found.id == created.id

    async def test_find_by_repo_quantization_returns_none_for_different_quantization(
        self, repository, sample_model_data
    ):
        """find_by_repo_quantization should return None for different quantization."""
        await repository.create(**sample_model_data)

        found = await repository.find_by_repo_quantization(
            repo_id="google/gemma-2-2b",
            quantization=QuantizationType.Q8,  # Different quantization
        )

        assert found is None


class TestModelRepositoryUpdate:
    """Tests for ModelRepository.update()."""

    async def test_update_modifies_fields(self, repository, sample_model_data):
        """update should modify specified fields."""
        created = await repository.create(**sample_model_data)

        updated = await repository.update(
            created.id,
            name="new-name",
            disk_size_mb=2000,
        )

        assert updated is not None
        assert updated.name == "new-name"
        assert updated.disk_size_mb == 2000
        # Other fields unchanged
        assert updated.repo_id == sample_model_data["repo_id"]

    async def test_update_updates_timestamp(self, repository, sample_model_data):
        """update should update the updated_at timestamp."""
        created = await repository.create(**sample_model_data)
        original_updated = created.updated_at

        updated = await repository.update(created.id, name="new-name")

        assert updated is not None
        assert updated.updated_at >= original_updated

    async def test_update_returns_none_for_nonexistent(self, repository):
        """update should return None for non-existent id."""
        updated = await repository.update(99999, name="new-name")

        assert updated is None


class TestModelRepositoryUpdateStatus:
    """Tests for ModelRepository.update_status()."""

    async def test_update_status_changes_status(self, repository, sample_model_data):
        """update_status should change the model's status."""
        created = await repository.create(**sample_model_data)

        updated = await repository.update_status(created.id, ModelStatus.LOADING)

        assert updated is not None
        assert updated.status == ModelStatus.LOADING

    async def test_update_status_sets_error_message_on_error(
        self, repository, sample_model_data
    ):
        """update_status should set error_message when status is ERROR."""
        created = await repository.create(**sample_model_data)

        updated = await repository.update_status(
            created.id,
            ModelStatus.ERROR,
            error_message="Something went wrong",
        )

        assert updated is not None
        assert updated.status == ModelStatus.ERROR
        assert updated.error_message == "Something went wrong"

    async def test_update_status_sets_loaded_at_when_loaded(
        self, repository, sample_model_data
    ):
        """update_status should set loaded_at when status changes to LOADED."""
        created = await repository.create(**sample_model_data)
        assert created.loaded_at is None

        updated = await repository.update_status(created.id, ModelStatus.LOADED)

        assert updated is not None
        assert updated.loaded_at is not None

    async def test_update_status_clears_loaded_at_when_not_loaded(
        self, repository, sample_model_data
    ):
        """update_status should clear loaded_at when status changes away from LOADED."""
        sample_model_data["status"] = ModelStatus.LOADED
        created = await repository.create(**sample_model_data)
        await repository.update_status(created.id, ModelStatus.LOADED)

        updated = await repository.update_status(created.id, ModelStatus.READY)

        assert updated is not None
        assert updated.loaded_at is None


class TestModelRepositoryDelete:
    """Tests for ModelRepository.delete()."""

    async def test_delete_removes_model(self, repository, sample_model_data):
        """delete should remove the model from database."""
        created = await repository.create(**sample_model_data)

        result = await repository.delete(created.id)

        assert result is True
        found = await repository.get_by_id(created.id)
        assert found is None

    async def test_delete_returns_false_for_nonexistent(self, repository):
        """delete should return False for non-existent id."""
        result = await repository.delete(99999)

        assert result is False


class TestModelRepositoryExists:
    """Tests for ModelRepository.exists()."""

    async def test_exists_returns_true_for_existing_model(
        self, repository, sample_model_data
    ):
        """exists should return True for existing model."""
        created = await repository.create(**sample_model_data)

        assert await repository.exists(created.id) is True

    async def test_exists_returns_false_for_nonexistent(self, repository):
        """exists should return False for non-existent id."""
        assert await repository.exists(99999) is False
