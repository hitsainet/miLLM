"""Two quantizations of one repository must both be reachable by name.

Stage 1 widened uniqueness to (repo_id, quantization, gguf_label) so several
quants could coexist — the entire point of the GGUF picker. The display name
stayed `repo_id.split("/")[-1]`, so the moment a second quant was downloaded
both rows carried the same name, `find_by_name`'s `scalar_one_or_none()` raised
"Multiple rows were found when one or none was required", and every OpenAI
request naming that model returned 500. Observed live:

    43 | gemma-4-31b-it-...-GGUF | Q4_K_M | ready
    44 | gemma-4-31b-it-...-GGUF | IQ4_XS | loaded

The model was loaded and serving; it was simply unreachable by name. Three
review rounds missed it because no fixture ever held two rows from one repo —
which is exactly the fixture below.

MUTATION CONTROLS (each must turn this file red):
  * (the naming half lives in tests/unit/services/test_model_service.py,
    TestTwoQuantsOfOneRepoGetDistinctNames, which runs download_model itself)
  * return the first candidate instead of raising   -> "ambiguity is refused" fails
  * drop the bare-name fallback                     -> "a bare name still works" fails
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from millm.core.errors import AmbiguousModelNameError
from millm.db.models.model import Model, QuantizationType
from millm.db.repositories.model_repository import ModelRepository
from tests.support.factories import make_model


async def _add(session, **overrides) -> Model:
    model = make_model(**overrides)
    session.add(model)
    await session.commit()
    return model


class TestTwoQuantsOfOneRepo:
    @pytest.mark.asyncio
    async def test_each_quant_resolves_to_itself(self, test_session):
        """The bug: this raised instead of returning either."""
        await _add(
            test_session, id=1, name="foo-GGUF:Q4_K_M", repo_id="o/foo-GGUF",
            gguf_label="Q4_K_M", cache_path="p1",
        )
        await _add(
            test_session, id=2, name="foo-GGUF:IQ4_XS", repo_id="o/foo-GGUF",
            gguf_label="IQ4_XS", cache_path="p2",
        )
        repo = ModelRepository(test_session)

        assert (await repo.find_by_name("foo-GGUF:Q4_K_M")).id == 1
        assert (await repo.find_by_name("foo-GGUF:IQ4_XS")).id == 2

    @pytest.mark.asyncio
    async def test_a_bare_name_still_works_while_unambiguous(self, test_session):
        """Existing callers and saved client selections must keep working."""
        await _add(
            test_session, id=1, name="foo-GGUF:Q4_K_M", repo_id="o/foo-GGUF",
            gguf_label="Q4_K_M", cache_path="p1",
        )
        repo = ModelRepository(test_session)

        assert (await repo.find_by_name("foo-GGUF")).id == 1

    @pytest.mark.asyncio
    async def test_ambiguity_is_refused_rather_than_guessed(self, test_session):
        """Picking one would make the served model depend on insertion order."""
        await _add(
            test_session, id=1, name="foo-GGUF:Q4_K_M", repo_id="o/foo-GGUF",
            gguf_label="Q4_K_M", cache_path="p1",
        )
        await _add(
            test_session, id=2, name="foo-GGUF:IQ4_XS", repo_id="o/foo-GGUF",
            gguf_label="IQ4_XS", cache_path="p2",
        )
        repo = ModelRepository(test_session)

        with pytest.raises(AmbiguousModelNameError) as exc:
            await repo.find_by_name("foo-GGUF")

        message = str(exc.value)
        assert "Q4_K_M" in message and "IQ4_XS" in message, (
            "the caller must be told which tags exist, not merely refused"
        )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_an_unknown_name_is_still_None(self, test_session):
        """Absent must stay absent, not become an ambiguity error."""
        repo = ModelRepository(test_session)
        assert await repo.find_by_name("nothing-like-this") is None

    @pytest.mark.asyncio
    async def test_an_ordinary_model_is_untouched(self, test_session):
        """A non-GGUF model has no tag and must resolve exactly as before."""
        await _add(
            test_session, id=1, name="gemma-2-2b", repo_id="google/gemma-2-2b",
            gguf_label="", cache_path="p1",
        )
        repo = ModelRepository(test_session)

        assert (await repo.find_by_name("gemma-2-2b")).id == 1


class TestTheRefusalReachesTheClientAsA400:
    """The repository raising is only half the fix.

    `AmbiguousModelNameError` is raised deep in the repository and nothing on
    the route catches it. It becomes a useful answer only because
    `millm_error_handler` is registered for `MiLLMError` in main.py and looks
    the code up in `ERROR_STATUS_MAP`. Either of those going missing turns a
    precise 400 back into the 500 this whole change exists to remove — and no
    test of the repository alone can see that.

    MUTATION CONTROLS:
      * remove the AMBIGUOUS_MODEL_NAME row from ERROR_STATUS_MAP -> the type
        assertion fails (it falls back to "server_error")
      * remove app.add_exception_handler(MiLLMError, ...) from main.py -> the
        request raises instead of returning 400
    """

    def _client(self):
        from millm.api.dependencies import get_inference_service, get_model_service
        from millm.main import create_app

        svc = MagicMock()
        svc.find_model_by_name = AsyncMock(
            side_effect=AmbiguousModelNameError(
                "'foo-GGUF' matches 2 quantizations: foo-GGUF:IQ4_XS, "
                "foo-GGUF:Q4_K_M. Name one of them exactly."
            )
        )
        app = create_app()
        app.dependency_overrides[get_model_service] = lambda: svc
        app.dependency_overrides[get_inference_service] = lambda: MagicMock()
        return TestClient(app)

    def test_an_ambiguous_name_is_a_400_naming_the_tags(self):
        response = self._client().post(
            "/v1/chat/completions",
            json={"model": "foo-GGUF", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert response.status_code == 400, response.text
        body = response.json()["error"]
        assert body["type"] == "invalid_request_error", (
            "an OpenAI client retries a server_error and edits the request on "
            "an invalid_request_error; this one is the caller's to fix"
        )
        assert "IQ4_XS" in body["message"] and "Q4_K_M" in body["message"]
