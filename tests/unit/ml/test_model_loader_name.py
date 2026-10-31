"""ModelLoader.model_name — the attribute /health reads.

Regression: it was never defined, so reading it raised AttributeError inside
the health check's try/except. Every successfully loaded model was reported as
`model_loader: unhealthy` with a null model_name — a health signal crying wolf
on a healthy runtime, which is worse than no signal.
"""

from datetime import datetime, timezone

import pytest

from millm.ml.model_loader import LoadedModel, LoadedModelState, ModelLoader


@pytest.fixture(autouse=True)
def clean_state():
    """LoadedModelState is a process singleton — leave it as we found it."""
    LoadedModelState().clear()
    yield
    LoadedModelState().clear()


def _loaded(name="lfm2.5-1.2b-instruct"):
    return LoadedModel(
        model_id=1,
        model_name=name,
        model=object(),
        tokenizer=object(),
        loaded_at=datetime.now(timezone.utc),
    )


def test_model_name_is_none_when_nothing_is_loaded():
    assert ModelLoader().model_name is None


def test_model_name_reports_the_loaded_model():
    loader = ModelLoader()
    loader.state.set(_loaded())
    assert loader.model_name == "lfm2.5-1.2b-instruct"


def test_reading_it_never_raises():
    """The health check calls this inside a try/except that downgrades ANY
    exception to unhealthy, so a raise here is indistinguishable from a real
    fault."""
    loader = ModelLoader()
    loader.model_name  # unloaded: must not raise
    loader.state.set(_loaded("x"))
    assert isinstance(loader.model_name, str)


def test_the_health_route_reads_this_exact_attribute():
    """Pins the coupling: the route reads `model_loader.model_name`, and the
    only reason the outage was invisible is that its try/except downgraded the
    AttributeError to a generic 'component check failed'."""
    import inspect

    from millm.api.routes.system import health

    src = inspect.getsource(health)
    assert "model_loader.model_name" in src
