"""The three `/api/weights` routers are included in the LIVE app (addendum §2.1, W0a).

They are empty until their features fill them, so `app.openapi()["paths"]` cannot see them
yet; this reads the app's own router list instead (FastAPI 0.141 wraps each included router
in an `_IncludedRouter` whose `original_router` is the router object). The route-level
reachability tests — paths from `app.openapi()["paths"]`, payload and call count — arrive with
the routes in wave W3. Control: delete one `include_router` line -> red.
"""

from __future__ import annotations

import pytest

from millm.api.routes.management import weights, weights_adapters, weights_checkpoints


def _included_routers(app) -> list:
    return [r.original_router for r in app.router.routes if hasattr(r, "original_router")]


@pytest.mark.parametrize("module", [weights, weights_checkpoints, weights_adapters],
                         ids=lambda m: m.__name__.rsplit(".", 1)[1])
def test_each_weights_router_is_included_in_the_live_app(module):
    from millm.main import create_app

    included = _included_routers(create_app())
    assert any(r is module.router for r in included), module.__name__
    assert module.router.prefix == "/api/weights"


def test_the_walk_finds_the_existing_routers():
    """Guard the guard: a walk that found nothing would fail every row for the wrong reason."""
    from millm.api.routes.management.models import router as models_router
    from millm.main import create_app

    assert any(r is models_router for r in _included_routers(create_app()))
