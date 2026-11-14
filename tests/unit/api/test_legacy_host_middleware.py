"""T09: requests under an old (miLLM) hostname are recorded (040 FR-040.11; FTDD §6.3; ID-4), and
T33's P7 inverse: once retired, they are not (FR-040.40).

Driven through the REAL module-level `miserver.main.app` (the wrapper is asserted to be there), plus the
middleware alone for the header and scope rules. Since P7 the real alias table is empty and the
middleware watches nothing (it stays until P8, FTDD §6.12):
`test_retired_host_not_recorded` holds that for every old host of the export. The recording rules
are still tested, on a table planted back into the transition state (`planted` fixture).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import miserver.main as main
from miserver.api.legacy_host import LegacyHostMiddleware
from miserver.core import legacy_names
from tests.support import legacy_identifiers as li

OLD, NEW = "k8s-millm.hitsai.local", "k8s-miserver.hitsai.local"
PLANTED = (
    legacy_names.HostAlias(OLD, NEW, "ingress"),
    legacy_names.HostAlias("millm.hitsai.net", "miserver.hitsai.net", "public"),
    legacy_names.HostAlias("k8s-millm-otel.hitsai.local", "k8s-miserver-otel.hitsai.local", "ingress",
                           observed=False),
)


@pytest.fixture(autouse=True)
def _clean():
    legacy_names.drain_hits()
    yield
    legacy_names.drain_hits()


@pytest.fixture
def planted():
    """The transition state: three old hosts in the table (one unobserved)."""
    saved = legacy_names.HOST_ALIASES
    legacy_names.HOST_ALIASES = PLANTED
    legacy_names._rebuild()
    try:
        yield
    finally:
        legacy_names.HOST_ALIASES = saved
        legacy_names._rebuild()


def _hosts() -> dict[str, int]:
    return {old: h.count for (kind, old), h in legacy_names.drain_hits().items() if kind == "host"}


async def _call(scope_type: str, headers: list[tuple[bytes, bytes]], path: str = "/") -> list[dict]:
    seen: list[dict] = []

    async def app(scope, receive, send):
        seen.append(scope)

    async def receive():
        return {"type": "http.request"}

    async def send(_message):
        return None

    await LegacyHostMiddleware(app)({"type": scope_type, "headers": headers, "path": path},
                                    receive, send)
    return seen


def test_the_app_is_wrapped():
    assert isinstance(main.app, LegacyHostMiddleware)
    assert isinstance(main.fastapi_app_of(main.app), FastAPI)


def test_retired_host_not_recorded():
    """T33 (040 FR-040.40), on the real tables: every old host the export carries is `retired`, and a
    request under it, through the real app, answers and records nothing."""
    records = li.load().hosts()
    assert records, "the export carries the host records until S001 drops them"
    assert {r["status"] for r in records} == {"retired"}, records
    for record in records:
        old = record["old"]
        assert TestClient(main.app, base_url=f"http://{old}").get("/api/health").status_code == 200, old
        forwarded = TestClient(main.app, base_url=f"http://{NEW}").get(
            "/api/health", headers={"X-Forwarded-Host": old})
        assert forwarded.status_code == 200, old
    assert _hosts() == {}, "a retired host is not a use"
    assert legacy_names.HOST_ALIASES == (), "P7 empties the table"


def test_old_host_recorded_once_per_request(planted):
    client = TestClient(main.app, base_url=f"http://{OLD}")
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health").status_code == 200
    assert _hosts() == {OLD: 2}


def test_new_host_not_recorded():
    client = TestClient(main.app, base_url=f"http://{NEW}")
    assert client.get("/api/health").status_code == 200
    assert _hosts() == {}


async def test_port_and_case_normalised(planted):
    await _call("http", [(b"host", b"K8S-MILLM.hitsai.LOCAL:8443")])
    assert _hosts() == {OLD: 1}


async def test_forwarded_host_read(planted):
    await _call("http", [(b"host", NEW.encode()), (b"x-forwarded-host", b"millm.hitsai.net, proxy")])
    assert _hosts() == {"millm.hitsai.net": 1}
    await _call("http", [(b"host", OLD.encode()), (b"x-forwarded-host", OLD.encode())])
    assert _hosts() == {OLD: 1}, "one request is one use, however many headers name it"


async def test_socketio_path_observed(planted):
    await _call("websocket", [(b"host", OLD.encode())], path="/socket.io/")
    assert _hosts() == {OLD: 1}


async def test_lifespan_scope_passes_through(planted):
    seen = await _call("lifespan", [(b"host", OLD.encode())])
    assert [s["type"] for s in seen] == ["lifespan"], "the lifespan must reach the app"
    assert _hosts() == {}


async def test_an_unobserved_alias_is_never_recorded(planted):
    await _call("http", [(b"host", b"k8s-millm-otel.hitsai.local")])
    assert _hosts() == {}


def test_fastapi_app_of_unwraps_both_layers():
    inner = FastAPI()
    import socketio

    wrapped = LegacyHostMiddleware(socketio.ASGIApp(socketio.AsyncServer(async_mode="asgi"),
                                                    other_asgi_app=inner))
    assert main.fastapi_app_of(wrapped) is inner
    assert main.fastapi_app_of(inner) is inner
