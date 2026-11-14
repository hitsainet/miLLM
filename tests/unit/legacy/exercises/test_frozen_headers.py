"""Exercise for DR-08: the `X-miLLM-*` HTTP headers stay (040 FR-040.5).

External clients (Open WebUI, miDataworks, miForge, agents, scripts) send and read these headers;
renaming one breaks every client silently (an unknown request header is ignored, a missing
response header reads as "not served"). The test drives each through the LIVE app and observes its
effect, not just its echo, and pins the set of names the code contains, so a new `X-miLLM-*` name
with no exercise here fails.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

from miserver.api.approval_gate import AGENT_HEADER, APPROVAL_HEADER
from miserver.api.probe_request_meta import SESSION_HEADER
from miserver.api.provenance import MODEL_HEADER
from miserver.api.request_policy import IGNORED_FIELDS_HEADER, STRICT_HEADER
from miserver.api.routes.openai.load_policy import LEASE_HEADER, LOAD_POLICY_HEADER
from miserver.core.errors import ModelBusyError
from miserver.db.models.probe_session import ProbeSession
from tests.unit.api.test_load_policy import _post
from tests.unit.api.test_probe_session_headers import _all, chat, post, served  # noqa: F401
from tests.unit.approval_fixtures import harness, make_model, rows  # noqa: F401 - fixture
from tests.unit.f28_fixtures import clean_state, db  # noqa: F401 - fixtures
from tests.unit.lease_fixtures import build_service, clear_resident, install_registry, resident_inference
from tests.support.package_root import package_rel

REPO = Path(__file__).resolve().parents[4]

#: Every `X-miLLM-*` header the CODE sends or reads: string constants in `miserver/`, docstrings and
#: comments excluded (review M02/M03: a regex over source text kept passing while a comment still
#: spelled a renamed header). 22 on 2026-11-08, pinned, never derived.
#: The proposed `…-Probe-Stream` opt-in header was never served, and no longer appears anywhere in
#: the tree, so census no longer finds it as a DR-08 literal (review rev2, 2026-11-08).
FROZEN = frozenset({
    "X-miLLM-Agent", "X-miLLM-Approval", "X-miLLM-Attachment", "X-miLLM-Backend", "X-miLLM-Batch",
    "X-miLLM-Circuit-Rung", "X-miLLM-Constrained", "X-miLLM-Ignored-Fields", "X-miLLM-Lease",
    "X-miLLM-Load-Policy", "X-miLLM-Model", "X-miLLM-Origin",
    "X-miLLM-Probe-Verdicts", "X-miLLM-Scoring-Precision", "X-miLLM-Scoring-Precision-Reason",
    "X-miLLM-Seed", "X-miLLM-Session", "X-miLLM-Session-Id", "X-miLLM-Steering",
    "X-miLLM-Steering-Intensity", "X-miLLM-Strict", "X-miLLM-Transcript",
})
_HEADER = re.compile(r"^X-mi[A-Za-z]+-[A-Za-z][A-Za-z-]*$", re.IGNORECASE)
#: What a served chat completion carries on every generation (provenance.py, probe_request_meta.py).
RESPONSE = ("X-miLLM-Model", "X-miLLM-Backend", "X-miLLM-Steering", "X-miLLM-Session-Id",
            "X-miLLM-Transcript")


def _names_in_code() -> set[str]:
    """Header-shaped string constants in `miserver/` (any brand), docstrings excluded."""
    files = subprocess.run(["git", "ls-files", "--", package_rel()], cwd=REPO, capture_output=True,
                           text=True, check=True).stdout.split()
    found: set[str] = set()
    for rel in files:
        if not rel.endswith(".py"):
            continue
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        docs = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docs.add(id(body[0].value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in docs and _HEADER.match(node.value):
                found.add(node.value)
    return found


async def test_frozen_headers(served, harness):  # noqa: F811
    # The constants the code sends and reads, by value.
    assert (MODEL_HEADER, AGENT_HEADER, APPROVAL_HEADER, SESSION_HEADER, STRICT_HEADER,
            IGNORED_FIELDS_HEADER, LOAD_POLICY_HEADER, LEASE_HEADER) == (
        "X-miLLM-Model", "X-miLLM-Agent", "X-miLLM-Approval", "X-miLLM-Session", "X-miLLM-Strict",
        "X-miLLM-Ignored-Fields", "X-miLLM-Load-Policy", "X-miLLM-Lease")
    assert _names_in_code() == FROZEN

    # Response headers on a served generation, and -Session naming the recorded session.
    client, factory, _ = served
    r = post(client, chat(["w1"]), {"X-miLLM-Session": "dr08"})
    assert r.status_code == 200, r.text
    for name in RESPONSE:
        assert r.headers.get(name), f"{name} missing from a served chat completion"
    assert r.headers["X-miLLM-Model"].startswith('"tiny"')
    assert [s.client_session_id for s in await _all(factory, ProbeSession)] == ["dr08"]

    # -Strict refuses a field it would otherwise report in -Ignored-Fields.
    body = {**chat(["w1"]), "zz_dr08_unknown": 1}
    loose = post(client, body)
    assert loose.status_code == 200 and "zz_dr08_unknown" in loose.headers["X-miLLM-Ignored-Fields"]
    strict = post(client, body, {"X-miLLM-Strict": "true"})
    assert strict.status_code == 400, strict.text

    # -Agent turns a gated action into 202 + -Approval: pending, and the action does not run.
    h = harness
    h.holder["row"] = row = await make_model(h.db)
    gated = h.client.post(f"/api/models/{row.id}/unload", headers={"X-miLLM-Agent": "agent:dr08"})
    assert gated.status_code == 202 and gated.headers["X-miLLM-Approval"] == "pending", gated.text
    assert h.service.unload_model.await_count == 0 and len(await rows(h.db)) == 1

    # -Load-Policy and -Lease reach the load decision (the real ModelService over a fake registry,
    # built here, after the served model's checks: its resident "m1" would hide "tiny").
    lc, service = _load_env()
    refused = _post(lc, "chat", headers={"X-miLLM-Load-Policy": "refuse"})
    assert refused.status_code == 409 and service.load_model_and_wait.await_count == 0
    service.load_model_and_wait.side_effect = ModelBusyError("stop", details={})
    _post(lc, "chat", headers={"X-miLLM-Load-Policy": "auto", "X-miLLM-Lease": "dr08-lease"})
    service.load_model_and_wait.assert_awaited_once_with(2, lease_id="dr08-lease")
    clear_resident()


def _load_env():
    """`tests/unit/api/test_load_policy.py::env`, built inline."""
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    from miserver.api.dependencies import get_inference_service, get_model_service
    from miserver.main import create_app

    clear_resident()
    install_registry()
    service, _repo = build_service()
    service.load_model_and_wait = AsyncMock(wraps=service.load_model_and_wait)
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: service
    app.dependency_overrides[get_inference_service] = resident_inference
    return TestClient(app), service
