"""P10 additions (041 FR-041.10, TID §3.11): frozen values a careless package rename would break.

P8 renames the Python package `millm` to `miserver`. These values contain `millm` and are NOT the package:
they are names other systems store or match on, so they keep their spelling after the rename (register
DR-23, DR-02, DR-34). A blanket `millm` -> `miserver` substitution over the package's strings (041 M10)
must turn each test here red. Each goes through the real code path, not a constant.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from tests.support import capture_pre038_goldens as cap
from tests.support.package_root import REPO_ROOT
from tests.unit.api.test_return_sae_activations import ACT, CHAT, clean, live  # noqa: F401
from tests.unit.telemetry.conftest import _telemetry_off, flush, stub, telemetry_on  # noqa: F401


async def test_span_name_probe_evaluate(stub, telemetry_on):  # noqa: F811
    """DR-23: a scored chat through the real pipeline exports a span named exactly
    `millm.probe.evaluate`, under the `millm` scope, as a child of the request span."""
    telemetry_on()
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        assert h.post_chat(cap.BODY, {}).status_code == 200
        flush()
    spans = [s for s in stub.spans if s["name"] == "millm.probe.evaluate"]
    assert spans, sorted({s["name"] for s in stub.spans})
    assert set(stub.scopes) == {"millm"}, set(stub.scopes)


def test_rule_runner_contract_kind():
    """DR-02's rule-side row. The kind a stored code rule carries is the runner contract's `format`
    (vendored from miDataworks, `midataworks.runner/v1`); every shared runner case carries it and the
    server's own check accepts it. `millm.rule-runner/v1` - the value an admin-UI test fixture uses -
    is not a kind this server accepts: a rename must not start minting it either."""
    from miserver.rules import runner_contract as rc

    cases = json.loads((REPO_ROOT / "docs" / "schemas" / "runner-cases-v1.json").read_text(encoding="utf-8"))
    kinds = {doc["runner_contract"] for doc in cases["documents"].values()}
    assert len(cases["documents"]) >= 5 and kinds == {rc.contract_format()}, kinds

    def doc(kind: str) -> tuple[str, str]:
        body = {"runner_contract": kind, "source": "def f(text):\n    return 1\n",
                "functions": [{"name": "f"}], "params_schema": {}, "derived_from": None}
        raw = rc.canonical_json(body)
        return hashlib.sha256(raw).hexdigest(), raw.decode("utf-8")

    assert rc.parse_code_document(*doc(rc.contract_format())).functions == frozenset({"f"})
    with pytest.raises(rc.CodeDocumentRefused) as refused:
        rc.parse_code_document(*doc("millm.rule-runner/v1"))
    assert refused.value.code == "runner_contract_unknown"


def test_activations_stream_chunk_key(live):  # noqa: F811
    """DR-34: a streamed chat that asks for SAE activations carries them under the key `millm` in a
    chunk, exactly as Open WebUI and miWorkbench read them."""
    client, *_ = live
    with client.stream("POST", "/v1/chat/completions",
                       json={**CHAT, "stream": True, "return_sae_activations": ACT}) as r:
        assert r.status_code == 200
        chunks = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: {")]
    carrying = [c for c in chunks if "sae_activations" in c.get("millm", {})]
    assert carrying, [sorted(c) for c in chunks]
    assert not any("miserver" in c for c in chunks)
