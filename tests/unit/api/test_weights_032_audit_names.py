"""D-42 for Feature 32 (review RX F6): one audit operation name per operation, everywhere.

A token refusal is audited by `require_weights_identity` under the ROUTE's name; the service
audits its own accepted and refused lines. Before D-42 the same save logged `save_checkpoint`
(the route function's name) for a token refusal and `save` from the service. Both now read
`weights_save` (and `weights_load`), so one grep finds one operation.

MUTATION CONTROLS (FX-B): the route's `name="weights_save"` removed, or the service constant
changed back to `"save"` -> red here.
"""

from __future__ import annotations

from structlog.testing import capture_logs

from millm.services import weights_checkpoint_service as svc
from tests.unit.weights_e2e_fixtures import (  # noqa: F401 - flow is a fixture
    OPS,
    ROW_ID,
    attach,
    flow,
    holder,
    manifest,
)


def _audits(logs, operation=None):
    lines = [e for e in logs if e["event"] == "weights_audit"]
    return [e for e in lines if operation is None or e["operation"] == operation]


def test_the_live_routes_carry_the_audit_names(flow):
    """Read from the live app's router, not the module: the name a token refusal is audited
    under is the registered route's."""
    assert svc.AUDIT_SAVE == "weights_save" and svc.AUDIT_LOAD == "weights_load"
    assert flow.app.url_path_for("weights_save") == "/api/weights/save"
    assert flow.app.url_path_for("weights_load") == "/api/weights/load"
    assert flow.app.url_path_for("weights_lineage") == "/api/weights/lineage"


async def test_token_refusal_and_service_lines_share_one_name(flow):
    async with flow.client() as client:
        with capture_logs() as logs:
            r = await client.post("/api/weights/save", json={"manifest": manifest()})
            assert r.status_code == 401, r.text
            r = await client.post("/api/weights/load", json={"model_id": ROW_ID})
            assert r.status_code == 401, r.text
        assert [(e["operation"], e["code"]) for e in _audits(logs)] == [
            ("weights_save", "WEIGHTS_UNAUTHORIZED"), ("weights_load", "WEIGHTS_UNAUTHORIZED")]

        attachment_id = await attach(client)
        with capture_logs() as logs:
            saved = await client.post("/api/weights/save", json={"manifest": manifest()},
                                      headers=holder(attachment_id))
            assert saved.status_code == 201, saved.text
            refused = await client.post("/api/weights/save", json={"manifest": manifest()},
                                        headers=holder(attachment_id))
            assert refused.status_code == 409, refused.text  # CHECKPOINT_EXISTS
        assert [(e["operation"], e["outcome"]) for e in _audits(logs)] == [
            ("weights_save", "accepted"), ("weights_save", "refused")]
        assert (await client.post("/api/weights/detach",
                                  headers=holder(attachment_id))).status_code == 200

        with capture_logs() as logs:
            loaded = await client.post("/api/weights/load", json={"model_id": ROW_ID},
                                       headers=OPS)
            assert loaded.status_code == 200, loaded.text
        assert [(e["operation"], e["outcome"]) for e in _audits(logs)] == [
            ("weights_load", "accepted")]
