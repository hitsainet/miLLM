"""040 CF-9 (T26 service half): re-importing a probe under a name that exists, with
`on_conflict=fail`, answers 422 `VALIDATION_ERROR` with `details.code == "NAME_CONFLICT"` and the
name, through the live route. The P2 cutover's re-import check requires exactly this answer."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.api.test_probe_event_filters import client  # noqa: F401 - fixture
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

DOC = json.loads((Path(__file__).resolve().parents[2] / "fixtures" / "lfm2_probe_definition.json").read_text())


def test_reimport_with_on_conflict_fail_answers_name_conflict(client, monkeypatch, tmp_path):  # noqa: F811
    from miserver.core.config import settings

    monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path))
    first = client.post("/api/probes/import", json=DOC)
    assert first.status_code == 200, first.text
    again = client.post("/api/probes/import?on_conflict=fail", json=DOC)
    assert again.status_code == 422, again.text
    error = again.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["details"]["code"] == "NAME_CONFLICT"
    assert error["details"]["name"] == DOC["name"]
