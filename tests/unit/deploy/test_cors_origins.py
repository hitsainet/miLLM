"""T21 (040 FR-040.25; R20) and its P7 inverse (FR-040.40): from P2 to P7 the backend's `CORS_ORIGINS`
kept every old origin beside its new twin; since P7 it holds the new origins only.

Read parsed from `k8s/base/backend.yaml`. The old origins are those whose host has an old brand; the
export's `host` records are all `retired`, so none may remain, and each retired host's new twin must
(a browser on the new address can still call the API). No public `k8s-*.hitsai.net` name exists for
the new brand (R20): the public name reaches the cluster through the tunnel.

Controls: NC-79 (remove `http://k8s-miserver.hitsai.local`), adding `http://k8s-miserver.hitsai.net`,
or putting back `http://k8s-millm.hitsai.local` → red.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from tests.support import legacy_identifiers as li

ROOT = Path(__file__).resolve().parents[3]
OLD_BRAND = re.compile(r"(?i)mi(llm|studio)")


def _origins() -> list[str]:
    docs = [d for d in yaml.safe_load_all((ROOT / "k8s/base/backend.yaml").read_text()) if d]
    (deploy,) = [d for d in docs if d["kind"] == "Deployment"]
    (backend,) = [c for c in deploy["spec"]["template"]["spec"]["containers"] if c["name"] == "backend"]
    (value,) = [e["value"] for e in backend["env"] if e["name"] == "CORS_ORIGINS"]
    return [o.strip() for o in value.split(",") if o.strip()]


def test_no_old_origin_after_p7():
    origins = _origins()
    old = [o for o in origins if OLD_BRAND.search(urlsplit(o).hostname or "")]
    assert not old, f"old-brand origins are retired at P7 (FR-040.40): {old}"


def test_each_retired_host_keeps_its_new_twin():
    hosts = {urlsplit(o).hostname for o in _origins()}
    records = [r for r in li.load().hosts() if r["status"] == "retired" and r["observed"]
               and not r["old"].startswith("millm-backend.")]  # in-cluster names are not browser origins
    assert records, "no retired browser host in the export: the check would pass vacuously"
    missing = sorted({r["new"] for r in records} - hosts - {"miserver.hitsai.local"})
    assert not missing, f"the new twin of a retired origin is missing: {missing}"
    assert "http://localhost:3000" in _origins()


def test_no_k8s_miserver_net_origin():
    bad = [o for o in _origins() if re.fullmatch(r"k8s-[a-z0-9-]+\.hitsai\.net", urlsplit(o).hostname or "")
           and not OLD_BRAND.search(urlsplit(o).hostname or "")]
    assert not bad, f"no public k8s-*.hitsai.net name exists for the new brand (R20): {bad}"
