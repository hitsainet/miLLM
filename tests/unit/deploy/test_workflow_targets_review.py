"""Review of 040 P0 (2026-11-08): what the workflows push to and check out with.

Controls W01, W02, W04 and W05 of the review survived the P0 suite:
* W01/W02: `sync-to-clean.yml`'s `MIRROR` could be set to another repository, or to the PRIVATE
  repository itself (`${{ github.repository }}`, a force-push of the orphan snapshot onto the private
  `main`), with every test green. 040 changed this line from a constant to a derived expression.
* W04: the miStudio checkout's deploy-key fallback could name a broader secret.
* W05: `fetch-depth: 2` (T34, CF-27) could drop back to 1; T34 then SKIPS in CI and the suite is green.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from tests.support import legacy_identifiers as li

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"
EXPECTED_MIRROR = "hitsainet/${{ github.event.repository.name }}"
EXPECTED_REMOTE = "https://x-access-token:${SYNC_TOKEN}@github.com/${MIRROR}.git"


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _on(doc: dict) -> dict:
    on = doc.get("on", doc.get(True))
    return on if isinstance(on, dict) else {}


def test_the_mirror_is_derived_from_this_repository_name_only():
    doc = _load("sync-to-clean.yml")
    # `github.event.repository` exists on a push event; another trigger could leave it empty.
    assert set(_on(doc)) == {"push"}, _on(doc)
    sync = doc["jobs"]["sync"]
    assert sync["env"]["MIRROR"] == EXPECTED_MIRROR

    # Resolve it for every repository the sync job may run in: each must land on a hitsainet
    # repository the export names (today's mirror, or its renamed self), never anywhere else.
    sources = re.findall(r"github\.repository == '([^']+)'", sync["if"])
    assert sources, "the sync job has no repository gate"
    known = {r[k].lower() for r in li.load().repos() for k in ("old", "new")}
    for source in sources:
        owner, name = source.split("/")
        assert owner == "Onegaishimas", source
        resolved = EXPECTED_MIRROR.replace("${{ github.event.repository.name }}", name)
        assert resolved.lower() in known, (source, resolved)
    assert "hitsainet/miserver" in known, "the mirror must be one of the targets"


def test_every_push_in_the_sync_job_goes_to_the_mirror():
    sync = _load("sync-to-clean.yml")["jobs"]["sync"]
    runs = "\n".join(s.get("run", "") for s in sync["steps"])
    remotes = re.findall(r'REMOTE="([^"]+)"', runs)
    assert remotes and set(remotes) == {EXPECTED_REMOTE}, remotes
    pushes = [line.strip() for line in runs.splitlines() if re.search(r"\bgit push\b", line)]
    assert pushes, "no push found: the scan is looking in the wrong place"
    for line in pushes:
        assert '"$REMOTE"' in line, line


def test_deploy_keys_are_deploy_keys():
    """Every `ssh-key:` in every workflow names deploy-key secrets only; the miStudio checkout's
    fallback is exactly the renamed sibling's key, then the old one."""
    seen = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for job in (yaml.safe_load(path.read_text()).get("jobs") or {}).values():
            for step in job.get("steps") or []:
                key = (step.get("with") or {}).get("ssh-key")
                if key is None:
                    continue
                seen.append((path.name, step.get("id"), key))
                secrets = re.findall(r"secrets\.([A-Z0-9_]+)", key)
                assert secrets and all(s.endswith("_DEPLOY_KEY") for s in secrets), (path.name, key)
    assert seen, "no ssh-key found: the scan is looking in the wrong place"
    sibling = [k for wf, sid, k in seen if wf == "backend-tests.yml" and sid == "sibling"]
    assert sibling == ["${{ secrets.MIWORKBENCH_DEPLOY_KEY }}"], "040 P7 removed the old secret's fallback"


def test_the_unit_shards_check_out_a_parent_commit():
    """T34 compares with HEAD~1 and SKIPS without it: the unit shards must fetch depth >= 2."""
    test_job = _load("backend-tests.yml")["jobs"]["test"]
    checkout = [s for s in test_job["steps"] if str(s.get("uses", "")).startswith("actions/checkout")]
    assert len(checkout) == 1
    assert int((checkout[0].get("with") or {}).get("fetch-depth", 1)) >= 2
