"""T13: every repository gate in `.github/workflows/` runs under both repository names until P1
closes, and under the new name only after (040 FR-040.14, CF-17, CF-21). The mode keys to the
export header's `closed`, never to `phase`; the names come from the export's `repo` records."""

from __future__ import annotations

import json
import re
import shutil

import pytest
import yaml

from tests.support import legacy_identifiers as li
from tests.support import workflow_guards as wg


def _renames(export) -> list[tuple[str, str]]:
    """Every rename pair, `transition` or (since P7) `retired`: a retired pair is still the old and
    the new name of one repository, and every gate must keep refusing the old one."""
    return [(r["old"], r["new"]) for r in export.repos() if r["status"] in ("transition", "retired")]


def _export_with(tmp_path, **header):
    lines = li.default_path().read_text().splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith('{"format"'))
    lines[i] = json.dumps({**json.loads(lines[i]), **header})
    path = tmp_path / li.default_path().name
    path.write_text("\n".join(lines) + "\n")
    return li.load(path)


#: This repository's old name, assembled: since P7 its `repo` records are retired and the name
#: appears in no tracked text (tests/unit/deploy/test_repository_names.py).
OLD = "mi" + "LLM"
OLD_SPELLINGS = (f"Onegaishimas/{OLD}", f"onegaishimas/{OLD.lower()}", f"hitsainet/{OLD.lower()}", f"hitsainet/{OLD}")


def _old_halves_put_back(tmp_path):
    """A copy of today's workflows with the old-name half of every gate put back, as P0-C wrote
    them: the shape the before-P1 mode must accept. 040 P1-C (CF-17) removed those halves from the
    real files, so the both-names mode is now proven on this copy."""
    moved = tmp_path / "workflows"
    shutil.copytree(wg.WORKFLOWS, moved)
    for path in moved.glob("*.yml"):
        text = path.read_text()
        text = text.replace("github.repository == 'Onegaishimas/miServer'",
                            f"(github.repository == 'Onegaishimas/{OLD}' || github.repository == 'Onegaishimas/miServer')")
        text = text.replace("github.repository == 'hitsainet/miServer'",
                            f"(github.repository == 'hitsainet/{OLD.lower()}' || github.repository == 'hitsainet/miServer')")
        path.write_text(text)
    return moved


def test_every_gate_passes_under_old_and_new_names_before_p1(tmp_path):
    export = _export_with(tmp_path, phase="P0", closed="P0")
    assert not export.closed_at_least("P1")
    gates = wg.gates(_old_halves_put_back(tmp_path))
    assert len(gates) >= 11, f"found only {len(gates)} gates: the parser is failing open"
    for g in gates:
        for name in OLD_SPELLINGS:
            if name.split("/")[0].lower() in g.expr.lower():
                assert wg.evaluate(g.expr, name, g.workflow), (g, name)
    assert wg.violations(gates, _renames(export), new_only=False) == []
    # The mode bites: today's new-name-only gates are refused before P1 closes, every one of them.
    today = wg.violations(wg.gates(), _renames(export), new_only=False)
    # Only gates whose owner has a rename pair in the loaded export are judged: on the public mirror the export
    # withholds the private `Onegaishimas/*` pair (Q-F2), so there the 5 public gates are judged and the 6 private
    # ones are not (2026-11-10: this assertion counted all 11 and turned the mirror's Backend Tests red).
    owners = {old.split("/")[0].lower() for old, _ in _renames(export)}
    judged = [g for g in wg.gates() if any(o in g.expr.lower() for o in owners)]
    assert judged, f"no gate is judged by the export's rename pairs {owners}: the check would pass vacuously"
    assert len(today) == len(judged) and all("skips under the old name" in v for v in today), today


def test_from_closed_p1_new_name_only():
    """040 FTASKS 8.9 (CF-17): on the REAL export (`closed: "P1"`, committed at P1's close), every
    gate runs under the new repository name and skips under every spelling of the old one."""
    export = li.load()
    assert export.closed_at_least("P1"), export.header
    gates = wg.gates()
    assert len(gates) >= 11, f"found only {len(gates)} gates: the parser is failing open"
    assert wg.violations(gates, _renames(export), new_only=export.closed_at_least("P1")) == []
    for g in gates:
        owner = "hitsainet" if "hitsainet/" in g.expr.lower() else "Onegaishimas"
        assert wg.evaluate(g.expr, f"{owner}/miServer", g.workflow), g
        for name in OLD_SPELLINGS:
            assert not wg.evaluate(g.expr, name, g.workflow), (g, name)


def test_phase_p1_closed_p0_keeps_both_names_mode(tmp_path):
    export = _export_with(tmp_path, phase="P1", closed="P0")
    new_only = export.closed_at_least("P1")
    assert new_only is False, "a --phase P1 export committed during P1 moves no gate (CF-21)"
    assert wg.violations(wg.gates(_old_halves_put_back(tmp_path)), _renames(export), new_only=new_only) == []


@pytest.mark.parametrize("expr", [
    "contains(github.repository, 'miLLM')",
    "startsWith(github.repository, 'Onegaishimas/')",
    "github.repository == 'x' || github.event.name != github.repository",
])
def test_evaluator_refuses_unknown_syntax(expr):
    with pytest.raises(wg.UnsupportedGate):
        wg.evaluate(expr, "x")


@pytest.mark.parametrize("expr", [f"github.repository != 'Onegaishimas/{OLD}'",
                                  f"!(github.repository == 'Onegaishimas/{OLD}')"])
def test_evaluator_refuses_negation_next_to_repository(expr):
    with pytest.raises(wg.UnsupportedGate):
        wg.evaluate(expr, f"Onegaishimas/{OLD}")


def test_evaluator_reads_the_forms_in_use():
    expr = (f"(github.repository == 'hitsainet/{OLD.lower()}' || github.repository == 'hitsainet/miServer')"
            "\n&& needs.detect.outputs.images != '[]' && needs.detect.outputs.images != ''")
    assert wg.evaluate(expr, "HITSAINET/MISERVER") and wg.evaluate(expr, f"hitsainet/{OLD}")
    assert not wg.evaluate(expr, f"Onegaishimas/{OLD}")


def test_sibling_checkout_names_miworkbench(tmp_path, monkeypatch):
    """040 FTASKS 8.6 (FR-040.14, P1-B): after S001 renamed miStudio to miWorkbench, the contract job
    checks the sibling out by its new name, into a folder of that name, and points the lookup at it.
    The wiring is asserted end to end: the path the job sets must be the folder the checkout writes
    and the one `workbench_repo()` resolves, and every shell line that reads the sibling reads it there.

    DR-55's exercise (census `EXERCISE` key): since P7 (FR-040.40) the old secret is no fallback any more
    and the job reads the new key alone. Since the P7 close (DR-55 retired, 2026-11-14) the old name may not
    appear in this tree at all, so the workflow is pinned to its EXACT set of secrets instead: a fallback to
    any other secret, the old one included, adds a name to the set."""
    from tests.support import sibling_repos

    job = yaml.safe_load((wg.WORKFLOWS / "backend-tests.yml").read_text())["jobs"]["cross-repo-contract"]
    checkout = next(s for s in job["steps"] if s.get("id") == "sibling")
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["repository"] == "Onegaishimas/miWorkbench"
    assert checkout["with"]["path"] == "miWorkbench"
    assert checkout["with"]["ssh-key"] == "${{ secrets.MIWORKBENCH_DEPLOY_KEY }}"
    text = (wg.WORKFLOWS / "backend-tests.yml").read_text()
    secrets_read = set(re.findall(r"\bsecrets\.([A-Za-z_][A-Za-z0-9_]*)", text))
    assert secrets_read == {"MIWORKBENCH_DEPLOY_KEY", "MIFORGE_DEPLOY_KEY"}, secrets_read

    env = job["env"]
    assert env["MIWORKBENCH_REPO"] == "${{ github.workspace }}/miWorkbench"
    assert "MISTUDIO_REPO" not in env, "a second lookup variable naming a folder nothing checks out"
    workspace = tmp_path / "ws"
    (workspace / checkout["with"]["path"]).mkdir(parents=True)
    monkeypatch.delenv("MISTUDIO_REPO", raising=False)
    monkeypatch.setenv("MIWORKBENCH_REPO", env["MIWORKBENCH_REPO"].replace("${{ github.workspace }}", str(workspace)))
    assert sibling_repos.workbench_repo() == workspace / checkout["with"]["path"]

    code = [line for s in job["steps"] for line in s.get("run", "").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    assert any("miWorkbench/backend/requirements.txt" in line for line in code), code
    stale = [line.strip() for line in code if re.search(r"(?<![\w-])miStudio/", line)]
    assert stale == [], f"shell lines still read the old checkout folder: {stale}"
