"""T12: the sibling lookup and the require switch (040 FR-040.13, TPD-18), and T33's P7 inverse
(FR-040.40): since P7 only the new spellings are honoured, plus the suite-wide
`MISTUDIO_REQUIRE_CROSS_REPO_CHECKS`, which is the suite's own name and stays (Q11, R17)."""

from __future__ import annotations

import pytest

from tests.support import sibling_repos

NEW_SWITCH = "MISERVER_REQUIRE_CROSS_REPO_CHECKS"
SUITE_SWITCH = "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS"
SWITCHES = (NEW_SWITCH, SUITE_SWITCH)
#: The spellings P7 retired.
OLD_SWITCH = "MILLM_REQUIRE_CROSS_REPO_CHECKS"
OLD_LOOKUP = "MISTUDIO_REPO"


@pytest.fixture
def layout(tmp_path, monkeypatch):
    root = tmp_path / "suite" / "miServer"
    root.mkdir(parents=True)
    monkeypatch.setattr(sibling_repos, "REPO_ROOT", root)
    for name in ("MIWORKBENCH_REPO", OLD_LOOKUP, OLD_SWITCH, *SWITCHES):
        monkeypatch.delenv(name, raising=False)
    return root.parent


def test_lookup_order_on_a_temp_layout(layout, monkeypatch):
    assert sibling_repos.workbench_repo() == layout / "miWorkbench"
    monkeypatch.setenv("MIWORKBENCH_REPO", "/elsewhere/workbench")
    assert str(sibling_repos.workbench_repo()) == "/elsewhere/workbench"


def test_old_spellings_no_longer_honoured(layout, monkeypatch):
    """T33: the old lookup variable, the old folder and the old switch spelling are all ignored."""
    (layout / "miStudio").mkdir()  # the workbench's old folder name; after P5 it is the SUITE's folder
    monkeypatch.setenv(OLD_LOOKUP, "/elsewhere/studio")
    assert sibling_repos.workbench_repo() == layout / "miWorkbench"
    assert not hasattr(sibling_repos, "mistudio_repo"), "the old function name is gone (FR-040.40)"

    monkeypatch.setenv(OLD_SWITCH, "1")
    assert not sibling_repos.cross_repo_checks_required(), f"{OLD_SWITCH} is retired"
    with pytest.raises(pytest.skip.Exception):
        sibling_repos.require_or_skip("sibling missing")

    monkeypatch.setenv(SUITE_SWITCH, "1")
    assert sibling_repos.cross_repo_checks_required(), "the suite-wide switch stays after P7 (Q11)"


@pytest.mark.parametrize("switch", SWITCHES)
def test_each_switch_spelling_alone_makes_a_missing_sibling_fail(layout, monkeypatch, switch):
    with pytest.raises(pytest.skip.Exception):
        sibling_repos.require_or_skip("sibling missing")
    monkeypatch.setenv(switch, "1")
    assert sibling_repos.cross_repo_checks_required()
    with pytest.raises(pytest.fail.Exception, match="sibling missing"):
        sibling_repos.require_or_skip("sibling missing")
