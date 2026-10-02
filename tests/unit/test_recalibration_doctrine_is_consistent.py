"""The bar-vs-detector boundary says the same thing in all five places it is written.

⚠ **WRITTEN BECAUSE FIVE COPIES OF A NORMATIVE SENTENCE EXISTED WITH NOTHING PINNING THEM.**

`on_conflict` has no `replace`, and the reason is stated — in identical words — in
`millm/services/probe_service.py`, `millm/api/routes/management/probes.py`, `docs/mcp-contract.md`,
`manual/docs/features/probe-monitors.md` and `0xcc/prds/024_FPRD|Probe_Monitor_Runtime.md`. Nothing
asserted that, so an amendment landing in four of the five would leave one copy stating the old
rule as current — and this estate has paid for exactly that shape: a CLAUDE.md phase entry
contradicted another entry 38 rows above it, and the contradiction was repeated back to the
operator as an open risk.

The recalibration carve-out is the first amendment to that doctrine, which makes it the right
moment to add the guard rather than the fifth time somebody has to remember.

⚠ AND THE GUARD MUST NOT FAIL OPEN. A source scrape that stops matching asserts nothing — three
separate arcs here have shipped one. So every path is checked to exist and to contain a known
anchor before the agreement itself is asserted.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Where the doctrine is written. All five, by name, because a glob would quietly stop covering a
#: file that moved.
#:
#: ⚠ SPLIT BY WHETHER THE PUBLIC MIRROR KEEPS THEM, AND THE SPLIT IS NOT COSMETIC. `sync-to-clean`
#: does `rm -rf 0xcc/` and strips everything under `docs/` except `schemas/` — and the MIRROR is
#: where the Docker image is built. The first version of this file asserted that at least four of
#: the five sites were on disk, which is true in the source repo and FALSE on the mirror, where
#: exactly three are. It passed locally, failed the mirror's Backend Tests, and the image build
#: refused to publish over a red suite — so a merge to main produced no backend image at all.
#: `test_probe_route_surface.py`'s own docstring records this exact trap for the FPRD; the
#: contract is the second file it applies to, and this file learned it the hard way.
ALWAYS_PRESENT = (
    "millm/services/probe_service.py",
    "millm/api/routes/management/probes.py",
    "manual/docs/features/probe-monitors.md",
)

#: Present only in the source repo. Each is SKIPPED LOUDLY when absent rather than deleted,
#: because the agreement genuinely cannot be checked without the file — and saying "unverified
#: here" is the honest description, not "fine".
SOURCE_ONLY = (
    "docs/mcp-contract.md",
    "0xcc/prds/024_FPRD|Probe_Monitor_Runtime.md",
)

SITES = ALWAYS_PRESENT + SOURCE_ONLY

#: The headline of the carve-out. Present in every site or none.
HEADLINE = "MOVING A BAR IS NOT REPLACING A DETECTOR."

#: The sentences that carry the rule. Each must appear wherever the headline does, because the
#: headline alone is a slogan and these are the parts a reader acts on.
CLAIMS = (
    "the event row ALREADY records the bar it was judged against",
    "The first may never change in place under a probe id",
    "`on_conflict` remains `rename|fail`",
)

#: The original refusal, as short a phrase as still names it. Longer anchors straddle a line
#: break in at least one site, and `_text` collapses whitespace precisely so they need not.


#: The original refusal, which the carve-out must NOT have removed anywhere.
STILL_REFUSED = "detectors as one"


def _text(rel: str) -> str:
    """The site's prose with whitespace collapsed.

    ⚠ COLLAPSED BECAUSE THE DOCTRINE IS PROSE AND EVERY SITE WRAPS IT DIFFERENTLY. The same
    sentence is a comment at 96 columns in one file, a markdown blockquote in another and a
    docstring indented four spaces in a third, so a raw substring match fails on the line break
    rather than on the content — which is a guard failing for a reason that has nothing to do
    with what it guards. Both of this file's first two failures were exactly that.
    """
    path = ROOT / rel
    if not path.exists():
        # ⚠ A MISSING *ALWAYS_PRESENT* SITE IS A FAILURE, NOT A SKIP. Only the source-only files
        # may be absent; if one of the three the mirror keeps has gone missing, this guard would
        # otherwise skip its way to green over a doctrine nobody states any more.
        assert rel in SOURCE_ONLY, (
            f"{rel} is listed as always present and is not on disk. Either it moved — fix the "
            f"path rather than letting this guard skip — or the doctrine has been deleted from a "
            f"site that ships."
        )
        pytest.skip(
            f"{rel} is not in this checkout — `sync-to-clean` does `rm -rf 0xcc/` and strips "
            "everything under `docs/` except `schemas/`, and the mirror is where images are "
            "built. The three sites that DO ship are still asserted."
        )
    # Strip each line's own leading marker BEFORE collapsing. A markdown blockquote prefixes
    # every line with "> " and a python comment with "# ", so collapsing alone leaves those
    # markers as tokens INSIDE the sentence — "the event row > ALREADY records…" — which is the
    # same failure as the line break, one step along.
    lines = [
        re.sub(r"^\s*(?:>\s?|#\s?)+", "", line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    return " ".join(" ".join(lines).split())


class TestTheGuardCanSeeItsOwnInputs:
    """A scrape that matches nothing passes and proves nothing."""

    def test_every_shipping_site_is_on_disk(self):
        """The three the mirror keeps must all be here, in EVERY checkout.

        This is what stops the guard degrading to nothing: the source-only files may be skipped,
        but if the shipping ones can be skipped too then a doctrine deleted everywhere passes.
        """
        missing = [rel for rel in ALWAYS_PRESENT if not (ROOT / rel).exists()]
        assert not missing, (
            f"{missing} are listed as always present and are not on disk — the paths are stale "
            f"and this guard would assert almost nothing"
        )

    def test_the_source_only_files_are_the_ones_the_MIRROR_strips(self):
        """⚠ PINNED AGAINST THE WORKFLOW THAT STRIPS THEM, so the two cannot drift.

        If `sync-to-clean` stops stripping a path, or starts stripping another, the split above is
        wrong and this guard would either skip something that ships or fail on something that does
        not. Read from the workflow rather than remembered.
        """
        workflow = ROOT / ".github/workflows/sync-to-clean.yml"
        if not workflow.exists():
            pytest.skip("the sync workflow is itself stripped from the mirror")
        text = workflow.read_text(encoding="utf-8")
        assert "rm -rf 0xcc/" in text, (
            "the mirror no longer strips 0xcc/ — the FPRD now ships and should move to "
            "ALWAYS_PRESENT"
        )
        assert "find docs/ -mindepth 1 -maxdepth 1 ! -name schemas" in text, (
            "the mirror's docs/ handling changed — re-derive which contract files ship"
        )
        for rel in ALWAYS_PRESENT:
            top = rel.split("/", 1)[0]
            assert f"rm -rf {top}/" not in text, (
                f"{rel} is listed as always present but the mirror strips {top}/"
            )

    @pytest.mark.parametrize("rel", SITES)
    def test_each_site_still_carries_the_ORIGINAL_refusal(self, rel: str):
        """The carve-out permits moving a bar. It must not have softened the import refusal
        anywhere — and a site that lost that sentence is a site somebody rewrote wholesale."""
        assert STILL_REFUSED in _text(rel), (
            f"{rel} no longer states why `replace` is refused on import. Permitting a bar to "
            f"move is not permitting a detector to be replaced, and if that sentence is gone "
            f"from a site, the two have been conflated there"
        )


class TestTheCarveOutIsWrittenInEveryPlaceOrNone:
    @pytest.mark.parametrize("rel", SITES)
    def test_the_headline_is_present(self, rel: str):
        assert HEADLINE in _text(rel), (
            f"{rel} does not carry the bar-vs-detector boundary. Four of five is how a reader "
            f"ends up acting on the version that was not amended"
        )

    @pytest.mark.parametrize("rel", SITES)
    @pytest.mark.parametrize("claim", CLAIMS)
    def test_each_load_bearing_claim_travels_with_it(self, rel: str, claim: str):
        text = _text(rel)
        if HEADLINE not in text:
            pytest.fail(f"{rel} is missing the headline; the companion test says so too")
        assert claim in text, (
            f"{rel} carries the headline without {claim!r}. The headline alone is a slogan — "
            f"these are the parts a reader acts on"
        )

    def test_the_route_that_the_doctrine_NAMES_is_the_one_that_is_served(self):
        """⚠ THE DOCTRINE POINTS AT A PATH, AND A DOCUMENT NAMING A ROUTE NOBODY SERVES IS THE
        DEFECT THIS ESTATE IS NAMED FOR. Read the built app, not the decorator."""
        from millm.main import create_app

        promised = "POST /api/probes/{probe_id}/recalibrate"
        for rel in SITES:
            text = _text(rel)
            if HEADLINE in text:
                assert "/api/probes/{probe_id}/recalibrate" in text, (
                    f"{rel} states the rule without naming the route that implements it"
                )
        served = set(create_app().openapi()["paths"].keys())
        assert "/api/probes/{probe_id}/recalibrate" in served, (
            f"the doctrine names {promised} and the app does not serve it"
        )
