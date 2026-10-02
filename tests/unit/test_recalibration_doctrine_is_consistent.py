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
SITES = (
    "millm/services/probe_service.py",
    "millm/api/routes/management/probes.py",
    "docs/mcp-contract.md",
    "manual/docs/features/probe-monitors.md",
    "0xcc/prds/024_FPRD|Probe_Monitor_Runtime.md",
)

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
        pytest.skip(
            f"{rel} is not in this checkout — `0xcc/` is stripped from the public mirror, which "
            "is where images are built. The other sites are still asserted."
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

    def test_every_site_exists_or_is_explicitly_skipped(self):
        found = [rel for rel in SITES if (ROOT / rel).exists()]
        assert len(found) >= 4, (
            f"only {len(found)} of the {len(SITES)} doctrine sites are on disk — the paths are "
            f"stale and this guard would assert almost nothing: {found}"
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
