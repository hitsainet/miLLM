"""F20 R2-22. Every manual page must be reachable from the sidebar.

`features/circuits.md` shipped an entire increment's user documentation while
absent from `manual/sidebars.ts` — a page that exists, builds, and renders, and
that no reader can navigate to. Docusaurus does not warn: an orphaned doc is a
valid doc.

This is the reachability rule applied to documentation. A page nobody can reach
is not shipped, for exactly the same reason an unregistered MCP tool is not
shipped, and it failed to be caught for exactly the same reason: everything
that touched it touched it DIRECTLY, by path, never through the entry point a
real reader uses.
"""

import re
from pathlib import Path

import pytest

MANUAL = Path(__file__).resolve().parents[2] / "manual"
DOCS = MANUAL / "docs"
SIDEBARS = MANUAL / "sidebars.ts"

#: Pages Docusaurus routes without a sidebar entry.
#:
#: `intro` is the landing page. Anything else added here needs a REASON, not
#: just a desire to make this test pass — the whole point is that "it is not in
#: the sidebar" should be a decision, not an oversight.
EXEMPT = {"intro"}


def _doc_ids() -> set[str]:
    """Docusaurus ids: path under docs/, no extension."""
    return {
        str(p.relative_to(DOCS).with_suffix("")).replace("\\", "/")
        for p in DOCS.rglob("*.md")
    } | {
        str(p.relative_to(DOCS).with_suffix("")).replace("\\", "/")
        for p in DOCS.rglob("*.mdx")
    }


#: A quoted string preceded by `label:` / `type:` / `id:` is NOT a doc id.
_KEYED = re.compile(r"\b(?:label|type|id|className|description)\s*:\s*$")


def _sidebar_ids() -> set[str]:
    """Doc ids in sidebars.ts: bare quoted strings in an items array or the
    top-level list.

    F20 R3-01 rewrote this. The original took EVERY quoted string, reasoning
    that over-collecting could only make the guard more permissive and never
    falsely accuse. That reasoning was wrong in a way I proved by attack:
    over-collecting means a doc id can be "covered" by an unrelated quoted
    word elsewhere in the file.

    Demonstrated: delete `'troubleshooting'` from the sidebar — genuinely
    orphaning `troubleshooting.md` — and rename any category label to
    `'troubleshooting'`. The page is unreachable and the guard passes. A
    top-level page (no directory, so no slash in its id) is exactly the shape
    most vulnerable, and this repo has one.

    That is the same defect the guard exists to catch, inside the guard.
    """
    ids: set[str] = set()
    for line in SIDEBARS.read_text().splitlines():
        stripped = line.strip()
        # A doc entry is a whole line that is just a quoted string, optionally
        # comma-terminated: `'features/circuits',`
        m = re.fullmatch(r"['\"]([A-Za-z0-9][A-Za-z0-9/_-]*)['\"],?", stripped)
        if m:
            ids.add(m.group(1))
            continue
        # Reject anything that is the VALUE of a key (label:, type:, …) — the
        # over-collection that made the original foolable.
        if _KEYED.search(line.split(":")[0] + ":"):
            continue
    return ids


@pytest.mark.skipif(not DOCS.is_dir(), reason=f"no manual at {DOCS}")
class TestEveryPageIsReachable:
    def test_the_extraction_works(self):
        """An empty set on either side passes every assertion below it."""
        docs, sidebar = _doc_ids(), _sidebar_ids()
        assert len(docs) > 10, (
            f"only found {len(docs)} doc pages — the layout changed and this "
            "guard is checking nothing"
        )
        assert len(sidebar) > 10, (
            f"only parsed {len(sidebar)} sidebar ids — the format changed and "
            "this guard is checking nothing"
        )

    def test_no_page_is_orphaned(self):
        orphans = sorted(_doc_ids() - _sidebar_ids() - EXEMPT)
        assert not orphans, (
            f"{len(orphans)} manual page(s) are not reachable from the "
            f"sidebar: {orphans}\n\n"
            "The page renders and builds; no reader can navigate to it. Add "
            "it to manual/sidebars.ts, or to EXEMPT in this file WITH A "
            "REASON if it is deliberately unlisted."
        )

    def test_the_sidebar_does_not_name_a_missing_page(self):
        """The other direction: a sidebar entry pointing at a deleted page
        breaks the docs build, which is loud — but it breaks it at deploy
        time, and this is cheaper."""
        # R3-01: this used to filter on `"/" in s` to dodge the noise the old
        # crude extraction produced. That filter also skipped every TOP-LEVEL
        # page (no directory ⇒ no slash), so a sidebar entry naming a deleted
        # top-level doc passed. The extraction no longer produces noise, so
        # the filter is gone and top-level entries are checked too.
        missing = sorted(_sidebar_ids() - _doc_ids())
        assert not missing, (
            f"sidebars.ts names {missing}, which do not exist under "
            f"{DOCS}. The docs build will fail on these."
        )
