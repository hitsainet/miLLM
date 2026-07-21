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


def _sidebar_ids() -> set[str]:
    """Every quoted string in sidebars.ts that names a doc.

    Deliberately crude: it over-collects (category labels, type names), which
    can only make this guard MORE permissive, never falsely accusing. The
    failure it must not have is the other direction — missing a real entry and
    reporting a reachable page as orphaned.
    """
    text = SIDEBARS.read_text()
    return set(re.findall(r"['\"]([a-z0-9][a-z0-9/_-]*)['\"]", text))


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
        docs = _doc_ids()
        # Only ids that look like doc paths (contain a slash) — the crude
        # extraction also picks up bare words like 'category'.
        named = {s for s in _sidebar_ids() if "/" in s}
        missing = sorted(named - docs)
        assert not missing, (
            f"sidebars.ts names {missing}, which do not exist under "
            f"{DOCS}. The docs build will fail on these."
        )
