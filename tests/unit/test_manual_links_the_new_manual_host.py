"""ENH-009 P0 (S001 0.89-0.94, 2026-11-09): miStudio's manual answers on its own domain.

`hitsainet.github.io/miStudio/...` now redirects to `docs-miworkbench.hitsai.net`, and the address will stop
being the miStudio repo's once P1 renames it. miServer's manual links the manual by its new host; this keeps
an old link from coming back. (miDataworks holds the same rule in `test_no_link_to_the_old_pages_path`.)
"""

from pathlib import Path

MANUAL = Path(__file__).resolve().parents[2] / "manual"
OLD = "hitsainet.github.io/miStudio"


def test_the_manual_links_the_miworkbench_manual_by_its_new_host() -> None:
    found = [
        str(p.relative_to(MANUAL))
        for p in MANUAL.rglob("*")
        if p.is_file()
        and "node_modules" not in p.parts
        and "build" not in p.relative_to(MANUAL).parts[:1]
        and p.suffix in {".md", ".mdx", ".ts", ".tsx", ".js"}
        and OLD in p.read_text(errors="ignore")
    ]
    assert found == [], found


def test_the_scan_reaches_the_page_that_carried_the_link() -> None:
    """The scan is not vacuous: the page that held the old link is inside it, and links the new host."""
    page = MANUAL / "docs" / "features" / "mcp-circuits.md"
    assert page.is_file()
    assert "https://docs-miworkbench.hitsai.net/advanced/mcp-client-setup" in page.read_text()
