"""Every place that states miLLM's version states the SAME version.

⚠ REPORTED FROM THE RUNNING UI 2026-09-30: the sidebar read `v0.5.0` while the Settings page,
under a card headed "About the miLLM server", read `1.0.0`. There were FIVE sources and THREE
answers:

    VERSION (repo root)          0.5.0   ← what GET /api/health/version actually serves
    millm.__version__            0.5.1   ← what the OpenAPI document reports
    pyproject.toml               0.5.1
    admin-ui/package.json        0.5.0
    Sidebar.tsx APP_VERSION      0.5.0   (since ENH-004 read from admin-ui/package.json)
    SettingsPage.tsx             1.0.0   ← a literal in the markup, matching nothing at all

The Settings one is the instructive failure. It was never wired to anything, so it could not go
stale — it was simply never right, and it sat under a heading promising it described the server
while the server had been serving its real version at `/api/health/version` the whole time. A
number that cannot be wrong by drifting is not a reliable number; it is an unchecked claim.

The `__version__` / VERSION disagreement is the quieter one: `/api/health/version` prefers the
FILE and the OpenAPI document uses the CONSTANT, so one running server answered two versions
depending on which endpoint you asked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = "0.5.0"


def _read(path: Path) -> str:
    assert path.exists(), f"{path} is missing — this guard cannot see it, so it asserts nothing"
    return path.read_text()


class TestTheSourcesAgree:
    def test_the_VERSION_file(self):
        assert _read(ROOT / "VERSION").strip() == EXPECTED

    def test_the_package_constant(self):
        """⚠ `/api/health/version` prefers the FILE; the OpenAPI document uses THIS. When they
        differ, one server reports two versions."""
        from millm import __version__

        assert __version__ == EXPECTED
        assert __version__ == _read(ROOT / "VERSION").strip(), (
            "millm.__version__ and the VERSION file disagree, so GET /api/health/version and the "
            "OpenAPI document report different versions of the same process"
        )

    def test_pyproject(self):
        match = re.search(r'^version\s*=\s*"([^"]+)"', _read(ROOT / "pyproject.toml"), re.M)
        assert match, "no version in pyproject.toml — the guard is looking at the wrong shape"
        assert match.group(1) == EXPECTED

    def test_the_admin_ui_package(self):
        assert json.loads(_read(ROOT / "admin-ui" / "package.json"))["version"] == EXPECTED

    def test_the_sidebar_reads_the_package_version(self):
        """The sidebar used to state its own literal (`APP_VERSION = '0.5.0'`), tied to nothing.
        Since ENH-004 it shows package.json's version: both Vite configs define `__APP_VERSION__`
        from the package, `src/config/version.ts` exports it, and the sidebar imports that. The
        rendered value is checked in admin-ui (helpWiring.test.tsx: the footer shows
        `v<package.json version>`); this checks no literal crept back in."""
        ui = ROOT / "admin-ui"
        sidebar = _read(ui / "src" / "components" / "layout" / "Sidebar.tsx")
        assert "from '@/config/version'" in sidebar, "the sidebar no longer imports the app version"
        assert not re.search(r"APP_VERSION\s*=", sidebar), (
            "Sidebar.tsx assigns APP_VERSION again; a literal there drifts from package.json"
        )
        assert not re.findall(r"\d+\.\d+\.\d+", sidebar), "a version literal is back in Sidebar.tsx"
        assert re.search(r"APP_VERSION[^=]*=\s*__APP_VERSION__", _read(ui / "src" / "config" / "version.ts"))
        for config in ("vite.config.ts", "vitest.config.ts"):
            source = _read(ui / config)
            assert "__APP_VERSION__: JSON.stringify(APP_VERSION)" in source, config
            assert "'package.json'" in source and ".version" in source, config


class TestTheSettingsPageAsksTheServer:
    """⚠ ASSERTS ABSENCE. The defect was a literal in the markup, so a test that checks the page
    renders *a* version passes against the literal that started this.
    """

    @staticmethod
    def _page() -> str:
        return _read(ROOT / "admin-ui" / "src" / "pages" / "SettingsPage.tsx")

    def test_it_fetches_the_version_rather_than_stating_one(self):
        assert "/api/health/version" in self._page(), (
            'the Settings page no longer asks the server for its version; a card headed "About '
            'the miLLM server" must not carry a number the server did not supply'
        )

    def test_no_version_literal_survives_in_the_markup(self):
        page = self._page()
        literals = re.findall(r'>\s*(\d+\.\d+\.\d+)\s*<', page)
        assert not literals, (
            f"hard-coded version literal(s) {literals} in SettingsPage.tsx — this is exactly the "
            f"1.0.0 that matched nothing in the repo for the life of the page"
        )

    def test_an_unreported_version_is_not_invented(self):
        """`null` must render as a dash. Falling back to a plausible constant is how a page ends
        up confidently displaying a version nothing ever set."""
        assert "serverVersion ?? '—'" in self._page()


class TestTheGuardCanSeeAKnownTrueValue:
    """⚠ Source scans fail OPEN. If these paths move, every assertion above quietly stops
    checking anything, so prove the files are where the guard thinks they are."""

    @pytest.mark.parametrize(
        "relative",
        [
            "VERSION",
            "pyproject.toml",
            "admin-ui/package.json",
            "admin-ui/src/components/layout/Sidebar.tsx",
            "admin-ui/src/config/version.ts",
            "admin-ui/vite.config.ts",
            "admin-ui/vitest.config.ts",
            "admin-ui/src/pages/SettingsPage.tsx",
        ],
    )
    def test_the_file_exists(self, relative):
        assert (ROOT / relative).exists(), f"{relative} moved; the guard above is now inert"
