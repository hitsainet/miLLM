"""Feature 20 task 5.4 — the contract must not claim a tool that does not exist.

`docs/mcp-contract.md` is normative for the MCP server that lives in the
miStudio repo. For an entire increment it carried `✅` marks that read as a
shipped tool surface while NO `millm_circuits` module existed — an agent
reading the contract would have called tools that were not registered.

This is the miLLM half of a reciprocal pair. It reads the SIBLING repo's
registry and checks the contract against it.

**It SKIPS LOUDLY when miStudio is not checked out**, naming what is missing.
A cross-repo guard that passes vacuously is worse than no guard: it reports
green for the one condition it exists to detect.
"""

import os
import re
import sys
from pathlib import Path

import pytest

CONTRACT = Path(__file__).resolve().parents[2] / "docs" / "mcp-contract.md"

#: Sibling checkout. Overridable so CI can point at a different layout rather
#: than silently skipping.
MISTUDIO = Path(
    os.environ.get(
        "MISTUDIO_REPO",
        str(Path(__file__).resolve().parents[3] / "miStudio"),
    )
)


def _registry() -> dict:
    """The miStudio MCP registry, or skip loudly."""
    backend = MISTUDIO / "backend"
    if not (backend / "src" / "mcp_server" / "tools" / "__init__.py").exists():
        pytest.skip(
            f"miStudio checkout not found at {MISTUDIO} — this guard checks the "
            "contract against the SIBLING repo's registry and cannot run "
            "without it. Set MISTUDIO_REPO to the checkout. (Skipping loudly "
            "rather than passing: a cross-repo guard that passes vacuously "
            "reports green for exactly the condition it exists to detect.)"
        )
    sys.path.insert(0, str(backend))
    try:
        from src.mcp_server.tools import MILLM_CATEGORY_MODULES  # noqa: PLC0415

        return MILLM_CATEGORY_MODULES
    except ImportError as exc:
        # miStudio's MCP deps (`mcp`) are not in THIS repo's venv. Skipping
        # loudly and naming the cause: "unverified" is not "verified clean",
        # and a green tick here would mean the opposite of what it says.
        pytest.skip(
            f"miStudio's MCP package is not importable from miLLM's venv "
            f"({exc}). The registry guard is UNVERIFIED, not verified-clean — "
            "run it from a checkout with miStudio's dependencies installed."
        )
    finally:
        sys.path.remove(str(backend))


class TestContractMatchesTheRegistry:
    def test_the_contract_file_exists(self):
        assert CONTRACT.exists(), f"contract missing at {CONTRACT}"

    def test_millm_circuits_is_registered(self):
        """The defect this feature closed: the contract described a tool
        surface that did not exist."""
        assert "millm_circuits" in _registry(), (
            "docs/mcp-contract.md documents the millm_circuits category, and "
            "miStudio's registry does not contain it — the contract is "
            "claiming a tool surface an agent cannot reach"
        )

    def test_every_category_the_contract_names_is_registered(self):
        registry = _registry()
        # Category headings look like: ### `millm_circuits` (v1.1 — …)
        named = set(re.findall(r"^### `(millm_[a-z_]+)`", CONTRACT.read_text(),
                               re.MULTILINE))
        assert named, "no millm_* category headings found — has the format changed?"
        missing = named - set(registry)
        assert not missing, (
            f"the contract documents {sorted(missing)}, which miStudio does "
            "not register — an agent reading this would call tools that are "
            "not there"
        )

    def test_the_status_correction_is_RESOLVED_not_deleted(self):
        """It is the record of how a contract table read as shipped for an
        entire increment. Resolving it keeps the history; deleting it would
        erase the only evidence the mistake happened."""
        text = CONTRACT.read_text()
        assert "STATUS CORRECTION" in text, (
            "the status-correction block was deleted rather than resolved"
        )
        assert "RESOLVED" in text, "the correction is still open"

    def test_the_evidence_ladder_constants_agree(self):
        """Reciprocal parity: miLLM renders the rung phrases, miStudio's MCP
        tools transport them. A drift means an agent relays language this
        server never produced."""
        registry = _registry()
        assert "millm_circuits" in registry

        from millm.core.circuit_evidence import RUNG_LANGUAGE

        backend = MISTUDIO / "backend"
        sys.path.insert(0, str(backend))
        try:
            from src.schemas.evidence_ladder import (  # noqa: PLC0415
                RUNG_LANGUAGE as THEIRS,
            )
        except ImportError:
            pytest.skip(
                "miStudio's evidence_ladder is not importable from this "
                "checkout — parity unverified, NOT verified-clean"
            )
        finally:
            sys.path.remove(str(backend))

        assert dict(RUNG_LANGUAGE) == dict(THEIRS), (
            "the two repos disagree about the evidence-ladder phrasing, so an "
            "agent would relay a phrase this server never renders"
        )


class TestEveryRegisteredToolHasACorrectRow:
    """F20 R1-02/03. The guard above checks category HEADINGS. The defect that
    shipped — twice — was in the Status COLUMN.

    Round 1 found every row still reading `REST ✅ · MCP not registered` while
    the resolution block above declared the surface shipped. An agent reading
    top-down would bypass 16 registered tools and hand-roll HTTP, or refuse the
    task as unsupported. That is the identical failure mode F20 exists to
    abolish, INVERTED: last increment the marks over-claimed, this time they
    under-claimed.

    The reachability rule was applied rigorously to the miStudio registry and
    not at all to the contract table — the artifact that actually failed. These
    close it in BOTH directions: no row may claim a tool that is not
    registered, and no registered tool may be missing or mismarked.
    """

    def _rows(self) -> dict:
        """`{tool_name: status}` from the millm_circuits table."""
        text = CONTRACT.read_text()
        rows = {}
        for line in text.splitlines():
            if not line.startswith("| `millm_"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 3:
                continue
            for name in re.findall(r"`(millm_[a-z_]+)`", cells[0]):
                rows[name] = cells[-1]
        return rows

    def _registered_tools(self) -> set:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        registry = _registry()
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:
            pytest.skip(f"mcp package not importable from this venv ({exc})")

        gate = MagicMock()
        gate.check = AsyncMock(return_value=(True, None))
        mcp = FastMCP("contract-check")
        for module in registry["millm_circuits"]:
            module.register(mcp, MagicMock(), gate)
        return {t.name for t in asyncio.run(mcp.list_tools())}

    def test_the_row_extraction_works(self):
        """An empty dict passes every assertion below it."""
        rows = self._rows()
        assert len(rows) >= 14, (
            f"only parsed {len(rows)} tool rows — the table format changed and "
            "this guard is checking nothing"
        )

    def test_no_row_claims_a_tool_that_is_not_registered(self):
        registered = self._registered_tools()
        claimed = {
            name for name, status in self._rows().items()
            if "MCP ✅" in status
        }
        phantom = sorted(claimed - registered)
        assert not phantom, (
            f"the contract marks {phantom} as MCP-registered and miStudio does "
            "not register them — an agent would call tools that do not exist, "
            "which is the defect this feature was built to close"
        )

    def test_every_registered_tool_has_a_row_marked_MCP(self):
        rows = self._rows()
        registered = self._registered_tools()

        missing = sorted(t for t in registered if t not in rows)
        assert not missing, (
            f"{missing} ship as MCP tools and have NO contract row — an agent "
            "using the contract as its index of capability never learns they "
            "exist"
        )

        mismarked = sorted(
            t for t in registered if "MCP ✅" not in rows[t]
        )
        assert not mismarked, (
            f"{mismarked} are registered and their rows say otherwise "
            f"({[rows[t] for t in mismarked]}) — an agent reads the row, "
            "bypasses the tool, and hand-rolls HTTP or refuses the task"
        )

    def test_the_recovery_tool_is_documented(self):
        """§4a-sexies tells an agent to use the claim-release endpoint when a
        refusal names a circuit that is not running. A prescribed remedy whose
        tool is absent from the table is a dead end."""
        rows = self._rows()
        assert "millm_release_circuit_claims" in rows, (
            "the contract prescribes claim release as the recovery path and "
            "does not list the tool that performs it"
        )
