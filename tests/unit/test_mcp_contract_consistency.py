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
