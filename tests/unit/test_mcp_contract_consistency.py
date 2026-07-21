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


#: F20 R1-14. Set this in an environment where the guard MUST run, and a
#: missing sibling repo becomes a FAILURE instead of a skip.
#:
#: The guard skipped in CI unconditionally — the workflow checks out only
#: miLLM, so `mcp` was never importable and the three tests that constitute the
#: entire cross-repo check could never execute. "Skips loudly" is true locally
#: and irrelevant in a pipeline: a loud skip nobody reads IS the vacuous green
#: it was built to prevent.
#:
#: An opt-in switch rather than always-fail, because a developer working in
#: miLLM alone should not be blocked by a sibling checkout they do not need.
REQUIRE_CROSS_REPO = os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1"


def _unavailable(reason: str):
    """Skip, or FAIL when this environment declared the guard mandatory."""
    if REQUIRE_CROSS_REPO:
        pytest.fail(
            f"MILLM_REQUIRE_CROSS_REPO_CHECKS=1 but the cross-repo guard "
            f"cannot run: {reason}"
        )
    pytest.skip(reason)


def _registry() -> dict:
    """The miStudio MCP registry, or skip loudly."""
    backend = MISTUDIO / "backend"
    if not (backend / "src" / "mcp_server" / "tools" / "__init__.py").exists():
        _unavailable(
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
        _unavailable(
            f"miStudio's MCP package is not importable from miLLM's venv "
            f"({exc}). The registry guard is UNVERIFIED, not verified-clean — "
            "run it from a checkout with miStudio's dependencies installed."
        )
    finally:
        sys.path.remove(str(backend))


#: The circuits table is located by this heading, not by scanning the whole
#: document — see `_rows` for the three defects that came from not scoping.
CIRCUIT_TABLE_HEADING = "### `millm_circuits`"

#: R2-19: `"MCP ✅" in status` is an unanchored substring test, so text AFTER
#: the mark is invisible. A row reading "REST ✅ · MCP ✅ REVOKED — do not call"
#: passed as a clean registered claim: the contract told a human not to call
#: the tool while the guard certified the row. Require the status to be ONLY
#: the marks, so a qualifier has to be expressed by changing the mark.
CLAIMS_MCP = re.compile(r"^REST\s*[✅❌]\s*·\s*MCP\s*✅$")


def _claims_mcp(status: str) -> bool:
    """True only when the status is exactly a clean MCP claim."""
    return bool(CLAIMS_MCP.match(status.strip()))


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
        erase the only evidence the mistake happened.

        F20 R1-15: this greped for "STATUS CORRECTION" and "RESOLVED", two
        strings that appear in prose nobody will remove — so it could not fail.
        It now asserts the two things that actually matter: the block still
        DESCRIBES the original defect, and no row is left in the state the
        correction says was fixed.
        """
        text = CONTRACT.read_text()
        assert "STATUS CORRECTION" in text, (
            "the status-correction block was deleted rather than resolved"
        )
        assert "RESOLVED" in text, "the correction is still open"

        # The record must still say what was WRONG, or "resolved" is a claim
        # with nothing behind it.
        assert "not registered" in text.split("RESOLVED", 1)[1][:2000], (
            "the resolution no longer describes the defect it resolved — the "
            "block has become a status line rather than a record"
        )

        # And no TABLE ROW may still carry the pre-fix mark. This is the check
        # that would have failed before R1-02: 16 rows read
        # "MCP not registered" while the block above said RESOLVED.
        stale = [
            line for line in text.splitlines()
            if line.startswith("| `millm_") and "MCP not registered" in line
        ]
        assert not stale, (
            f"{len(stale)} rows still read 'MCP not registered' while the "
            "correction above declares it resolved — an agent reading "
            "top-down is told the surface shipped, then told row by row that "
            "it did not"
        )

    @pytest.mark.skipif(
        os.environ.get("MISTUDIO_SETTINGS_AVAILABLE") != "1",
        reason=(
            "F20 R2-13: importing miStudio's schema package pulls its OWN app "
            "settings (database, redis, celery, cache paths) under the SAME "
            "env-var names miLLM uses — the two repos cannot both be "
            "configured in one process. Set MISTUDIO_SETTINGS_AVAILABLE=1 "
            "where they can. Recorded as a KNOWN LIMIT rather than left "
            "permanently red or quietly green: a parity check that cannot run "
            "is UNVERIFIED, and saying so is the honest state. The path half "
            "of this parity IS covered by test_mcp_tool_paths_are_real.py, "
            "which reads the tool module as TEXT and needs no settings."
        ),
    )
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
            _unavailable(
                "miStudio's evidence_ladder is not importable from this "
                "checkout — parity unverified, NOT verified-clean"
            )
        except Exception as exc:
            # Importing miStudio's schema package pulls its app settings,
            # which can fail on a cache directory this process cannot write.
            # That is an ENVIRONMENT limit, not a parity result — report it as
            # unverified rather than as agreement or disagreement.
            _unavailable(
                f"miStudio's evidence_ladder could not be imported ({exc}). "
                "Parity is UNVERIFIED — this is not a clean result."
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
        """`{tool_name: status}` from the millm_circuits table ONLY.

        R2-16..18 rebuilt this. The original scanned every line in the document
        for ``| `millm_`` with no notion of which TABLE it was in, then read
        ``cells[-1]`` as the status. Three defects followed:

        * It captured `millm_import_cluster` from the CLUSTERS table 50 lines
          earlier, whose Endpoint cell contains an escaped pipe — so a 2-column
          row split into 3 cells and passed the ``len(cells) < 3`` filter. Its
          "status" was the prose fragment ``` `fail`) ```. The 14 other
          non-circuit rows were excluded by ACCIDENT (they have exactly 2
          cells), not by design: the arity filter was doing the table-scoping,
          and doing it by coincidence.
        * That phantom row inflated the count to 17, so the ``>= 14`` tripwire
          on the row extraction passed with only 13 real circuit rows. Three
          rows could vanish and the "this guard is checking nothing" alarm
          stayed green.
        * ``rows[name] = ...`` is last-write-wins across the WHOLE document, so
          a stale duplicate row could overwrite a correct status — or, in the
          other order, MASK a lie.

        Now: find the circuits table by its section heading, stop at the next
        heading, and locate the Status column BY NAME from the header row.
        Duplicates inside the table are an error rather than a silent
        overwrite.
        """
        text = CONTRACT.read_text()
        lines = text.splitlines()

        start = next(
            (i for i, l in enumerate(lines) if CIRCUIT_TABLE_HEADING in l), None
        )
        assert start is not None, (
            f"No {CIRCUIT_TABLE_HEADING!r} heading in {CONTRACT}. This guard "
            "locates the circuits table by that heading; without it every "
            "assertion below would run against an empty table and pass."
        )
        end = next(
            (
                i
                for i, l in enumerate(lines[start + 1 :], start + 1)
                if l.startswith("#")
            ),
            len(lines),
        )
        section = lines[start:end]

        # Locate the Status column by NAME. Position is not a contract: adding
        # a trailing Notes column would silently move `cells[-1]` onto it.
        status_col = None
        for line in section:
            if line.startswith("|") and "Status" in line:
                header = [c.strip() for c in line.strip("|").split("|")]
                if "Status" in header:
                    status_col = header.index("Status")
                    break
        assert status_col is not None, (
            "The circuits table has no column literally named 'Status'. The "
            "guard reads that column to decide whether a row claims MCP "
            "support; it will not guess at a position."
        )

        rows: dict = {}
        for line in section:
            if not line.startswith("| `millm_"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) <= status_col:
                continue
            for name in re.findall(r"`(millm_[a-z_]+)`", cells[0]):
                assert name not in rows, (
                    f"`{name}` appears in the circuits table twice. The guard "
                    "refuses to pick one: a stale duplicate can overwrite a "
                    "correct status, or mask a wrong one depending on order. "
                    "Delete the obsolete row."
                )
                rows[name] = cells[status_col]
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
            if _claims_mcp(status)
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
            t for t in registered if not _claims_mcp(rows[t])
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


class TestTheExampleFlowIsRunnable:
    """F20 R2-11/12. §7's circuit flow showed
    `millm_import_circuit(definition=…, activate=true, acknowledge_unvalidated=…)`
    — two arguments the tool does not take — so an agent following the
    reference flow failed on LINE ONE. It also contradicted its own §4 row,
    which says import does not activate.

    R1-03's lesson recurring one section later: the corrected guard read the
    Status COLUMN and never the EXAMPLE CODE beneath it, which is where the
    surviving defect lived. An example is executable prose, and nothing was
    executing it.
    """

    def _circuit_flow(self) -> str:
        """The fenced CODE BLOCK only.

        The prose above it quotes the broken call deliberately — it is the
        postmortem — and scanning that made the guard flag its own
        explanation.
        """
        text = CONTRACT.read_text()
        start = text.index("Circuit flow (v")
        block = text[start:]
        fence = block.index("```")
        end = block.index("```", fence + 3)
        return block[fence:end]

    def _tool_signatures(self) -> dict:
        """`{tool_name: {param names}}` from the LIVE registry."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        registry = _registry()
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:
            _unavailable(f"mcp not importable ({exc})")

        gate = MagicMock()
        gate.check = AsyncMock(return_value=(True, None))
        mcp = FastMCP("flow-check")
        for module in registry["millm_circuits"]:
            module.register(mcp, MagicMock(), gate)
        return {
            # `Tool.inputSchema`, not `.parameters` — the attribute name is
            # the MCP SDK's, and guessing it produced an AttributeError that
            # only surfaced when the guard finally ran.
            t.name: set((getattr(t, "inputSchema", None) or {}).get("properties", {}))
            for t in asyncio.run(mcp.list_tools())
        }

    def test_every_tool_named_in_the_flow_exists(self):
        flow = self._circuit_flow()
        signatures = self._tool_signatures()
        named = set(re.findall(r"\b(millm_[a-z_]+)\(", flow))
        assert named, "no tool calls found in the flow — has the format changed?"
        unknown = sorted(named - set(signatures))
        assert not unknown, (
            f"the reference flow calls {unknown}, which do not exist — an "
            "agent following it fails on that line"
        )

    def test_every_ARGUMENT_in_the_flow_exists_on_its_tool(self):
        """The defect: `activate=true` on a tool with no `activate`."""
        flow = self._circuit_flow()
        signatures = self._tool_signatures()

        bad = []
        for call in re.finditer(r"\b(millm_[a-z_]+)\(([^)]*)\)", flow):
            tool, args = call.group(1), call.group(2)
            if tool not in signatures:
                continue
            for kwarg in re.findall(r"(\w+)\s*=", args):
                if kwarg not in signatures[tool]:
                    bad.append(f"{tool}({kwarg}=…)")
        assert not bad, (
            f"the reference flow passes arguments that do not exist: {bad} — "
            "an agent copying it gets an unexpected-keyword error"
        )

    def test_the_flow_does_not_contradict_the_import_row(self):
        """§4 says import does NOT activate. The flow said otherwise."""
        flow = self._circuit_flow()
        # Strip the block explaining the DEFECT — it quotes the broken call on
        # purpose, and matching your own postmortem is a false positive.
        body = flow.split("```", 1)[1] if "```" in flow else flow
        assert "activate=true" not in body, (
            "the flow shows import activating, and the table row two hundred "
            "lines above says it does not"
        )
        assert "NEVER activates" in body or "does NOT activate" in body

    def test_the_flow_covers_the_recovery_tools(self):
        """R2-12: the two tools with the most dangerous failure modes —
        irreversible deletion, and the stuck-claim recovery — were absent from
        the only end-to-end narrative in the contract."""
        flow = self._circuit_flow()
        assert "millm_release_circuit_claims" in flow
        assert "millm_circuit_sensing_clear" in flow
        assert "scope is REQUIRED" in flow
