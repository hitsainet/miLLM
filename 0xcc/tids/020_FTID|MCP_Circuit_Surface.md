# Technical Implementation Document: MCP Circuit Surface & Reachability Assurance

## miLLM Feature 20

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `020_FPRD|MCP_Circuit_Surface.md` · `020_FTDD|MCP_Circuit_Surface.md` · `docs/mcp-contract.md` (v1.1 → v1.2)

---

## 1. File Structure

**Note the repo split.** Implementation lands in **miStudio**; the contract and doc chain live in
**miLLM**. Paths below are prefixed accordingly.

```
miStudio/backend/src/mcp_server/
├── tools/millm_circuits.py        (NEW — 13 tools; sibling of millm_sensing.py)
├── tools/__init__.py              (MOD — import + MILLM_CATEGORY_MODULES entry)
├── config.py                      (MOD — VALID_CATEGORIES += "millm_circuits"; DEFAULT unchanged)
└── server.py                      (UNCHANGED — the registration loop is already generic)

miStudio/backend/tests/unit/
├── test_mcp_millm_circuits.py     (NEW — per-tool method/path/gate/validation)
├── test_reachability.py           (NEW — registry + built-server + caller + RCH-5 regressions)
└── test_causal_language_audit.py  (MOD — MCP module scan, description scan, parity guard)

miStudio/admin-ui/src/components/... (NO CHANGE — RCH-5 tests an existing control)

miLLM/
├── docs/mcp-contract.md           (MOD — v1.2: marks, sensing paths, §6 wiring, §7 flow)
├── tests/unit/core/test_circuit_evidence.py (MOD — reciprocal parity guard)
└── tests/unit/test_mcp_contract_consistency.py (NEW — doc rows vs registry claims)
```

## 2. Load-Bearing Implementation Points (verified against live code)

- **`server.py` needs NO edit.** The miLLM registration loop (`server.py:118-132`) iterates
  `MILLM_CATEGORY_MODULES`, filters by `settings.enabled_categories()`, and calls
  `module.register(mcp, millm_client, gate)`. A registry entry is sufficient — and that genericity is
  exactly why the original omission was silent: nothing notices a missing category, it just loops one
  fewer time. Do NOT add special-casing; add the entry.
- **Two registration surfaces, both required.** `tools/__init__.py:8-22` (the package import) AND
  `:40-44` (`MILLM_CATEGORY_MODULES`). A module imported but not registered, or registered but not
  imported, fails differently — the first silently, the second with an `ImportError` at startup.
  Additionally `config.py:8-15` `VALID_CATEGORIES` must gain `"millm_circuits"`, or
  `enabled_categories()` raises "Unknown MCP tool categories" (`config.py:47-52`) and the server
  refuses to start for anyone who opts in.
- **`DEFAULT_CATEGORIES` (`config.py:16`) must NOT change.** It deliberately excludes every `millm_*`
  category ("opt-in, functional only with MILLM_API_URL set — never in DEFAULT_CATEGORIES").
- **Copy the `@mcp.tool()` + `@gated(gate, "millm")` pair exactly** (`millm_sensing.py:17-18`). Order
  matters: `@mcp.tool()` outermost so FastMCP registers the gated callable. `gated` returns the
  structured `{"unavailable": product, "reason": …}` (`health_gate.py:159-181`) and never raises.
- **The one tool that must NOT use the plain `@gated` decorator** is `millm_import_circuit`, if it
  follows `millm_clusters.py:26-67`'s precedent: that tool does its argument validation BEFORE the gate
  check (calling `await gate.check("millm")` inline at :50-52) so a malformed call gets a useful
  argument error even when miLLM is down. Mirror it — an agent debugging its own payload should not be
  told "millm unavailable".
- **Envelope pass-through, no unwrapping** (contract §2: "unwrap in the client, never in tools").
  Return `await millm.get(...)` directly. In particular a `success:false` body is a normal RETURN, not
  an exception — `NO_ACTIVE_CIRCUIT` and `UNVALIDATED_CIRCUIT` are 200 + `success:false` (house style,
  seen live at `circuits.py:266-270`).
- **`millm_export_circuit` returns a RAW document.** `GET /api/circuits/{id}/export`
  (`circuits.py:296-310`) has deliberately NO `response_model` — "a mirror would strip unknown additive
  fields from newer producers". The tool must not re-model it either, and its description must warn
  that unwrapping `.data` yields `None` (the cluster precedent `millm_export_cluster` already behaves
  this way).
- **Sensing prefix is `/api/circuit-sensing`, NOT `/api/circuits/{id}/sensing`**
  (`circuit_sensing.py:38`). The contract's §4 table still shows the reserved paths; the prose above it
  corrects them. Write tools against the CODE, and fix the table in v1.2.
- **`GET /api/circuits/active` `steering` field is tri-state** (`circuits.py:101-129`): `true`,
  `false`, or `null` when the computation raised (the route wraps it in a bare `except` so "a status
  nicety must never fail the status call"). The description must say `null` = *not evaluated*, never
  *not steering* — the contract §4b already states this rule for the header and it applies identically
  here.
- **`since` must be tz-aware.** Copy `millm_sensing.py:42-52` verbatim in behavior: parse with
  `datetime.fromisoformat(since.replace("Z", "+00:00"))`, reject `tzinfo is None` with the explanatory
  message. A naive timestamp shifts the polling window silently — on an evidence surface that means
  silently under-reporting observations.
- **Copy audit port targets** — miLLM's `TestCopyAudit` at `tests/unit/core/test_circuit_evidence.py`:
  `ALLOWED` (:110-113), `UNRELATED_SENSE` (:117-119), `EVIDENCE_CONTEXT` (:125-127), `_scan`
  (:129-153), `_COMMENT`/`_code_only` (:157-166), marker allow-list (:184-207). Port ALL of it; the
  three-stage filter is what keeps the audit from firing on `CausalLMOutputWithPast`.
- **miStudio's existing audit is looser.** `backend/tests/unit/test_causal_language_audit.py` uses a
  two-stage `CAUSAL` / `ALLOWED_CONTEXT` pair (:34, :45) with no comment stripping. Do NOT simply add
  the MCP path to it — port miLLM's stricter logic, or the audit will pass on a docstring that miLLM's
  audit would reject.

## 3. Key Implementations

```python
# tools/millm_circuits.py — the lifecycle tools (shape mirrors millm_clusters.py)

def register(mcp: FastMCP, millm: MiLLMClient, gate: HealthGate) -> None:

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_status() -> Any:
        """The circuit currently serving, or null when none is.

        Carries serving_mode ('full' | 'slice_fallback'), the attached-SAE set,
        rung + rung_language, and `steering` — the SERVER's verdict on whether
        this circuit is genuinely influencing generation. Read `steering`; do
        NOT infer it from `is_active`, which overclaims for a slice-fallback,
        unparseable or unattached circuit. `steering: null` means the server
        did not evaluate it, NOT that it is not steering.

        In slice_fallback the circuit is served through one layer's cluster
        slice: a slice is never the whole circuit, and the dial then follows
        CLUSTER rules (0.5 floor), not circuit rules."""
        return await millm.get("/api/circuits/active")

    @mcp.tool()
    async def millm_import_circuit(definition: dict,
                                   on_conflict: Optional[str] = None) -> Any:
        """Import a mistudio.circuit-definition/v1 document into miLLM.

        Import does NOT activate — call millm_activate_circuit separately, so
        the evidence gate is always an explicit step. on_conflict: 'rename'
        (default) or 'fail'. Caps: 1 MB, 16 layers, 200 edges."""
        # Argument errors BEFORE the gate (millm_clusters.py:44 precedent): an
        # agent debugging its own payload should not be told "millm is down".
        if on_conflict not in (None, "rename", "fail"):
            return {"error": "`on_conflict` must be 'rename' or 'fail'"}
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        return await millm.post("/api/circuits/import",
                                json_body=definition, on_conflict=on_conflict)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_activate_circuit(circuit_id: str,
                                     acknowledge_unvalidated: bool = False) -> Any:
        """Activate an imported circuit.

        A circuit below rung 2 requires acknowledge_unvalidated=true; without
        it the call is REFUSED with UNVALIDATED_CIRCUIT as a 200 + success:false
        envelope carrying the rung and rung_language, so you can surface the
        evidence level to the user and re-send with the acknowledgement.

        Activation degrades to slice_fallback when not all referenced SAEs are
        attached — check serving_mode in the result."""
        return await millm.post(
            f"/api/circuits/{circuit_id}/activate",
            acknowledge_unvalidated=str(bool(acknowledge_unvalidated)).lower())
```

```python
# tools/millm_circuits.py — sensing: the description carries the three semantics
# an agent otherwise gets wrong (MCP-E4).

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_sensing_events(circuit_id: Optional[str] = None,
                                           edge_key: Optional[str] = None,
                                           limit: int = 50,
                                           since: Optional[str] = None) -> Any:
        """Edge observations newest-first: an edge's UPSTREAM member fired and
        its DOWNSTREAM partner then fired within the lag window, in the
        authored direction. Rows carry nested up/down {layer, feature_idx,
        pos, act}, token_lag, the +/-K context_parts window, and edge_rung +
        edge_rung_language.

        AN OBSERVATION IS NOT VALIDATION. It is co-activation evidence in the
        authored direction. It never raises an edge's rung, and a high
        observation count is not evidence of causality — do not present it as
        such. edge_rung_language is stored AS OF THE MOMENT OF OBSERVATION, so
        render the stored phrase, never today's.

        ABSENCE OF ROWS IS NOT ABSENCE OF FIRING. Check
        millm_circuit_sensing_status.unsensable_edges before reporting that an
        edge did not fire; edges on unattached layers are never watched.

        `since` (ISO-8601) MUST carry an explicit UTC offset (e.g.
        2026-07-20T12:00:00Z) — naive timestamps shift the window silently."""
        if since is not None:
            from datetime import datetime
            try:
                parsed = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                return {"error": f"`since` is not ISO-8601: {since!r}"}
            if parsed.tzinfo is None:
                return {"error": "`since` must carry a UTC offset "
                                 "(e.g. ...T12:00:00Z) — naive timestamps "
                                 "shift the polling window silently"}
        return await millm.get("/api/circuit-sensing/events",
                               circuit_id=circuit_id, edge_key=edge_key,
                               limit=limit, since=since)
```

```python
# tests/unit/test_reachability.py — the three shapes. Each alone is an
# existence assertion; only the conjunction is reachability evidence.

def test_category_is_registered():
    """FAILS if the MILLM_CATEGORY_MODULES entry is deleted."""
    from src.mcp_server.tools import MILLM_CATEGORY_MODULES
    from src.mcp_server.config import VALID_CATEGORIES
    assert "millm_circuits" in MILLM_CATEGORY_MODULES
    assert "millm_circuits" in VALID_CATEGORIES

def test_tools_present_on_a_built_server(monkeypatch):
    """FAILS if the module registers nothing, or the gating drops it.

    Builds through the REAL build_server() — hand-calling register() would
    bypass exactly the gating that failed originally.
    """
    settings = MCPSettings(auth_token="t",
                           tool_categories="millm_circuits",
                           ...)  # MILLM_API_URL set
    mcp, _ = build_server(settings)
    names = {t.name for t in mcp._tool_manager.list_tools()}
    assert CONTRACT_CIRCUIT_TOOLS <= names, sorted(CONTRACT_CIRCUIT_TOOLS - names)

@pytest.mark.parametrize("tool_name,method,path", CONTRACT_ROWS)
async def test_tool_calls_its_documented_endpoint(tool_name, method, path):
    """FAILS if the call is removed or re-pointed at a different path."""
    client = RecordingMiLLMClient()
    tool = registered_tool(tool_name, client)
    await tool(**minimal_args(tool_name))
    assert client.calls == [(method, path)]
```

## 4. Implementation Pitfalls

1. **Registering a tool is not making it reachable.** Three separate places must agree —
   `tools/__init__.py` import, `MILLM_CATEGORY_MODULES`, `VALID_CATEGORIES` — and a fourth
   (`MCP_TOOL_CATEGORIES` in the deployment) must opt in. The original defect was exactly one of these
   missing. Assert all of them.
2. **Do not hand-call `register()` in the reachability test.** It bypasses `build_server`'s category
   filtering, which is the layer that actually failed. Build the server.
3. **Do not add `millm_circuits` to `DEFAULT_CATEGORIES`.** miStudio-only deployments would then
   request a category whose `MILLM_API_URL` is unset and take the warning path on every start.
4. **The audit must scan descriptions, not only source.** A `__doc__` composed at runtime, inherited,
   or assembled from constants is invisible to a line grep but fully visible to the agent. Scan
   `mcp._tool_manager.list_tools()` descriptions too.
5. **Strip comments before auditing** (`_code_only`, miLLM :157-166). An R3 finding: a marker in a
   trailing comment previously exempted a claim in a string literal — that is how an allow-list quietly
   stops guarding.
6. **The parity guard must skip LOUDLY.** When the sibling checkout is absent, an explicit skip naming
   the missing repo. A parity test that passes vacuously when it cannot read the other side is
   `TestRingPruningIsWired` one level up — the precise defect this feature exists to abolish.
7. **Do not register hub tools.** `/api/circuits/hub/*` 404s today (contract §4: "F15 — not served").
   A registered tool against an unserved endpoint is a new unreachable capability created by the
   feature that abolishes them.
8. **`NO_ACTIVE_CIRCUIT` / `UNVALIDATED_CIRCUIT` are 200 + `success:false`.** Never branch on HTTP
   status alone (contract §2), and never convert them to exceptions — the agent needs the envelope to
   read the rung and re-send with the acknowledgement.
9. **Never re-derive the circuit rung.** It is MIN over edges, computed server-side. A tool that
   recomputes it will disagree with the server precisely when the edges are heterogeneous — i.e.
   exactly when the answer matters.
10. **`steering: null` ≠ not steering.** The route's bare `except` (`circuits.py:126-128`) maps any
    failure to `None`. Treating `null` as `false` reintroduces the R3 overclaim from the other
    direction.
11. **13 tools vs 12 contract rows.** The contract collapses enable/disable and omits `delete`
    (shipped at `circuits.py:284`). Reconcile in v1.2 rather than leaving the count mismatched — a
    row-count assertion is part of the consistency test.
12. **Write the tools against POST-refactor code.** F20 is sequenced last for this reason (BRD locked
    decision 1). If Feature 18 has not landed, the serving derivations are still moving and a tool bug
    is indistinguishable from a refactor bug.

## 5. Config Additions

**miStudio** (`backend/src/mcp_server/config.py`) — one line, and one non-change:

```python
VALID_CATEGORIES = {
    ..., "millm_runtime", "millm_clusters", "millm_sensing",
    "millm_circuits",          # Feature 20 — opt-in, requires MILLM_API_URL
}
# DEFAULT_CATEGORIES: UNCHANGED (no millm_* category is ever a default)
```

**Deployment** (documented in contract §6):

```
MCP_TOOL_CATEGORIES=...,millm_runtime,millm_clusters,millm_sensing,millm_circuits
MILLM_API_URL=http://millm-backend.millm.svc.cluster.local:8000
```

**miLLM:** no config changes. No new settings keys, no migration, no runtime state.
