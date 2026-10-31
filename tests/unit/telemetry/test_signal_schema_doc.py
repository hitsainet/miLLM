"""The telemetry guide and the contract text agree with the code (038 FR-38.26, FR-38.27; FTASKS 12.6 – 12.8).

* The guide's signal-schema block (`manual/docs/features/telemetry-export.md`) equals the block
  generated from a LIVE meter provider and the allowlist tables: an instrument registered but not
  documented, or documented but gone, is red — control M27 (remove an instrument from the guide).
* The probe page's reason-code table equals `probe_reasons.REASON_TABLE`.
* Every `TELEMETRY_*` setting (and `OTEL_SDK_DISABLED`) is in the configuration reference.
* The guide states what review R1 of S2 asked it to (credentials in headers, never the URL; no
  sampled flag on result records; `gen_ai.request.model` only when it names the loaded model; the
  sender's redirect, response-size and timeout behaviour).
* `docs/mcp-contract.md` is v1.15 and lists every published reason code and both refusals.

The manual ships on the public mirror, so those checks run there too. `docs/` does not (the mirror
sync strips it), so the contract checks skip LOUDLY when it is absent, as
`test_mcp_contract_consistency.py` does; they run in the source repo and its CI.
"""

from __future__ import annotations

import re

import pytest

from millm.core.config import Settings
from millm.services.probe_reasons import REASON_TABLE
from tests.support import telemetry_signal_doc as gen

ROOT = gen.ROOT
CONFIG_PAGE = ROOT / "manual" / "docs" / "reference" / "configuration.md"
SIDEBAR = ROOT / "manual" / "sidebars.ts"
CONTRACT = ROOT / "docs" / "mcp-contract.md"


def _contract() -> str:
    if not CONTRACT.exists():
        pytest.skip(f"{CONTRACT} is absent: docs/ is stripped from the public mirror. UNVERIFIED "
                    "here; this check runs in the source repo.")
    return CONTRACT.read_text()


def test_the_guides_signal_schema_block_is_the_generated_one():
    """M27: delete an instrument's row from the guide (or register a new instrument) → red."""
    committed = gen.committed_block(gen.GUIDE.read_text())
    assert committed is not None, "the signal-schema markers are gone from the guide"
    assert committed == gen.render(), (
        "the guide's signal schemas differ from the live registry; regenerate with "
        "`venv/bin/python -m tests.support.telemetry_signal_doc --write`"
    )


def test_the_generated_block_names_every_live_instrument():
    """The generator itself cannot drop one: every live instrument has its own table row."""
    block = gen.render()
    instruments = gen.live_instruments()
    assert len(instruments) >= 7
    for name, _kind, _unit in instruments:
        assert re.search(rf"^\| `{re.escape(name)}` \|", block, re.MULTILINE), name


def test_the_probe_pages_reason_table_is_the_published_list():
    committed = gen.committed_block(gen.PROBE_PAGE.read_text(), gen.REASONS_BEGIN,
                                    gen.REASONS_END)
    assert committed is not None, "the reason-code markers are gone from probe-monitors.md"
    assert committed == gen.render_reasons()
    for code in REASON_TABLE:
        assert f"| `{code}` |" in committed, code


def test_every_telemetry_setting_is_in_the_configuration_reference():
    page = CONFIG_PAGE.read_text()
    names = [n for n in Settings.model_fields if n.startswith("TELEMETRY_")] + ["OTEL_SDK_DISABLED"]
    assert len(names) >= 25
    missing = [n for n in names if f"`{n}`" not in page]
    assert missing == [], f"undocumented settings: {missing}"


def test_the_guide_is_in_the_manual_sidebar():
    assert "'features/telemetry-export'" in SIDEBAR.read_text()


@pytest.mark.parametrize("statement", [
    # Review R1 (S2) debt 4: a token in the endpoint path would be displayed.
    "Credentials go in headers, never in the URL",
    # Debt 3: what a result record does and does not claim.
    "Result records carry no sampled flag",
    "`gen_ai.request.model` is sent only when it names the loaded model",
    # The sender's behaviour, R1-2 / R1-3 and debt 1.
    "Redirects are never followed",
    "The response body is not read",
    "not the whole exchange",
    # FR-38.26's list.
    "count once",
    "What a restart can lose",
    "does not erase copies already exported",
    "Parity tolerances (three, not one)",
])
def test_the_guide_states(statement: str):
    assert statement.lower() in gen.GUIDE.read_text().lower(), statement


def test_the_contract_is_v1_15_and_lists_every_reason_code_and_both_refusals():
    text = _contract()
    # v1.15 is 038's; later versions keep its section and its changelog entry (039: v1.16).
    assert "**v1.15 (2026-10-23)**" in text
    section = text[text.index("### 4i."):text.index("## 5.")]
    missing = [code for code in REASON_TABLE if f"`{code}`" not in section]
    assert missing == [], f"reason codes missing from contract §4i: {missing}"
    for code in ("PROBE_IDENTITY_MISMATCH", "PROBE_IDENTITY_ALGORITHM_UNKNOWN"):
        assert f"`{code}`" in text[text.index("## 5."):], code
    for field in ("millm_probe_verdicts", "export_results", "result_id", "trace_id",
                  "`telemetry`"):
        assert field in section, field
