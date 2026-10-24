"""The MCP contract names the probe socket event the code actually emits (036 FR-36.17.1).

`docs/mcp-contract.md` said `probe:verdict` while `progress_emitter.emit_probe_event` emits
`probe:event`. The name is READ FROM THE EMITTER (its AST), not typed here, so renaming the event
in code without the contract turns this red.
"""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _emitted_names(method: str) -> set[str]:
    tree = ast.parse((REPO / "millm" / "sockets" / "progress.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method:
            return {
                c.args[0].value for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == "emit" and c.args and isinstance(c.args[0], ast.Constant)
            }
    raise AssertionError(f"{method} not found in progress.py")


def test_the_contract_uses_the_emitted_probe_event_name():
    # 039 (TD39-28): one `probe:results` message per request replaced the per-verdict
    # `probe:event`; the contract must name the event the code emits.
    names = _emitted_names("emit_probe_results")
    assert names == {"probe:results"}, names
    path = REPO / "docs" / "mcp-contract.md"
    if not path.exists():
        pytest.skip("docs/ (but schemas) is stripped from the public mirror; checked in the source repo")
    contract = path.read_text()
    assert "`probe:results`" in contract
    assert "probe:verdict" not in contract


def test_the_session_event_is_emitted_under_its_contract_name():
    assert _emitted_names("emit_probe_session") == {"probe:session"}
