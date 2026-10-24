"""Documentation pin (036 FR-36.5): the Open WebUI filter tells the operator how to name sessions.

No code in Open WebUI or in the filter is needed — a connection custom header carries the chat id
(`research/openwebui-session-id.md`). This pins that the filter's docstring says so, names the
header and the `{{CHAT_ID}}` template, and warns off the user-info forwarding switch.
"""

import ast
from pathlib import Path

FILTER = Path(__file__).resolve().parents[3] / "integrations" / "openwebui" / "millm_dial_filter.py"


def test_the_docstring_names_the_session_header_and_the_chat_id_template():
    doc = ast.get_docstring(ast.parse(FILTER.read_text())) or ""
    assert "X-miLLM-Session: {{CHAT_ID}}" in doc
    assert "ENABLE_FORWARD_USER_INFO_HEADERS" in doc
