"""At DEBUG, formatting a chat logs its shape, never its text (Feature 35, found 2026-10-20).

`LOG_LEVEL` can be switched to DEBUG while the server runs. `_format_chat_messages` logged the first
200 characters of every message and the first 500 of the rendered prompt at DEBUG, which would
have put users' chat text into the pod log.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import structlog

from millm.services.inference_service import InferenceService

SECRET = "MY-PRIVATE-PROMPT-TEXT"


class _Tokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        return "<s>" + " ".join(m["content"] for m in messages) + "</s>"


def test_debug_logs_carry_no_message_text():
    svc = InferenceService.__new__(InferenceService)
    svc._model_state = SimpleNamespace(
        is_loaded=True, current=SimpleNamespace(tokenizer=_Tokenizer())
    )
    messages = [SimpleNamespace(role="user", content=f"{SECRET} one"),
                SimpleNamespace(role="assistant", content=f"{SECRET} two")]
    with structlog.testing.capture_logs() as logs:
        formatted = svc._format_chat_messages(messages)
    assert SECRET in formatted  # the prompt itself is built as before
    events = [e for e in logs if e["event"] in ("chat_message", "formatted_prompt")]
    assert [e["event"] for e in events] == ["chat_message", "chat_message", "formatted_prompt"]
    assert events[0]["content_chars"] == len(f"{SECRET} one")
    assert SECRET not in json.dumps(logs, default=str)
