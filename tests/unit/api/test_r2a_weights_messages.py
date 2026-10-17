"""The management envelope's top-level `message` for a weights refusal must not contradict the
refusal (review R2-A; the WEIGHTS_UNAUTHORIZED fix, cd145d8, was the same class of defect).

The management handler replaces every MiLLMError's message with the code's ERROR_MESSAGES
sentence and keeps the crafted one only as `details.technical_message`. D-39 added a second
meaning to NOT_QUIESCED (resume while the holder's quiesce is still draining), and the sentence
still read "Serving is not paused. Quiesce first." — telling a holder whose quiesce is in flight
to quiesce again, which is refused ALREADY_QUIESCED.

MUTATION CONTROL (R2-A): the sentence restored to "Serving is not paused. Quiesce first." -> red.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from millm.api.exception_handlers import millm_error_handler
from millm.core.errors import NotQuiescedError


async def test_a_resume_refused_while_quiesce_drains_says_to_wait():
    exc = NotQuiescedError(
        "resume refused: quiesce is still draining in-flight work; resume once quiesce has "
        "returned.",
        details={"operation": "resume", "barrier": "serving", "reason": "quiesce_in_progress"},
    )
    request = SimpleNamespace(url=SimpleNamespace(path="/api/weights/resume"), method="POST")
    response = await millm_error_handler(request, exc)
    error = json.loads(response.body)["error"]
    assert response.status_code == 409 and error["code"] == "NOT_QUIESCED"
    assert error["details"]["reason"] == "quiesce_in_progress"
    message = error["message"]
    assert "quiesce_in_progress" in message and "wait" in message, message
