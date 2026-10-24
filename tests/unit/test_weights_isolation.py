"""The autouse reset in `tests/conftest.py` drops the process-wide weight barrier and weight
service after every test (addendum §8). The first test LEAKS a torn barrier and an attachment on
purpose; the second (run after it, file order) must still see a clean world. Removing the
conftest reset turns the second test red — the negative control for the fixture itself.
"""

from __future__ import annotations

from datetime import datetime, timezone

from millm.core.weights_auth import WeightsIdentity
from millm.services.weight_attachment import Attachment, get_weight_service
from millm.services.weight_barrier import get_weight_barrier


def test_1_leaks_a_torn_barrier_and_an_attachment():
    get_weight_barrier().mark_torn("leaked by a test")
    get_weight_service()._attachment = Attachment(
        digest="d" * 64, ref="dddddddd", identity=WeightsIdentity("miforge", "trainer"),
        attached_at=datetime.now(timezone.utc), model_id=1, pending=False,
    )
    assert get_weight_barrier().admission_refusal() is not None


def test_2_the_next_test_starts_clean():
    assert get_weight_barrier().snapshot().weights_state == "clean"
    assert get_weight_barrier().admission_refusal() is None
    assert get_weight_service().is_attached() is False
