"""Regression tests for Feature 031 mutation controls that survived the 2026-10-14 consolidated
sweep (031 FTID §8.4).

M15b: `WeightService._match` compares the presented attachment ID's digest with the stored one
by `hmac.compare_digest`. Replacing it with `==` left every weights suite green, because both
give the same boolean; only the timing differs. FTID §8.4 M15 pins the constant-time compare for
TOKENS (`weights_auth.TokenSet.identify`); the attachment ID is the other secret a gated call
presents, and nothing pinned its compare. These tests assert the call, its arguments and that the
label is still compared when the digest does not match (no early exit).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from datetime import datetime, timezone
from typing import Any

import pytest

import millm.services.weight_attachment as attachment_module
from millm.core.weights_auth import WeightsIdentity
from millm.services.weight_attachment import Attachment, WeightService

TRAINER = WeightsIdentity("miforge", "trainer")
OTHER = WeightsIdentity("intruder", "trainer")
ATTACHMENT_ID = "the-attachment-id-presented-by-the-holder"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture
def service() -> WeightService:
    svc = WeightService()
    digest = _digest(ATTACHMENT_ID)
    svc._attachment = Attachment(
        digest=digest,
        ref=digest[:8],
        identity=TRAINER,
        attached_at=datetime.now(timezone.utc),
        model_id=7,
        pending=False,
    )
    return svc


@pytest.fixture
def spy(monkeypatch) -> list[tuple[Any, Any]]:
    calls: list[tuple[Any, Any]] = []
    real = attachment_module.hmac.compare_digest

    def compare(a: Any, b: Any) -> bool:
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(attachment_module.hmac, "compare_digest", compare)
    return calls


class TestAttachmentIdIsComparedInConstantTime:
    """M15b."""

    def test_the_holder_is_matched_through_compare_digest(self, service, spy):
        assert service.holder_of(TRAINER, ATTACHMENT_ID) is service._attachment
        assert (_digest(ATTACHMENT_ID), _digest(ATTACHMENT_ID)) in spy
        assert len(spy) == 2  # the digest, then the label

    def test_a_wrong_id_is_refused_and_the_label_is_still_compared(self, service, spy):
        assert service.holder_of(TRAINER, "a-guessed-id") is None
        assert spy[0] == (_digest("a-guessed-id"), _digest(ATTACHMENT_ID))
        assert spy[1] == (b"miforge", b"miforge")
        assert len(spy) == 2

    def test_a_wrong_label_with_the_right_id_is_refused(self, service, spy):
        assert service.holder_of(OTHER, ATTACHMENT_ID) is None
        assert len(spy) == 2

    def test_match_has_no_equality_compare_on_a_digest(self):
        source = textwrap.dedent(inspect.getsource(WeightService._match))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Compare) and any(
                isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops
            ):
                names = {getattr(n, "id", getattr(n, "attr", None)) for n in ast.walk(node)}
                assert not names & {"digest", "presented"}, ast.unparse(node)
