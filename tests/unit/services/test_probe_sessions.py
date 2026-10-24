"""Session resolution: the header, the salted continuation fingerprint, the idle end (FR-36.1–36.4).

Every fixture pair differs in exactly the message (or minute) under test, so a resolver that ignores
that field joins when it must not, or splits when it must not — and the test says which.

Mutation controls (036 FTDD §10), recorded in `0xcc/reviews/036_037_controls_2026-10-21.md`:
* drop `continuation_hash=` from `write_turn`            → `test_exact_continuation_joins` red
* idle `<` → `<=` in `is_open`                           → `test_idle_boundary_exactly_30_minutes` red
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from millm.core.config import settings
from millm.db.models.app_setting import AppSetting
from millm.db.models.probe_session import ProbeSession, ProbeSessionTurn, ProbeTranscript
from millm.services import probe_sessions
from millm.services.probe_sessions import (
    ProbeSessionService,
    canonical_pairs,
    fingerprint,
    resolve_in,
)

pytestmark = pytest.mark.asyncio

T0 = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    probe_sessions._forget_for_tests()
    now = {"t": T0}
    monkeypatch.setattr(probe_sessions, "_now", lambda: now["t"])
    yield now
    probe_sessions._forget_for_tests()


def u(text):
    return {"role": "user", "content": text}


def a(text):
    return {"role": "assistant", "content": text}


def s(text):
    return {"role": "system", "content": text}


async def turn(db, messages, reply="hello", header=None):
    ref = await resolve_in(db, messages=messages, client_session_id=header, served_model="m")
    await ProbeSessionService(db).write_turn(
        ref, request_id=f"chatcmpl-{uuid.uuid4().hex[:8]}", served_model="m", reply_text=reply
    )
    return ref


# ── canonical form ───────────────────────────────────────────────────────────────


class TestCanonicalForm:
    def test_roles_lowered_line_endings_normalised_outer_whitespace_stripped(self):
        assert canonical_pairs([{"role": "User", "content": "  a\r\nb\rc  "}]) == [("user", "a\nb\nc")]

    def test_an_image_part_hashes_as_its_type(self):
        msg = {"role": "user", "content": [
            {"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "x"}},
        ]}
        assert canonical_pairs([msg]) == [("user", "look\n[image_url]")]

    def test_reasoning_and_other_keys_are_dropped(self):
        plain = canonical_pairs([a("hi")])
        assert canonical_pairs([{**a("hi"), "reasoning_content": "secret", "name": "x"}]) == plain

    def test_the_hash_is_keyed_by_the_installation_salt(self):
        pairs = [("user", "hi")]
        assert fingerprint(b"1" * 32, pairs) != fingerprint(b"2" * 32, pairs)
        assert fingerprint(b"1" * 32, pairs) == fingerprint(b"1" * 32, pairs)


async def test_the_salt_is_written_once_and_reused(test_session):
    first = await probe_sessions.installation_salt(test_session)
    probe_sessions._forget_for_tests()
    second = await probe_sessions.installation_salt(test_session)
    rows = (await test_session.execute(select(AppSetting))).scalars().all()
    assert first == second and len(first) == 32
    assert [r.key for r in rows] == [probe_sessions.SALT_KEY]


# ── the header (FR-36.1) ─────────────────────────────────────────────────────────


async def test_header_names_the_session(test_session):
    r1 = await turn(test_session, [u("hi")], header="s1")
    r2 = await turn(test_session, [u("something else entirely")], header="s1")
    r3 = await turn(test_session, [u("hi")], header="s2")
    assert r1.id == r2.id != r3.id
    assert (r1.source, r1.new, r2.new, r3.new) == ("header", True, False, True)
    rows = (await test_session.execute(select(ProbeSession))).scalars().all()
    assert sorted(r.client_session_id for r in rows) == ["s1", "s2"]


async def test_the_header_wins_over_the_fingerprint(test_session):
    first = await turn(test_session, [u("hi")], reply="hello")
    # A perfect continuation of `first`, but named as another conversation.
    second = await turn(test_session, [u("hi"), a("hello"), u("more")], header="other")
    assert second.id != first.id and second.source == "header"


# ── the fingerprint table (FTDD §10) ─────────────────────────────────────────────


async def test_exact_continuation_joins(test_session):
    """The STRONG key decides here: two open conversations share the first message (so the weak
    key matches both and would refuse as ambiguous); only the verbatim reply tells them apart."""
    first = await turn(test_session, [u("hi")], reply="hello A")
    other = await turn(test_session, [u("hi")], reply="hello B")
    second = await turn(test_session, [u("hi"), a("hello A"), u("more")])
    assert second.id == first.id != other.id and not second.new
    assert second.turn_index == 2


@pytest.mark.parametrize(
    "first, reply, second, joins",
    [
        # trimmed reply → the weak key
        ([u("hi")], "hello there", [u("hi"), a("hello"), u("more")], True),
        # reasoning re-inserted by the client → the weak key
        ([u("hi")], "hello", [u("hi"), a("<details>thought</details>hello"), u("more")], True),
        # edited first user turn
        ([u("hi")], "hello", [u("hi!"), a("hello"), u("more")], False),
        # edited system prompt
        ([s("A"), u("hi")], "hello", [s("B"), u("hi"), a("hello"), u("more")], False),
        # same text, CRLF line endings
        ([u("hi")], "hello\nworld", [u("hi"), a("hello\r\nworld"), u("more")], True),
        # trailing space inside a message
        ([u("hi")], "hello", [u("hi"), a("hello "), u("more")], True),
        # internal whitespace changed in the first user turn
        ([u("hi there")], "hello", [u("hi  there"), a("hello"), u("more")], False),
    ],
    ids=["trimmed-reply", "reasoning-reinserted", "edited-user", "edited-system", "crlf",
         "trailing-space", "internal-whitespace"],
)
async def test_fingerprint_table(test_session, first, reply, second, joins):
    one = await turn(test_session, first, reply=reply)
    two = await turn(test_session, second)
    assert (two.id == one.id) is joins


async def test_two_identical_open_conversations_are_link_ambiguous(test_session):
    one = await turn(test_session, [u("hi")], reply="hello")
    two = await turn(test_session, [u("hi")], reply="hello")
    assert one.id != two.id  # an empty prefix always opens a new session
    third = await turn(test_session, [u("hi"), a("hello"), u("more")])
    assert third.id not in (one.id, two.id) and third.new
    row = await test_session.get(ProbeSession, third.id)
    assert row.link_ambiguous is True


async def test_a_weak_match_on_two_sessions_is_ambiguous_not_joined(test_session):
    one = await turn(test_session, [u("hi")], reply="hello A")
    two = await turn(test_session, [u("hi")], reply="hello B")
    # Neither reply verbatim: only the weak key can match, and it matches both.
    third = await turn(test_session, [u("hi"), a("hello C"), u("more")])
    assert third.id not in (one.id, two.id)
    assert (await test_session.get(ProbeSession, third.id)).link_ambiguous is True


async def test_a_text_completion_is_a_single_turn_session(test_session):
    ref = await resolve_in(test_session, messages=None, client_session_id=None, served_model="m")
    assert (ref.source, ref.turn_index, ref.chat) == ("single", 1, False)


# ── idle end and successors (FR-36.3) ────────────────────────────────────────────


async def test_29_minutes_joins_and_31_opens_a_linked_successor(test_session, clock):
    first = await turn(test_session, [u("hi")], reply="hello")
    clock["t"] = T0 + timedelta(minutes=29)
    second = await turn(test_session, [u("hi"), a("hello"), u("more")], reply="ok")
    assert second.id == first.id
    clock["t"] = T0 + timedelta(minutes=29 + 31)
    third = await turn(test_session, [u("hi"), a("hello"), u("more"), a("ok"), u("again")])
    assert third.id != first.id
    assert (await test_session.get(ProbeSession, third.id)).predecessor_id == first.id


async def test_idle_boundary_exactly_30_minutes(test_session, clock):
    """⚠ STRICT: idle for exactly the limit is CLOSED. `<=` would join here."""
    assert settings.PROBE_SESSION_IDLE_MINUTES == 30
    first = await turn(test_session, [u("x")], header="h")
    clock["t"] = T0 + timedelta(minutes=30)
    again = await turn(test_session, [u("y")], header="h")
    assert again.id != first.id
    assert (await test_session.get(ProbeSession, again.id)).predecessor_id == first.id


async def test_the_idle_limit_is_read_at_request_time(test_session, clock, monkeypatch):
    first = await turn(test_session, [u("x")], header="h")
    monkeypatch.setattr(settings, "PROBE_SESSION_IDLE_MINUTES", 5)
    clock["t"] = T0 + timedelta(minutes=6)
    assert (await turn(test_session, [u("y")], header="h")).id != first.id


# ── turns (FR-36.4) ──────────────────────────────────────────────────────────────


async def test_turn_index_counts_user_messages_and_hashes_are_written(test_session):
    ref = await turn(test_session, [u("1"), a("a"), u("2"), a("b"), u("3")], reply="c")
    row = (await test_session.execute(select(ProbeSessionTurn))).scalar_one()
    assert (row.turn_index, row.session_id) == (3, ref.id)
    assert len(row.prefix_hash) == len(row.weak_hash) == len(row.continuation_hash) == 64
    assert row.has_transcript is False
    session = await test_session.get(ProbeSession, ref.id)
    assert session.n_turns == 1


async def test_no_transcript_while_recording_is_off(test_session):
    assert settings.PROBE_TRANSCRIPTS == "off"
    await turn(test_session, [u("SENTINEL-OFF")], reply="reply")
    assert (await test_session.execute(select(ProbeTranscript))).scalars().all() == []


async def test_lookup_timing_is_recorded_not_asserted(test_session, capsys):
    """FPRD §8: p95 of a fingerprint lookup is REPORTED (review record), never asserted in CI."""
    import time

    await turn(test_session, [u("hi")], reply="hello")
    samples = []
    for i in range(40):
        t = time.perf_counter()
        await resolve_in(test_session, messages=[u("hi"), a("hello"), u(f"q{i}")],
                         client_session_id=None, served_model="m")
        samples.append((time.perf_counter() - t) * 1000)
    samples.sort()
    print(f"resolve_session p95 = {samples[int(len(samples) * 0.95) - 1]:.2f} ms (SQLite)")
