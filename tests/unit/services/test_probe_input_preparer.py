"""`ProbeInputPreparer` — each input kind and each boundary rule (T-72, T-49, TD6). Pure: a real
tokenizer and chat template, no model."""

from __future__ import annotations

from millm.api.schemas.probe_scoring import ProbeScoreInput
from millm.services.probe_scoring import TOKENIZATION_FAILED, ProbeInputPreparer
from tests.unit.f25_fixtures import word_tokenizer


def _render(tok):
    def render(messages, generation_prompt):
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=generation_prompt)
    return render


def prep(item: dict, render=None):
    tok = word_tokenizer()
    return ProbeInputPreparer(tok, render or _render(tok)).prepare(0, ProbeScoreInput(**item)), tok


def encode(tok, messages, gp):
    """The expected ids, written out: `word_tokenizer`'s template begins with `<s>`, so the
    render is encoded WITHOUT the tokenizer's own BOS — one BOS, never two. Until 2026-10-08
    this helper was `tok(render)`, which pinned the duplicate BOS live serving produced."""
    ids = list(tok(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gp),
                   add_special_tokens=False)["input_ids"])
    assert ids[0] == tok.bos_token_id and ids[1] != tok.bos_token_id
    return ids


class TestKinds:
    def test_token_ids_are_used_as_given(self):
        p, _ = prep({"token_ids": [5, 6, 7], "prompt_tokens": 1})
        assert p.ids == [5, 6, 7] and p.prompt_tokens == 1 and p.input_kind == "token_ids"
        assert p.last_user_span is None and p.last_user_reason == "token_ids_have_no_turns"

    def test_bare_token_ids_have_no_boundary(self):
        p, _ = prep({"token_ids": [5, 6, 7]})
        assert p.prompt_tokens is None

    def test_a_user_ended_conversation_is_all_prompt(self):
        """TD6: no response exists, so the whole input is prompt; nothing is guessed."""
        msgs = [{"role": "user", "content": "w1 w2"}]
        p, tok = prep({"messages": msgs})
        assert p.ids == encode(tok, msgs, True)
        assert p.prompt_tokens == len(p.ids)

    def test_an_assistant_ended_conversation_derives_its_boundary(self):
        msgs = [{"role": "user", "content": "w1 w2"}, {"role": "assistant", "content": "w3 w4"}]
        p, tok = prep({"messages": msgs})
        full, head = encode(tok, msgs, False), encode(tok, msgs[:1], True)
        assert p.ids == full and full[: len(head)] == head
        assert p.prompt_tokens == len(head)

    def test_a_failed_prefix_check_leaves_the_boundary_unknown(self):
        """Never guessed: if the head render is not a prefix of the full one, no boundary."""
        tok = word_tokenizer()

        def drifting(messages, generation_prompt):
            text = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=generation_prompt)
            return text if not generation_prompt else "w9 " + text

        msgs = [{"role": "user", "content": "w1"}, {"role": "assistant", "content": "w2"}]
        p, _ = prep({"messages": msgs}, render=drifting)
        assert p.prompt_tokens is None and p.ids

    def test_text_is_one_user_turn(self):
        """T-49: rendered exactly as `messages` with one user turn."""
        p_text, _ = prep({"text": "w1 w2"})
        p_msgs, _ = prep({"messages": [{"role": "user", "content": "w1 w2"}]})
        assert p_text.ids == p_msgs.ids and p_text.input_kind == "text"
        assert p_text.prompt_tokens == p_msgs.prompt_tokens

    def test_last_user_span_comes_from_the_turns(self):
        msgs = [{"role": "user", "content": "w1"}, {"role": "assistant", "content": "w2"},
                {"role": "user", "content": "w3 w4 w5"}]
        p, _ = prep({"messages": msgs})
        assert p.last_user_span is not None and p.last_user_reason is None
        start, end = p.last_user_span
        assert end - start >= 3

    def test_a_render_failure_is_a_per_input_error(self):
        def broken(messages, generation_prompt):
            raise RuntimeError("template exploded")

        p, _ = prep({"messages": [{"role": "user", "content": "w1"}]}, render=broken)
        assert p.error["code"] == TOKENIZATION_FAILED and p.ids == []


def test_kinds_reports_every_kind_present():
    item = ProbeScoreInput(token_ids=[1], text="x")
    assert item.kinds() == ["token_ids", "text"]
