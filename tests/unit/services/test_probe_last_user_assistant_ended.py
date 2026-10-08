"""`last_user` resolves on an ASSISTANT-ended `/api/probes/score` input (2026-10-08).

THE DEFECT. `/api/probes/score` serves an assistant-ended conversation WITHOUT the generation prompt
(`probe_scoring.served_render`, T-72), but `probe_turns.last_user_token_span` placed the span by
matching the served ids against a render of the whole conversation WITH it — hard-coded. The two
renders differ by the generation prompt, so the match failed and the `last_user` window reported
`last_user_span_unresolved` for every assistant-ended input. Live chats were unaffected: they are
always rendered with the generation prompt, which is what the hard-coded value happened to be.

THE FIX. `last_user_token_span` takes the render that made the ids as a REQUIRED keyword,
`generation_prompt`; scoring passes the value `served_render` used, and live serving passes True.

The span itself is miStudio's: message i owns `[len(render(m[:i])), len(render(m[:i+1])))` under
renders WITHOUT the generation prompt (`probe_monitor_render.render_messages`), the newest user
message from its own role header. Computed longhand here rather than through the code under test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from millm.api.schemas.probe_scoring import ProbeScoreInput
from millm.services.inference_service import InferenceService
from millm.services.probe_scoring import ProbeInputPreparer, template_renderer
from millm.services.probe_turns import last_user_token_span

WORDS = ["<s>", "[UNK]", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>",
         "a", "b", "x", "y"] + [f"w{i}" for i in range(20)]
TEMPLATE = (
    "<s> {% for m in messages %}<|{{ m.role }}|> {{ m.content }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)


@pytest.fixture(scope="module", params=["template_bos_only", "template_and_tokenizer_bos"])
def tok(request) -> PreTrainedTokenizerFast:
    """A real fast tokenizer and Jinja template; the second variant also prepends a BOS, so the
    one-BOS rule is exercised under the span too."""
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if request.param == "template_and_tokenizer_bos":
        backend.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", 0)]
        )
    fast = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", unk_token="[UNK]")
    fast.chat_template = TEMPLATE
    return fast


MULTI_ASSISTANT_ENDED = [
    {"role": "system", "content": "w1"},
    {"role": "user", "content": "w2 w3"},
    {"role": "assistant", "content": "w4"},
    {"role": "user", "content": "w5 w6 w7"},
    {"role": "assistant", "content": "w8 w9"},
]
SINGLE_ASSISTANT_ENDED = [{"role": "user", "content": "w5 w6 w7"}, {"role": "assistant", "content": "w8"}]
USER_ENDED = MULTI_ASSISTANT_ENDED[:4]


def _tokens(tok, ids, span):
    return tok.convert_ids_to_tokens(ids[span[0]: span[1]])


def _prepare(tok, messages):
    return ProbeInputPreparer(tok, template_renderer(tok)).prepare(
        0, ProbeScoreInput(messages=messages)
    )


class TestScoringResolvesAssistantEnded:
    def test_multi_turn_assistant_ended_resolves_to_the_newest_user_turn(self, tok):
        p = _prepare(tok, MULTI_ASSISTANT_ENDED)
        assert p.error is None
        assert p.last_user_reason is None, p.last_user_reason
        assert _tokens(tok, p.ids, p.last_user_span) == ["<|user|>", "w5", "w6", "w7", "<|end|>"]

    def test_the_span_is_mistudios_message_span(self, tok):
        """Longhand: the newest user message owns `[len(render(m[:3])), len(render(m[:4])))`
        under renders WITHOUT the generation prompt — miStudio's construction."""
        p = _prepare(tok, MULTI_ASSISTANT_ENDED)

        def n(conv):
            text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            return len(tok(text, add_special_tokens=False)["input_ids"])

        assert p.ids[0] == 0 and p.ids[1] != 0, "one BOS"
        assert p.last_user_span == (n(MULTI_ASSISTANT_ENDED[:3]), n(MULTI_ASSISTANT_ENDED[:4]))

    def test_a_single_turn_assistant_ended_input_starts_at_the_user_header(self, tok):
        """Message 0 holds the BOS too; the window starts at its own header (review round 1, H1)."""
        p = _prepare(tok, SINGLE_ASSISTANT_ENDED)
        assert p.last_user_reason is None, p.last_user_reason
        assert _tokens(tok, p.ids, p.last_user_span) == ["<|user|>", "w5", "w6", "w7", "<|end|>"]

    def test_user_ended_still_resolves(self, tok):
        p = _prepare(tok, USER_ENDED)
        assert p.last_user_reason is None
        assert _tokens(tok, p.ids, p.last_user_span) == ["<|user|>", "w5", "w6", "w7", "<|end|>"]
        assert tok.convert_ids_to_tokens(p.ids)[-1] == "<|assistant|>", "served WITH the prompt"

    def test_the_wrong_render_does_not_resolve(self, tok):
        """The defect, pinned from the other side: ids of the assistant-ended served render, span
        computed against a render WITH the generation prompt, must refuse rather than guess."""
        p = _prepare(tok, MULTI_ASSISTANT_ENDED)
        span, reason = last_user_token_span(
            tok, MULTI_ASSISTANT_ENDED, p.ids, None, generation_prompt=True
        )
        assert span is None and reason == "last_user_span_unresolved"

    def test_the_render_has_no_default(self, tok):
        with pytest.raises(TypeError):
            last_user_token_span(tok, USER_ENDED, [0])  # type: ignore[call-arg]


class TestLiveChatsAreUnchanged:
    """A live chat is ALWAYS rendered with the generation prompt, whatever its last role, so the
    live caller must keep passing True — deriving it from the last role (the scoring rule) would
    break a live chat that ends on an assistant turn."""

    @pytest.mark.parametrize("messages", [USER_ENDED, MULTI_ASSISTANT_ENDED], ids=["user", "assistant"])
    def test_the_live_hook_resolves_the_span(self, tok, messages):
        from millm.services.prompt_encoding import rendered_chat_ids

        ids = rendered_chat_ids(
            tok, tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
        recorded: list = []
        context = SimpleNamespace(
            probes=[SimpleNamespace(windows=("last_user",))],
            set_last_user_span=lambda span, reason: recorded.append((span, reason)),
        )
        InferenceService._probe_note_last_user_span(
            SimpleNamespace(_tokenizer=tok), context,
            [SimpleNamespace(role=m["role"], content=m["content"]) for m in messages], ids, None,
        )
        (span, reason), = recorded
        assert reason is None, reason
        assert _tokens(tok, ids, span) == ["<|user|>", "w5", "w6", "w7", "<|end|>"]


# ── the real high-stakes definition's assistant-ended vectors, on the real tokenizer ─────────


def test_real_assistant_ended_vectors_resolve_last_user():
    """`pm_f736aa73969d`'s eight assistant-ended vectors (one keep-tail truncated), through the scoring preparer with the
    real Llama-3.1 template: the ids reproduce the recorded ones (the truncated one as its tail) and the span resolves to the
    newest user turn, header to end-of-turn."""
    from tests.unit.services.test_probe_parity_render import DEFINITIONS, _real_tokenizer

    d = DEFINITIONS["pm_f736aa73969d"]
    llama = _real_tokenizer(d["model"]["chat_template_sha256"])
    checked = truncated = 0
    for vector in d["test_vectors"]["vectors"]:
        messages = vector["messages"]
        if messages[-1]["role"] != "assistant":
            continue
        p = _prepare(llama, messages)
        recorded = vector["token_ids"]
        if len(p.ids) > len(recorded):
            # Keep-tail truncated to the producer's cap; the span is over the full input served.
            assert p.ids[-len(recorded):] == recorded
            truncated += 1
        else:
            assert p.ids == recorded
        assert p.last_user_reason is None, p.last_user_reason
        text = llama.decode(p.ids[p.last_user_span[0]: p.last_user_span[1]])
        last_user = [m for m in messages if m["role"] == "user"][-1]["content"]
        assert text.startswith("<|start_header_id|>user<|end_header_id|>")
        assert text.endswith("<|eot_id|>") and last_user.strip() in text
        checked += 1
    assert (checked, truncated) == (8, 1)
