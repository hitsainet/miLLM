"""The served name is captured INSIDE the admission slot, and every response reports it
(032 FR-32.6.6-8, 033 review findings 6/17; addendum §2.1 "W0b").

Every response builder used to read the loader's name (`model_info.name`,
`_model_state.current.model_name`) — the non-streaming ones AFTER the slot was released, the
streaming ones BEFORE it was taken. Either way a weights load, a save or an adapter flip landing
outside the slot relabels an answer the old weights produced.

The race is made deterministic: a served-name overlay answers `before-slot` until the request
queue grants the slot, `in-slot` while it is held, and `after-slot` the moment it is released.
Every generation site (DISCOVERED, `tests/support/generation_entry_points.py`), the scoring
paths, the embeddings path and the serial batch fallback must report `in-slot` — in the body
`model`, in EVERY stream chunk, and in the terminal chunk's `millm_model`. The continuous-
batching paths hold no slot (D-13); they capture once at entry, so the overlay flips during
generation instead.

MUTATION CONTROLS (recorded in the W0b report):
  C1  capture moved out of the slot (before `acquire`)        -> every non-CBM case
  C2  one streaming read moved back before `_admit`            -> the stream cases
  C3  `_steering_stream_chunk` drops `millm_model`             -> the stream cases
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import patch

import pytest

from millm.api.schemas.openai import EmbeddingRequest
from millm.ml.model_loader import ENGINE_LLAMACPP, LoadedModel, LoadedModelState
from millm.services import resident_identity
from millm.services.inference_service import (
    ServedModel,
    current_served_name,
    reset_steering_memo,
    served_model_header_value,
    structured_field_string,
)
from tests.support.generation_entry_points import (
    SCENARIOS,
    EventLog,
    FakeCBM,
    FakeLlama,
    _chat,
    _text,
    discover_sites,
)
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer
from tests.unit.f28_fixtures import (  # noqa: F401 - fixtures
    clean_state,
    db,
)


class _Phase:
    """The overlay's answer, moved by the queue wrapper (or the fake CBM)."""

    def __init__(self, start: str) -> None:
        self.name = start

    def __call__(self) -> str:
        return self.name


@pytest.fixture
def phase():
    p = _Phase("before-slot")
    resident_identity.register_served_name_overlay(p)
    try:
        yield p
    finally:
        with resident_identity._lock:
            resident_identity._overlays.remove(p)
        clear_loaded()


def _flip_on_slot(svc, phase: _Phase) -> None:
    """`in-slot` once the queue grants a slot, `after-slot` the moment it is released."""
    queue = svc._request_queue
    for attr in ("acquire", "acquire_background"):
        real = getattr(queue, attr)

        def wrapped(*a, _real=real, **k):
            @asynccontextmanager
            async def cm():
                async with _real(*a, **k):
                    phase.name = "in-slot"
                    try:
                        yield
                    finally:
                        phase.name = "after-slot"
            return cm()

        setattr(queue, attr, wrapped)


class _FlippingCBM(FakeCBM):
    """CBM holds no slot: the name flips DURING generation, after the entry capture."""

    def __init__(self, log, phase):
        super().__init__(log)
        self.phase = phase

    async def generate(self, input_ids, max_new_tokens, request_id):
        self.phase.name = "after-slot"
        return await super().generate(input_ids, max_new_tokens, request_id)

    async def generate_stream(self, input_ids, max_new_tokens, request_id):
        self.phase.name = "after-slot"
        async for item in super().generate_stream(input_ids, max_new_tokens, request_id):
            yield item


def _build(scenario: dict, phase: _Phase):
    svc = make_service(word_model(), word_tokenizer())
    log = EventLog()
    if scenario.get("cbm"):
        phase.name = "in-slot"  # captured at entry: there is no slot to wait for
        svc._cbm_backend = _FlippingCBM(log, phase)
    if scenario["engine"] == "llamacpp":
        LoadedModelState().set(LoadedModel(
            model_id=1, model_name="tiny", model=FakeLlama(log), tokenizer=None,
            loaded_at=datetime(2026, 10, 6), engine=ENGINE_LLAMACPP,
        ))
    _flip_on_slot(svc, phase)
    return svc


async def _drive(svc, scenario) -> tuple[object, list[str]]:
    extra = scenario.get("request", {})
    chunks: list[str] = []
    result = None
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        if scenario["call"] == "chat":
            result = await svc.create_chat_completion(_chat(**extra))
        elif scenario["call"] == "stream":
            async for chunk in svc.stream_chat_completion(_chat(stream=True, **extra)):
                chunks.append(chunk)
        elif scenario["call"] == "text":
            result = await svc.create_text_completion(_text(**extra))
        elif scenario["call"] == "embeddings":
            result = await svc.create_embeddings(EmbeddingRequest(model="tiny", input="w1 w2"))
    return result, chunks


def _payloads(chunks: list[str]) -> list[dict]:
    out = []
    for chunk in chunks:
        body = chunk[len("data: "):].strip()
        if body and body != "[DONE]":
            out.append(json.loads(body))
    return out


#: Paths that are not generation sites (so not in SCENARIOS) but build a response.
EXTRA_CASES = {
    "scoring_text": dict(call="text", engine="transformers", request=dict(max_tokens=1,
                                                                         logprobs=2)),
    "scoring_chat": dict(call="chat", engine="transformers", request=dict(max_tokens=1,
                                                                         logprobs=True)),
    "embeddings": dict(call="embeddings", engine="transformers"),
    "embeddings_llamacpp": dict(call="embeddings", engine="llamacpp"),
}

CASES = {**SCENARIOS, **EXTRA_CASES}


def test_the_case_table_covers_every_discovered_generation_site():
    sites = discover_sites()
    assert sites and sites <= set(CASES), sorted(sites - set(CASES))


@pytest.mark.parametrize("case", sorted(CASES))
async def test_every_response_reports_the_name_captured_inside_the_slot(
    case, phase, clean_state, db,  # noqa: F811
):
    scenario = CASES[case]
    reset_steering_memo()
    svc = _build(scenario, phase)
    if case == "embeddings_llamacpp":
        svc._model.create_embedding = lambda text: {
            "data": [{"embedding": [0.1, 0.2]}], "usage": {"prompt_tokens": 2}}
        svc._embedding_limit = lambda: 64
    result, chunks = await _drive(svc, scenario)

    assert phase.name == "after-slot", "precondition: the name moved after the slot/capture"
    if scenario["call"] == "stream":
        payloads = _payloads(chunks)
        assert payloads, f"{case}: no chunks"
        named = [p for p in payloads if "model" in p]
        assert named and all(p["model"] == "in-slot" for p in named), (
            case, [p.get("model") for p in named])
        terminal = [p for p in payloads if "millm_steering" in p]
        assert len(terminal) == 1, case
        assert terminal[0]["millm_model"].startswith('"in-slot";origin=model'), terminal[0]
    else:
        assert result.model == "in-slot", (case, result.model)
        assert current_served_name() == "in-slot"


async def test_the_serial_batch_fallback_reports_the_capture(phase, clean_state, db):  # noqa: F811
    """`_serial_chat_fallback` builds its response after N separate slots."""
    from millm.api.schemas.openai import ChatMessage

    svc = make_service(word_model(), word_tokenizer())
    _flip_on_slot(svc, phase)
    request = _chat(extra_messages=[[ChatMessage(role="user", content="w3")]])
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        response = await svc._serial_chat_fallback(
            request, [request.messages, [ChatMessage(role="user", content="w3")]], "chatcmpl-x", 0
        )
    assert phase.name == "after-slot"
    assert response.model == "in-slot"


async def test_reentry_by_the_slot_owner_recaptures(phase):
    """A batch row enters `_admit` re-entrantly inside its chunk's slot; it captures too."""
    svc = make_service(word_model(), word_tokenizer())
    phase.name = "chunk"
    async with svc._admit(background=True):
        assert current_served_name() == "chunk"
        phase.name = "row"
        async with svc._admit():
            assert current_served_name() == "row"


async def test_nothing_loaded_captures_nothing(phase):
    svc = make_service(word_model(), word_tokenizer())
    clear_loaded()
    async with svc._admit():
        assert current_served_name() is None


class TestHeaderValue:
    def test_plain(self):
        assert served_model_header_value(ServedModel("m")) == '"m";origin=model'

    def test_commits_and_origin(self):
        value = served_model_header_value(ServedModel("a:b:step-4", "checkpoint", 3))
        assert value == '"a:b:step-4";origin=checkpoint;commits=3'

    def test_unrecorded_commits_are_omitted_never_zero(self):
        assert "commits" not in served_model_header_value(ServedModel("m", "checkpoint", None))
        assert served_model_header_value(ServedModel("m", "model", 0)).endswith(";commits=0")

    @pytest.mark.parametrize("raw, expected", [
        ('a"b', '"a\\"b"'),
        ("a\\b", '"a\\\\b"'),
        ('x\\"', '"x\\\\\\""'),
        ("line\r\nX-Evil: 1", '"line??X-Evil: 1"'),
        ("tab\there", '"tab?here"'),
        ("café-模型", '"caf?-??"'),
        ("\x7f\x00", '"??"'),
        ("plain-Name_1.2:3/4", '"plain-Name_1.2:3/4"'),
    ])
    def test_escaping(self, raw, expected):
        assert structured_field_string(raw) == expected

    def test_escaped_value_is_a_legal_header_and_parses_back(self):
        """Round trip through RFC 8941 sf-string unescaping recovers the printable input."""
        raw = 'we"ird\\naïme'
        value = structured_field_string(raw)
        assert value.isascii() and all(" " <= c <= "~" for c in value)
        inner, out, i = value[1:-1], [], 0
        while i < len(inner):
            if inner[i] == "\\":
                out.append(inner[i + 1])
                i += 2
            else:
                assert inner[i] != '"'
                out.append(inner[i])
                i += 1
        assert "".join(out) == 'we"ird\\na?me'


def test_no_response_field_reads_the_loaders_name():
    """Structural backstop for the behavioural cases above: no `model=` keyword and no `"model"`
    dict key in the service is built from `model_info.name` or `.model_name` (the loader's row
    name). The only reader left is `_response_model_name`'s fallback."""
    import ast
    import inspect

    from millm.services import inference_service

    tree = ast.parse(inspect.getsource(inference_service))

    def reads_loader(node) -> bool:
        return any(
            isinstance(n, ast.Attribute) and (
                n.attr == "model_name"
                or (n.attr == "name" and isinstance(n.value, ast.Name)
                    and n.value.id == "model_info")
            )
            for n in ast.walk(node)
        )

    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            offenders += [kw.value.lineno for kw in node.keywords
                          if kw.arg == "model" and reads_loader(kw.value)]
        if isinstance(node, ast.Dict):
            offenders += [v.lineno for k, v in zip(node.keys, node.values)
                          if isinstance(k, ast.Constant) and k.value == "model"
                          and reads_loader(v)]
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "model_name" for t in node.targets
        ) and reads_loader(node.value):
            offenders.append(node.lineno)
    assert offenders == [], f"response fields read the loader's name at lines {offenders}"
