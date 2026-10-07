"""Generation entry points of `InferenceService`, DISCOVERED from its AST (FR-27.9; shared with
Feature 28, FTASKS 5.8 / 7.3).

A *generation site* is a method whose body — nested closures included — calls, or passes as a
callable, a generation primitive. Entry points are the public coroutines from which a site is
reachable through the `self.<method>` call graph. ⚠ Calls are matched in the AST, never names in
the source text, so a comment naming a primitive cannot satisfy a guard.

`SCENARIOS` says how to reach each site; consumers check it for EQUALITY against discovery, so a
new site fails red until somebody writes the request that reaches it. Feature 27 (probe contexts)
and Feature 28 (steering reports) both drive every site through it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import millm.services.inference_service as inference_module
from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage, TextCompletionRequest

#: The generation primitives. Methods of the service, and attribute chains under `self`.
PRIMITIVE_METHODS = {"_generate_sync", "_generate_in_thread", "_llamacpp_sync"}
PRIMITIVE_ATTRS = {
    ("_cbm_backend", "generate"),
    ("_cbm_backend", "generate_stream"),
    ("_model", "generate"),
    ("_model", "create_completion"),
    ("_model", "create_chat_completion"),
}
KNOWN_ENTRY_POINTS = {"create_chat_completion", "stream_chat_completion", "create_text_completion"}

#: Methods that run forward passes but never generate. Each names why it is exempt; a test below
#: asserts none of them is a generation site, so an exemption can never hide generation.
EXEMPT = {
    "_score_prompts": "scoring mode runs one unsteered forward per prompt and never generates "
                      "(FR-25.7.4); it opens no probe context by design (X-09)",
    "_unsteered_next_token_logits": "a single scoring forward, no generation",
    "_next_token_logits": "a single scoring forward, no generation",
    "create_embeddings": "embeddings run a forward pass and return vectors, never tokens",
    "_llamacpp_embeddings": "llama.cpp embeddings, never tokens",
    "run_model_work": "non-generation model work (probe scoring, parity, arm), unsteered",
}


def _class_methods() -> dict[str, ast.AST]:
    tree = ast.parse(Path(inference_module.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "InferenceService":
            return {
                item.name: item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError("InferenceService not found in its own module")


def _is_self_attr(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _is_primitive(node: ast.AST) -> bool:
    if any(_is_self_attr(node, name) for name in PRIMITIVE_METHODS):
        return True
    if isinstance(node, ast.Attribute):
        for owner, attr in PRIMITIVE_ATTRS:
            if node.attr == attr and _is_self_attr(node.value, owner):
                return True
    return False


def _referenced(method: ast.AST, predicate) -> list[ast.AST]:
    """Every node that is a call's function OR an argument of any call, anywhere in `method`
    (ast.walk descends into nested closures, which is what `_complete` and `_open_stream` need)."""
    found = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        candidates = [node.func, *node.args, *(kw.value for kw in node.keywords)]
        found.extend(c for c in candidates if predicate(c))
    return found


def discover_sites() -> set[str]:
    return {
        name for name, node in _class_methods().items()
        if name not in PRIMITIVE_METHODS and _referenced(node, _is_primitive)
    }


def discover_entry_points() -> set[str]:
    methods = _class_methods()
    graph = {
        name: {
            ref.attr for ref in _referenced(
                node, lambda c: isinstance(c, ast.Attribute) and isinstance(c.value, ast.Name)
                and c.value.id == "self" and c.attr in methods
            )
        }
        for name, node in methods.items()
    }
    sites = discover_sites()

    def reaches(start: str) -> bool:
        seen, stack = set(), [start]
        while stack:
            current = stack.pop()
            if current in sites:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(graph.get(current, ()))
        return False

    return {
        name for name, node in methods.items()
        if not name.startswith("_") and isinstance(node, ast.AsyncFunctionDef) and reaches(name)
    }




class EventLog:
    """One lock-guarded list; spies, site wrappers and fakes all append to it."""

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self.events: list[tuple[str, Any]] = []

    def add(self, kind: str, payload: Any = None) -> None:
        with self._lock:
            self.events.append((kind, payload))


class FakeCBM:
    """The continuous-batching backend at its boundary: records `gen`, returns a shaped result."""

    is_running = True
    _default_temperature = 1.0
    _default_top_p = 1.0

    def __init__(self, log: EventLog) -> None:
        self.log = log

    def sampling_params_match(self, temperature, top_p) -> bool:
        return True

    async def generate(self, input_ids, max_new_tokens, request_id):
        self.log.add("gen", "cbm.generate")
        return [7], "stop"

    async def generate_stream(self, input_ids, max_new_tokens, request_id):
        self.log.add("gen", "cbm.generate_stream")
        yield [7]


class FakeLlama:
    """llama.cpp's `Llama` at its boundary."""

    def __init__(self, log: EventLog) -> None:
        self.log = log

    def create_completion(self, prompt, stream=False, **params):
        self.log.add("gen", "llama.create_completion")
        return {"choices": [{"text": "w1", "finish_reason": "stop"}], "usage": {}}

    def create_chat_completion(self, messages, stream=False, **params):
        self.log.add("gen", "llama.create_chat_completion")
        if stream:
            return iter([
                {"choices": [{"delta": {"content": "w1"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ])
        return {"choices": [{"message": {"content": "w1"}, "finish_reason": "stop"}],
                "usage": {}}

    def tokenize(self, data):
        return [1, 2]


def _chat(**over) -> ChatCompletionRequest:
    base = dict(model="tiny", messages=[ChatMessage(role="user", content="w1 w2")],
                max_tokens=2, temperature=0.0)
    base.update(over)
    return ChatCompletionRequest(**base)


def _text(**over) -> TextCompletionRequest:
    base = dict(model="tiny", prompt="w1 w2", max_tokens=2, temperature=0.0)
    base.update(over)
    return TextCompletionRequest(**base)


#: Site -> how to reach it. KEYED BY SITE, and checked for equality against discovery.
SCENARIOS: dict[str, dict[str, Any]] = {
    "create_chat_completion": dict(call="chat", engine="transformers"),
    "stream_chat_completion": dict(call="stream", engine="transformers"),
    "create_text_completion": dict(call="text", engine="transformers"),
    "_generate_batch_chunk": dict(
        call="chat", engine="transformers",
        request=dict(extra_messages=[[ChatMessage(role="user", content="w3")]]),
    ),
    "_cbm_chat_completion": dict(call="chat", engine="transformers", cbm=True),
    "_cbm_stream_chat_completion": dict(call="stream", engine="transformers", cbm=True),
    "_cbm_text_completion": dict(call="text", engine="transformers", cbm=True),
    "_llamacpp_chat_completion": dict(call="chat", engine="llamacpp"),
    "_llamacpp_stream_chat_completion": dict(call="stream", engine="llamacpp"),
    "_llamacpp_text_completion": dict(call="text", engine="llamacpp"),
}


