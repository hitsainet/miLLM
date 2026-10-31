"""OpenAI function calling on the transformers engine (operator bug 2026-10-28).

THE DEFECT IT REMOVES. `tools` and `tool_choice` were refused on every cell of the request policy
("function calling is not implemented"), so Open WebUI in native function-calling mode could not
use any model served by miLLM. This module is the whole of the transformers engine's tool support:

* `plan_tools` decides, from the request and the resident tokenizer, what the template is given
  (`tools`) and what the assistant turn is opened with (`prefill`), or refuses with a reason;
* `template_messages` builds the message list the template sees, tool-call history included
  (an assistant message's `tool_calls`, a `role: tool` message's `tool_call_id`);
* `parse_tool_calls` turns the model's output into OpenAI `tool_calls`, or returns None so the
  caller serves the text as plain content — UNPARSEABLE OUTPUT IS NEVER AN ERROR;
* `ToolCallStreamFilter` does the same for a stream: content passes through until a call starts,
  then the call is withheld and emitted as one `tool_calls` delta at the end.

THE FORMATS, read off the real templates (LiquidAI/LFM2.5-1.2B-Instruct rev 0f604ada,
Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.1-8B-Instruct; 2026-10-28):

* `lfm2`   — `<|tool_call_start|>[name(arg='v', n=1)]<|tool_call_end|>` (Pythonic). The two marker
             tokens are NOT special in the LFM2.5 vocabulary, so they survive the
             `skip_special_tokens=True` decode every path uses.
* `hermes` — `<tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call>` (Qwen2.5 and other
             Hermes-style templates). `<tool_call>` is a NON-special added token in Qwen2.5.
* `llama3_json` — the whole reply is `{"name": ..., "parameters": {...}}` (Llama 3.1/3.2/3.3).
             `<|python_tag|>` is special and is removed by the decode; its text is tolerated too.

A template that takes tools in a format not listed (Gemma 4's `<|tool_call>call:...`, whose
markers are special tokens the decode removes) is REFUSED with the reason, never served with the
call flattened into prose.

`tool_choice`: `none` renders no tools (the model cannot call what it was not shown); `auto`
renders them and parses; `required` and a named function also OPEN the call in the assistant turn
(`prefill`), which is how a plain transformers model is made to call a tool — the forced prefix is
part of the prompt, and the parser reads `prefill + generated text`.

THE PROMPT A PROBE SEES. The served prompt is `render + prefill`. The render goes through
`prompt_encoding.encode_rendered_chat` like every chat (the template still begins with the BOS
text, so the double-BOS rule is unchanged). A `last_user` window is resolved against the RENDER
(tools included): `prefill_token_count` says how many trailing served ids are the prefill.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from millm.core.errors import FieldNotHonouredError

LFM2 = "lfm2"
HERMES = "hermes"
LLAMA3_JSON = "llama3_json"

#: Every recognised format, in detection order.
FORMATS: tuple[str, ...] = (LFM2, HERMES, LLAMA3_JSON)

_LFM2_START, _LFM2_END = "<|tool_call_start|>", "<|tool_call_end|>"
_HERMES_START, _HERMES_END = "<tool_call>", "</tool_call>"
_PYTHON_TAG = "<|python_tag|>"

#: The marker that opens a call in the generated text, per format (None: the reply IS the call).
START_MARKERS: dict[str, str | None] = {
    LFM2: _LFM2_START,
    HERMES: _HERMES_START,
    LLAMA3_JSON: None,
}

UNSUPPORTED_TEMPLATE = (
    "this model's chat template takes no tools, so they cannot be shown to the model"
)
UNRECOGNISED_FORMAT = (
    "miLLM does not recognise this model's tool-call output format, so a call could not be "
    "returned as tool_calls (recognised: LFM2 <|tool_call_start|>, Hermes/Qwen2.5 <tool_call>, "
    "Llama 3.1 JSON)"
)


@dataclass(frozen=True)
class ToolPlan:
    """What one request's tools change: the template's `tools`, the assistant turn's opening, and
    the parser to run on the reply."""

    tools: list[dict[str, Any]]
    fmt: str
    prefill: str = ""
    names: frozenset[str] = field(default_factory=frozenset)
    parallel: bool = True


def _template_text(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str):
        return template
    if isinstance(template, Mapping):  # named templates: any of them may be the tool one
        return "\n".join(str(v) for v in template.values())
    if isinstance(template, Sequence):
        return "\n".join(str(t.get("template", "")) if isinstance(t, Mapping) else str(t)
                         for t in template)
    return ""


def detect_format(template: str) -> str | None:
    """The tool-call output format a chat template teaches, or None when none is recognised."""
    if _LFM2_START in template:
        return LFM2
    if _HERMES_START in template:
        return HERMES
    if _PYTHON_TAG in template or ("ipython" in template and '"parameters"' in template):
        return LLAMA3_JSON
    return None


def template_takes_tools(template: str) -> bool:
    """Whether the template reads a `tools` variable at all (a template that does not would
    render the request exactly as without tools — the silent drop this module exists to end)."""
    return re.search(r"\btools\b", template) is not None


def _tool_name(tool: Any) -> str | None:
    function = tool.get("function") if isinstance(tool, Mapping) else None
    name = function.get("name") if isinstance(function, Mapping) else None
    return name if isinstance(name, str) else None


def _choice_name(tool_choice: Any) -> str | None:
    """The function a `tool_choice` forces, from the request model or a plain dict."""
    if isinstance(tool_choice, Mapping):
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
    else:
        name = getattr(getattr(tool_choice, "function", None), "name", None)
    return name if isinstance(name, str) else None


def _prefill(fmt: str, forced_name: str | None) -> str:
    """The text that opens a call in the assistant turn (`required`, or a named function)."""
    if fmt == LFM2:
        return _LFM2_START + "[" + (f"{forced_name}(" if forced_name else "")
    if fmt == HERMES:
        return _HERMES_START + "\n" + (
            '{"name": ' + json.dumps(forced_name) + ', "arguments": ' if forced_name else ""
        )
    # Llama 3.1: the reply is the JSON object itself.
    return '{"name": ' + (json.dumps(forced_name) + ', "parameters": ' if forced_name else '"')


def plan_tools(request: Any, tokenizer: Any) -> ToolPlan | None:
    """The request's tool plan, None when it renders no tools, or a 400 naming `tools`.

    Raises:
        FieldNotHonouredError: the template takes no tools, or its output format is not one this
            module can turn back into `tool_calls`.
    """
    tools = getattr(request, "tools", None)
    if not tools:
        return None
    choice = getattr(request, "tool_choice", None)
    if choice is None:
        choice = "auto"
    if choice == "none":
        return None
    template = _template_text(tokenizer)
    if not template or not template_takes_tools(template):
        raise FieldNotHonouredError(
            f"'tools' is not honoured on /v1/chat/completions: {UNSUPPORTED_TEMPLATE}",
            details={"param": "tools"},
        )
    fmt = detect_format(template)
    if fmt is None:
        raise FieldNotHonouredError(
            f"'tools' is not honoured on /v1/chat/completions: {UNRECOGNISED_FORMAT}",
            details={"param": "tools"},
        )
    rendered = [tool_dict(t) for t in tools]
    names = frozenset(n for n in (_tool_name(t) for t in rendered) if n)
    forced = _choice_name(choice)
    prefill = _prefill(fmt, forced) if (choice == "required" or forced) else ""
    parallel = getattr(request, "parallel_tool_calls", None) is not False
    return ToolPlan(tools=rendered, fmt=fmt, prefill=prefill, names=names, parallel=parallel)


def tool_dict(tool: Any) -> dict[str, Any]:
    """A tool as the plain dict a template expects (`{"type": "function", "function": {...}}`)."""
    if hasattr(tool, "model_dump"):
        dumped: dict[str, Any] = tool.model_dump(exclude_none=True)
        return dumped
    return dict(tool)


def _arguments_object(arguments: Any) -> Any:
    """History arguments as the MAPPING templates expect. OpenAI clients send a JSON string; the
    LFM2.5 template raises on one and Qwen2.5/Llama 3.1 would double-encode it (`| tojson`)."""
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except ValueError:
            return arguments  # not JSON: the template decides (LFM2.5 refuses it, loudly)
        return parsed if isinstance(parsed, dict) else arguments
    return {} if arguments is None else arguments


def _history_tool_calls(tool_calls: Iterable[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for call in tool_calls:
        call = dict(call) if isinstance(call, Mapping) else {}
        function = dict(call.get("function") or {})
        function["arguments"] = _arguments_object(function.get("arguments"))
        entry: dict[str, Any] = {"type": call.get("type") or "function", "function": function}
        if call.get("id") is not None:
            entry["id"] = call["id"]
        out.append(entry)
    return out


def template_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """The message dicts a chat template receives.

    `role` and `content` always (as before); `tool_calls` only on a message that carries them and
    `tool_call_id` only on one that has it — Llama 3.1's template tests `'tool_calls' in message`,
    so a key present with a null value would be rendered as a call.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        entry: dict[str, Any] = {"role": m.role, "content": m.content}
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = _history_tool_calls(tool_calls)
        tool_call_id = getattr(m, "tool_call_id", None)
        if tool_call_id is not None:
            entry["tool_call_id"] = tool_call_id
        out.append(entry)
    return out


def carries_tool_history(messages: Sequence[Any]) -> bool:
    """Whether any message is a tool result or an assistant tool call."""
    return any(
        getattr(m, "role", None) == "tool" or getattr(m, "tool_calls", None) for m in messages
    )


def prefill_token_count(tokenizer: Any, plan: ToolPlan | None) -> int:
    """How many trailing ids of a served prompt are the plan's prefill (0 without one)."""
    if plan is None or not plan.prefill:
        return 0
    return len(tokenizer(plan.prefill, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------


def _call(name: str, arguments: Any) -> dict[str, Any]:
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


_PY_LITERALS = {"true": True, "false": False, "null": None, "none": None}


def _py_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Name) and node.id.lower() in _PY_LITERALS:
        return _PY_LITERALS[node.id.lower()]
    return ast.literal_eval(node)


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _pythonic_calls(text: str) -> list[tuple[str, Any]] | None:
    """`[f(a=1, b='x'), g()]` (or one bare call) as (name, arguments) pairs, or None."""
    try:
        tree = ast.parse(text.strip(), mode="eval").body
    except SyntaxError:
        return None
    nodes = tree.elts if isinstance(tree, (ast.List, ast.Tuple)) else [tree]
    calls: list[tuple[str, Any]] = []
    for node in nodes:
        if not isinstance(node, ast.Call) or node.args:
            return None  # positional arguments have no names to map to parameters
        name = _dotted(node.func)
        if name is None:
            return None
        try:
            arguments = {kw.arg: _py_value(kw.value) for kw in node.keywords if kw.arg}
        except (ValueError, SyntaxError, TypeError):
            return None
        if len(arguments) != len(node.keywords):
            return None  # `**kwargs` has no literal meaning
        calls.append((name, arguments))
    return calls


def _json_calls(value: Any, keys: tuple[str, ...]) -> list[tuple[str, Any]] | None:
    items = value if isinstance(value, list) else [value]
    calls: list[tuple[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return None
        arguments: Any = None
        for key in keys:
            if key in item:
                arguments = item[key]
                break
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except ValueError:
                return None
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return None
        calls.append((item["name"], arguments))
    return calls


def _blocks(text: str, start: str, end: str) -> tuple[list[str], str]:
    """The bodies between `start` and `end` (the LAST block may be unterminated: generation can
    stop on EOS right after the call) and the text outside every block."""
    bodies: list[str] = []
    outside: list[str] = []
    pos = 0
    while True:
        i = text.find(start, pos)
        if i < 0:
            outside.append(text[pos:])
            break
        outside.append(text[pos:i])
        j = text.find(end, i + len(start))
        if j < 0:
            bodies.append(text[i + len(start):])
            break
        bodies.append(text[i + len(start):j])
        pos = j + len(end)
    return bodies, "".join(outside)


def _parse_lfm2(text: str) -> tuple[str, list[tuple[str, Any]]] | None:
    bodies, outside = _blocks(text, _LFM2_START, _LFM2_END)
    if not bodies:
        return None
    calls: list[tuple[str, Any]] = []
    for body in bodies:
        found = _pythonic_calls(body)
        if found is None:
            try:
                found = _json_calls(json.loads(body), ("arguments", "parameters"))
            except ValueError:
                found = None
        if not found:
            return None
        calls.extend(found)
    return outside, calls


def _parse_hermes(text: str) -> tuple[str, list[tuple[str, Any]]] | None:
    bodies, outside = _blocks(text, _HERMES_START, _HERMES_END)
    if not bodies:
        return None
    calls: list[tuple[str, Any]] = []
    for body in bodies:
        try:
            found = _json_calls(json.loads(body.strip()), ("arguments", "parameters"))
        except ValueError:
            return None
        if not found:
            return None
        calls.extend(found)
    return outside, calls


def _parse_llama3_json(text: str) -> tuple[str, list[tuple[str, Any]]] | None:
    body = text.strip()
    if body.startswith(_PYTHON_TAG):
        body = body[len(_PYTHON_TAG):].lstrip()
    if not body.startswith("{"):
        return None
    decoder = json.JSONDecoder()
    calls: list[tuple[str, Any]] = []
    pos = 0
    while pos < len(body):
        try:
            value, end = decoder.raw_decode(body, pos)
        except ValueError:
            return None
        found = _json_calls(value, ("parameters", "arguments"))
        if not found:
            return None
        calls.extend(found)
        pos = end
        while pos < len(body) and body[pos] in " \t\r\n;":
            pos += 1
    return "", calls


_PARSERS = {LFM2: _parse_lfm2, HERMES: _parse_hermes, LLAMA3_JSON: _parse_llama3_json}


@dataclass(frozen=True)
class ParsedReply:
    """A reply with its calls taken out: `content` is the text outside them (None when empty)."""

    content: str | None
    tool_calls: list[dict[str, Any]]


def parse_tool_calls(text: str | None, plan: ToolPlan) -> ParsedReply | None:
    """The calls in `text` (which must already include the plan's prefill), or None.

    None means "serve the text as content": no call found, a call that does not parse, or a call
    naming a function the request did not offer (a model answering with a JSON object is not
    calling a tool because the object has a `name` key).
    """
    if not text:
        return None
    try:
        parsed = _PARSERS[plan.fmt](text)
    except (RecursionError, MemoryError):  # pathological model output: content, not a 500
        return None
    if parsed is None:
        return None
    outside, calls = parsed
    if not calls or any(name not in plan.names for name, _ in calls):
        return None
    if not plan.parallel:
        calls = calls[:1]
    content = outside.strip() or None
    return ParsedReply(content=content, tool_calls=[_call(n, a) for n, a in calls])


def stream_deltas(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`tool_calls` in the streaming delta shape: each call with its `index`."""
    return [{"index": i, **call} for i, call in enumerate(tool_calls)]


class ToolCallStreamFilter:
    """Passes content through until a call may have started, then withholds the rest.

    `feed(piece)` returns the content to emit now ("" when everything is withheld); `finish()`
    returns `(content_to_emit, tool_calls)`: the parsed calls, or — when the withheld text does
    not parse — that text as content, so nothing the model wrote is lost.
    """

    def __init__(self, plan: ToolPlan) -> None:
        self.plan = plan
        self._marker = START_MARKERS[plan.fmt]
        # A forced call (prefill) is a call from the first token: withhold everything.
        self._held = plan.prefill
        self._holding = bool(plan.prefill)
        self._passthrough = False
        self._pending = ""  # a tail that may be the start of the marker

    def feed(self, piece: str | None) -> str:
        if not piece:
            return ""
        if self._holding:
            self._held += piece
            return ""
        if self._passthrough:
            return piece
        text = self._pending + piece
        self._pending = ""
        if self._marker is None:  # Llama 3.1 JSON: a call is the WHOLE reply, opening with `{`
            stripped = text.lstrip()
            if not stripped:
                self._pending = text
                return ""
            if stripped.startswith("{") or stripped.startswith(_PYTHON_TAG[: len(stripped)]):
                self._holding = True
                self._held = text
                return ""
            self._passthrough = True
            return text
        i = text.find(self._marker)
        if i >= 0:
            self._holding = True
            self._held = text[i:]
            return text[:i]
        # Keep back a tail that could be the first characters of the marker.
        for k in range(min(len(self._marker) - 1, len(text)), 0, -1):
            if self._marker.startswith(text[-k:]):
                self._pending = text[-k:]
                return text[:-k]
        return text

    def finish(self) -> tuple[str | None, list[dict[str, Any]]]:
        if not self._holding:
            tail, self._pending = self._pending, ""
            return (tail or None), []
        parsed = parse_tool_calls(self._held, self.plan)
        if parsed is None:
            return (self._held or None), []
        # Content outside the call that was withheld with it (e.g. text after the call).
        return parsed.content, parsed.tool_calls
