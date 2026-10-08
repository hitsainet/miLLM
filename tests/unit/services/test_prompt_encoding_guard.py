"""No tokenize call in `millm/` bypasses `prompt_encoding` (the double-BOS fix, 2026-10-08).

Every live, scoring and probe path that turns a chat into ids must go through
`millm.services.prompt_encoding`, because the defect was a path that tokenized a template render
with the tokenizer's default and so prepended a SECOND begin-of-text token. Eight such call sites
existed and agreed with each other, wrongly.

Two checks, both on the AST — a CALL, never a name in the source text, because comments and
docstrings in this repo name `tokenizer(prompt)` and `encode_rendered_chat` constantly:

1. DISCOVERY. Every direct call of a tokenizer (`tokenizer(...)`, `self._tokenizer(...)`,
   `tok.encode(...)`, ...) anywhere under `millm/` is found, and the set must EQUAL `ALLOWED`,
   each entry naming why that call may tokenize directly. A new direct call fails red until it
   is routed through the helper or justified here.
2. WIRING. Each function that serves, scores or probes a prompt must CALL the helper it is meant
   to — so a site that silently stops tokenizing (or moves) is caught too, not only one that
   tokenizes the wrong way.
"""

from __future__ import annotations

import ast
from pathlib import Path

import millm

ROOT = Path(millm.__file__).resolve().parent
TOKENIZER_METHODS = {"encode", "encode_plus", "batch_encode_plus", "__call__"}

#: (module path under millm/, enclosing qualname) -> why it may call a tokenizer directly.
ALLOWED = {
    ("services/prompt_encoding.py", "encode_rendered_chat"):
        "THE rule: add_special_tokens is decided by rendered_chat_adds_special_tokens",
    ("services/prompt_encoding.py", "encode_rendered_chats"):
        "THE rule, for a padded batch",
    ("services/prompt_encoding.py", "encode_prompt"):
        "raw text: the request's own add_special_tokens; chat delegates to encode_rendered_chat",
    ("services/inference_service.py", "InferenceService._embed_inputs"):
        "embedding input is raw text, never a template render",
    ("services/probe_turns.py", "_encode_render"):
        "span render (add_special_tokens=False, as miStudio computes it); placed in the served "
        "ids by OFFSET, so it holds whatever BOS serving uses",
    ("services/probe_turns.py", "_user_header_ids"):
        "header isolation: a difference of two renders, BOS-independent",
    ("services/probe_turns.py", "last_user_token_span.encode"):
        "span render, placed by offset (see _encode_render)",
}

#: Functions that must CALL the named helper.
REQUIRED_CALLS = {
    ("services/inference_service.py", "InferenceService.check_stream_admission"):
        "encode_rendered_chat",
    ("services/inference_service.py", "InferenceService.count_prompt_tokens"):
        "encode_rendered_chat",
    ("services/inference_service.py", "InferenceService.create_chat_completion"):
        "encode_rendered_chat",
    ("services/inference_service.py", "InferenceService.stream_chat_completion"):
        "encode_rendered_chat",
    ("services/inference_service.py", "InferenceService._generate_batch_chunk"):
        "encode_rendered_chats",
    ("services/inference_service.py", "InferenceService._chunk_batch_for_memory"):
        "encode_rendered_chat",
    ("services/inference_service.py", "InferenceService._cbm_chat_completion"):
        "encode_rendered_chat",
    ("services/inference_service.py", "InferenceService._cbm_stream_chat_completion"):
        "encode_rendered_chat",
    ("services/inference_service.py", "InferenceService._score_prompts"): "encode_prompt",
    ("services/inference_service.py", "InferenceService._score_specs_packed"): "encode_prompt",
    ("services/inference_service.py", "InferenceService.create_text_completion"): "encode_prompt",
    ("services/inference_service.py", "InferenceService._cbm_text_completion"): "encode_prompt",
    ("services/inference_service.py", "InferenceService._llamacpp_continuation_prompt"):
        "llamacpp_completion_prompt",
    ("services/probe_scoring.py", "ProbeInputPreparer._encode"): "rendered_chat_ids",
}


def _is_tokenizer_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "tok" or node.id.endswith("tokenizer")
    if isinstance(node, ast.Attribute):
        return node.attr.endswith("tokenizer")
    return False


def _is_tokenize_call(call: ast.Call) -> bool:
    func = call.func
    if _is_tokenizer_expr(func):
        return True
    return (isinstance(func, ast.Attribute) and func.attr in TOKENIZER_METHODS
            and _is_tokenizer_expr(func.value))


class _Walker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.tokenize_sites: set[str] = set()
        self.calls: dict[str, list[ast.Call]] = {}

    def _scope(self, node, name):
        self.stack.append(name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node):
        self._scope(node, node.name)

    def visit_FunctionDef(self, node):
        self._scope(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        qual = ".".join(self.stack) or "<module>"
        if _is_tokenize_call(node):
            self.tokenize_sites.add(qual)
        # Calls are attributed to EVERY enclosing function, so a call inside a nested closure
        # still counts for the method that owns it.
        for depth in range(1, len(self.stack) + 1):
            self.calls.setdefault(".".join(self.stack[:depth]), []).append(node)
        self.generic_visit(node)


def _walk(path: Path) -> _Walker:
    walker = _Walker()
    walker.visit(ast.parse(path.read_text(), filename=str(path)))
    return walker


def discover_tokenize_sites() -> set[tuple[str, str]]:
    found = set()
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        for qual in _walk(path).tokenize_sites:
            found.add((rel, qual))
    return found


def _called_names(rel: str, qual: str) -> set[str]:
    calls = _walk(ROOT / rel).calls.get(qual)
    assert calls is not None, f"{rel}::{qual} no longer exists — update the guard, do not drop it"
    names = set()
    for call in calls:
        if isinstance(call.func, ast.Name):
            names.add(call.func.id)
        elif isinstance(call.func, ast.Attribute):
            names.add(call.func.attr)
    return names


def test_discovery_finds_the_known_sites():
    """A parser that finds nothing would pass the equality check below by checking nothing."""
    sites = discover_tokenize_sites()
    assert ("services/inference_service.py", "InferenceService._embed_inputs") in sites
    assert len(sites) >= 5


def test_every_direct_tokenize_call_is_accounted_for():
    sites = discover_tokenize_sites()
    allowed = set(ALLOWED)
    assert sites == allowed, (
        f"direct tokenizer calls with no justification — route them through "
        f"millm.services.prompt_encoding: {sorted(sites - allowed)}; "
        f"stale entries: {sorted(allowed - sites)}"
    )


def test_every_serving_path_calls_its_helper():
    missing = {
        f"{rel}::{qual}": helper
        for (rel, qual), helper in REQUIRED_CALLS.items()
        if helper not in _called_names(rel, qual)
    }
    assert not missing, f"paths that no longer call their prompt_encoding helper: {missing}"


def test_the_span_renders_stay_without_special_tokens():
    """`probe_turns` renders with `add_special_tokens=False` and places the span by offset — the
    construction miStudio uses. A True there would shift every boundary by the BOS."""
    for path_qual in [k for k in ALLOWED if k[0] == "services/probe_turns.py"]:
        rel, qual = path_qual
        tokenize = [c for c in _walk(ROOT / rel).calls[qual] if _is_tokenize_call(c)]
        assert tokenize, f"{qual}: no tokenize call"
        for call in tokenize:
            kw = {k.arg: k.value for k in call.keywords}
            assert isinstance(kw.get("add_special_tokens"), ast.Constant) and (
                kw["add_special_tokens"].value is False), f"{qual}: add_special_tokens is not False"


def _keyword_true(call: ast.Call, name: str) -> bool:
    return any(k.arg == name and isinstance(k.value, ast.Constant) and k.value.value is True
               for k in call.keywords)


def test_chat_scoring_marks_its_texts_as_renders():
    calls = _walk(ROOT / "services/inference_service.py").calls[
        "InferenceService._score_chat_completion"]
    scorer = [c for c in calls
              if isinstance(c.func, ast.Attribute) and c.func.attr == "_score_prompts"]
    assert len(scorer) == 1 and _keyword_true(scorer[0], "rendered_chat")


def test_batched_chat_scoring_marks_its_texts_as_renders():
    calls = _walk(ROOT / "services/batch/packing.py").calls["run_packed_scoring"]
    specs = [c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "ScoreSpec"]
    assert len(specs) == 2, "one chat spec and one text spec"
    assert sum(_keyword_true(c, "rendered_chat") for c in specs) == 1
