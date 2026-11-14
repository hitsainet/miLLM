"""Source scanner shared by the generality guard and the brand guard (040 FR-040.15, TID §3.5.5).

Moved unchanged from ``tests/unit/test_generality_guard.py`` (038 FR-38.1): Python by abstract syntax
tree (docstrings and bare string statements are prose and skipped), a minimal TypeScript lexer
(comments dropped), YAML and JSON by parsed value (JSON-Schema prose keys skipped), ``.env.example``
keys and values. 040 adds a Markdown extractor, a file-name extractor and a pathspec on the lister.
Each guard keeps its own matching rule; this module only yields ``(line, text)`` items and lists files.

``prose=True`` (040 P4 follow-up, 2026-11-11) also yields the values of the prose keys. The generality
guard reads executable tokens and leaves them out; the brand guard reads what people read and asks for
them: with the values skipped, a dashboard's ``description`` and a manual page's front-matter ``title``
could name an old brand and no finding was written.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

#: JSON-Schema keys whose values are prose.
PROSE_KEYS = frozenset({"description", "title", "$comment", "markdownDescription"})


# ── tokens ───────────────────────────────────────────────────────────────────────

_CASE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Split on non-alphanumerics, then on case and letter/digit boundaries; lower case."""
    out: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", text):
        out.extend(part.lower() for part in _CASE.findall(chunk))
    return out


# ── extractors: each yields (line, text) for every executable token source ─────────


def python_items(source: str) -> Iterator[tuple[int, str]]:
    tree = ast.parse(source)
    prose: set[int] = set()
    for node in ast.walk(tree):
        # Docstrings and every other bare string statement are prose, not executable.
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            prose.add(id(node.value))
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in prose:
                yield line, node.value
        elif isinstance(node, ast.Name):
            yield line, node.id
        elif isinstance(node, ast.Attribute):
            yield line, node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield line, node.name
        elif isinstance(node, ast.arg):
            yield line, node.arg
        elif isinstance(node, ast.keyword) and node.arg:
            yield line, node.arg
        elif isinstance(node, ast.alias):
            yield line, node.name
            if node.asname:
                yield line, node.asname


def python_strings(source: str, docstrings: bool = False) -> Iterator[tuple[int, str]]:
    """Every string literal (f-string parts included); docstrings and other bare string statements
    only when `docstrings` (040 FR-040.15: under `miserver/api/` they feed the OpenAPI text)."""
    tree = ast.parse(source)
    prose: set[int] = set()
    if not docstrings:
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                prose.add(id(node.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose:
            yield getattr(node, "lineno", 0), node.value


def typescript_items(source: str) -> Iterator[tuple[int, str]]:
    """A minimal lexer: comments are dropped, everything else is yielded as text.

    Strings and template literals are yielded whole (their text is matched); `${...}` inside a
    template is lexed again as code. Everything outside comments and strings — identifiers, JSX
    text, attribute names — is yielded as code text.
    """
    i, n, line = 0, len(source), 1
    code_start, code_line = 0, 1

    def flush(end: int) -> Iterator[tuple[int, str]]:
        if end > code_start:
            yield code_line, source[code_start:end]

    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            yield from flush(i)
            j = source.find("\n", i)
            j = n if j < 0 else j
            i = j
            code_start, code_line = i, line
            continue
        if c == "/" and nxt == "*":
            yield from flush(i)
            j = source.find("*/", i + 2)
            j = n if j < 0 else j + 2
            line += source.count("\n", i, j)
            i = j
            code_start, code_line = i, line
            continue
        if c in ("'", '"'):
            yield from flush(i)
            j, buf = i + 1, []
            while j < n and source[j] != c:
                if source[j] == "\\" and j + 1 < n:
                    buf.append(source[j + 1])
                    j += 2
                    continue
                if source[j] == "\n":
                    break
                buf.append(source[j])
                j += 1
            yield line, "".join(buf)
            # An unterminated quote (an apostrophe in JSX text) ends at the line, which is then
            # counted by the newline branch rather than skipped.
            i = j if j < n and source[j] == "\n" else j + 1
            code_start, code_line = i, line
            continue
        if c == "`":
            yield from flush(i)
            j, buf, start_line = i + 1, [], line
            while j < n and source[j] != "`":
                if source[j] == "\\" and j + 1 < n:
                    buf.append(source[j + 1])
                    j += 2
                    continue
                if source[j] == "$" and j + 1 < n and source[j + 1] == "{":
                    depth, k = 1, j + 2
                    while k < n and depth:
                        depth += {"{": 1, "}": -1}.get(source[k], 0)
                        k += 1
                    for sub_line, text in typescript_items(source[j + 2:k - 1]):
                        yield line + sub_line - 1, text
                    line += source.count("\n", j, k)
                    j = k
                    continue
                if source[j] == "\n":
                    line += 1
                buf.append(source[j])
                j += 1
            yield start_line, "".join(buf)
            i = j + 1
            code_start, code_line = i, line
            continue
        if c == "\n":
            yield from flush(i)
            line += 1
            i += 1
            code_start, code_line = i, line
            continue
        i += 1
    yield from flush(n)


def derived_title(name: str) -> str:
    """The `title` a JSON-Schema generator writes for a property it was given no title for: the
    property's own name, underscores to spaces, each word capitalised (pydantic's default)."""
    return name.replace("_", " ").title()


def data_items(value: Any, path: str = "", prose: bool = False) -> Iterator[tuple[int, str]]:
    """Parsed YAML/JSON: every key and every scalar value. The values of the prose keys are skipped
    unless `prose` (the brand guard reads them; the generality guard does not).

    With `prose`, one `title` is still not yielded: the one that is only its own property's name
    re-cased (`"mistudio_sae_id": {"title": "Mistudio Sae Id"}`). Nobody wrote it; it changes when the
    name does, and the name itself was yielded one level up, where the guard judges it."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield 0, str(key)
            if str(key) in PROSE_KEYS and isinstance(item, str):
                if not prose:
                    continue
                if str(key) == "title" and item == derived_title(path.rpartition(".")[2]):
                    continue
            yield from data_items(item, f"{path}.{key}", prose)
    elif isinstance(value, list):
        for item in value:
            yield from data_items(item, path, prose)
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        yield 0, str(value)


def env_items(source: str) -> Iterator[tuple[int, str]]:
    for number, raw in enumerate(source.splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        yield number, text.split("#", 1)[0] if " #" in text else text


def yaml_items(source: str, prose: bool = False) -> Iterator[tuple[int, str]]:
    import yaml

    for document in yaml.safe_load_all(source):
        yield from data_items(document, prose=prose)


def json_items(source: str, prose: bool = False) -> Iterator[tuple[int, str]]:
    yield from data_items(json.loads(source), prose=prose)


def markdown_items(source: str, prose: bool = False) -> Iterator[tuple[int, str]]:
    """Every text line of a Markdown file (040 FR-040.15): code spans and fenced code blocks
    included, front matter (a leading ``---`` block) by parsed YAML value."""
    lines = source.splitlines()
    start = 0
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is not None:
            import yaml

            front = yaml.safe_load("\n".join(lines[1:end])) or {}
            for _line, text in data_items(front, prose=prose):
                yield 1, text
            start = end + 1
    for number in range(start, len(lines)):
        if lines[number].strip():
            yield number + 1, lines[number]


def filename_items(rel: str) -> Iterator[tuple[int, str]]:
    """The repository-relative path itself, as one item (040 FR-040.15: a file name is a surface)."""
    yield 0, rel


# ── trees ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Tree:
    """One scanned root: what is listed, how each file is parsed."""

    name: str
    roots: tuple[str, ...]
    suffixes: tuple[str, ...]
    exclude: tuple[str, ...] = ()

    def wants(self, rel: str) -> bool:
        if not rel.endswith(self.suffixes):
            return False
        if "__pycache__" in rel or "node_modules" in rel:
            return False
        return not any(re.search(pattern, rel) for pattern in self.exclude)


def _walk(tree: Tree, root_dir: Path) -> list[str]:
    found = []
    for root in tree.roots:
        base = root_dir / root
        if not base.exists():
            continue
        if root == ".":
            entries = [p for p in root_dir.iterdir() if p.is_file()]
        else:
            entries = [p for p in base.rglob("*") if p.is_file()]
        for path in entries:
            rel = path.relative_to(root_dir).as_posix()
            if tree.wants(rel):
                found.append(rel)
    return sorted(found)


def _git_listed(tree: Tree, root_dir: Path, pathspec: tuple[str, ...] = ()) -> list[str] | None:
    """`git ls-files` over the tree, or None when there is no repository (the image build).

    ``pathspec`` adds git pathspecs (040 CF-20: the brand guard excludes the legacy-identifier
    export with ``:(exclude,glob)**/docs/legacy-identifiers*.md``)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", *tree.roots, *pathspec], cwd=root_dir, capture_output=True, text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    listed = [rel for rel in out.stdout.splitlines() if tree.wants(rel)]
    # A tracked file deleted in the working tree is not a file to parse.
    return [rel for rel in listed if (root_dir / rel).is_file()]


def items_of(rel: str, source: str, prose: bool = False) -> Iterator[tuple[int, str]]:
    if rel.endswith(".py"):
        return python_items(source)
    if rel.endswith((".ts", ".tsx")):
        return typescript_items(source)
    if rel.endswith((".yaml", ".yml")):
        return yaml_items(source, prose)
    if rel.endswith(".json"):
        return json_items(source, prose)
    if rel.endswith(".env.example"):
        return env_items(source)
    if rel.endswith((".md", ".mdx")):
        return markdown_items(source, prose)
    raise AssertionError(f"no extractor for {rel}")
