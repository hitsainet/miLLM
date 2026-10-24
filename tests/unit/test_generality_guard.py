"""No executable or deployment file in miLLM names a specific behaviour (FR-38.1, D20, D38.3).

miLLM serves whatever behaviours miStudio's catalogue defines. A behaviour named in code is a
capability hard-wired for one customer's pilot, and the next behaviour then needs a code change —
the opposite of "configuration, never code". This guard is the only thing that keeps that true.

**What it scans** (BR-041's scope):

* ``millm/`` — Python by abstract syntax tree: identifiers, attributes, definition and argument
  names, dict keys and every string literal **except** docstrings and other bare string
  statements (prose). Migrations and configuration defaults live here too.
* ``admin-ui/src/`` without tests — a minimal TypeScript lexer: identifiers, strings, template
  literals and JSX text; comments excluded.
* ``k8s/``, ``docker-compose*.yml``, ``docs/schemas/*.json`` and data files under ``millm/`` —
  YAML and JSON by PARSED value, keys included. A JSON-Schema ``description``/``title``/
  ``$comment`` is prose and skipped; ``examples``, ``default``, ``const`` and ``enum`` are scanned.
* ``.env.example`` — keys and default values; comments excluded.

**Which names.** Every ``key``, ``name`` and ``words`` entry of every file in
``tests/fixtures/behaviour_packs/`` — read, never listed here, so a new pack extends the guard by
existing. Tests, fixtures and prose are not scanned.

**How it matches.** Text is split into tokens on non-alphanumerics and case boundaries, lower
cased, and a name matches when its token sequence occurs contiguously and whole: ``card_pan``
matches ``cardPan`` and ``CARD-PAN``, never ``discard_panel``.

**Fail-open guard.** Per tree, the files parsed must be at least the files ``git ls-files`` lists
(a filesystem walk when there is no ``.git``, as in the image build, where ``0xcc/`` is also
stripped), and at least one. A self-test plants every name in every token category of synthetic
sources; deleting the plants turns it red.

**Extension point (miLLM 039).** ``EXEMPTIONS`` is the one place an exemption may be added, as a
``(relative path, name)`` pair with its own control. It is empty in 038.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKS = ROOT / "tests" / "fixtures" / "behaviour_packs"

#: (relative path, name) pairs allowed to match. Empty: 039's two exemptions are below, each
#: derived from an authority rather than listed here.
EXEMPTIONS: frozenset[tuple[str, str]] = frozenset()

#: 039 S2 (TD39-27): inside this package ONLY, a string literal EXACTLY equal to a registered
#: validator id — read from the LIVE registry, never typed here — is data (the producer's id),
#: not a behaviour named in code. A key, a word or an identifier still fails there.
VALIDATOR_PACKAGE = "millm/rules/validators/"

#: 039 S2: files whose bytes are another repo's, pinned byte-for-byte by a test, and read by
#: tests only (no `millm` module opens them). The producer's conformance vectors name its pilot's
#: formats by design; editing them is impossible (the pin), so the guard cannot ask for it.
PINNED_FOREIGN_DATA = {
    "docs/schemas/rule-signal-cases-v1.json":
        "tests/unit/rules/test_vendored_rule_contracts.py (miDataworks 016's vectors)",
}


def registered_validator_ids() -> frozenset[str]:
    """The live validator registry's ids (the exemption's authority)."""
    from millm.rules.validators import registered_ids

    return registered_ids()

#: JSON-Schema keys whose values are prose.
PROSE_KEYS = frozenset({"description", "title", "$comment", "markdownDescription"})


# ── names ────────────────────────────────────────────────────────────────────────


def _pack_strings(value: Any, key: str | None = None) -> Iterator[str]:
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _pack_strings(v, k)
    elif isinstance(value, list):
        for item in value:
            yield from _pack_strings(item, key)
    elif isinstance(value, str) and key in ("key", "name", "words"):
        yield value


def pack_names() -> dict[tuple[str, ...], str]:
    """`{token sequence: the name as written}` from every behaviour pack."""
    names: dict[tuple[str, ...], str] = {}
    files = sorted(PACKS.glob("*.json"))
    assert files, f"no behaviour packs in {PACKS} — the guard would look for nothing"
    for path in files:
        for name in _pack_strings(json.loads(path.read_text())):
            tokens = tuple(tokenize(name))
            if tokens:
                names.setdefault(tokens, name)
    return names


# ── tokens ───────────────────────────────────────────────────────────────────────

_CASE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Split on non-alphanumerics, then on case and letter/digit boundaries; lower case."""
    out: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", text):
        out.extend(part.lower() for part in _CASE.findall(chunk))
    return out


def find(tokens: list[str], names: dict[tuple[str, ...], str]) -> list[str]:
    hits = []
    for seq, name in names.items():
        n = len(seq)
        if any(tuple(tokens[i:i + n]) == seq for i in range(len(tokens) - n + 1)):
            hits.append(name)
    return hits


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


def data_items(value: Any, path: str = "") -> Iterator[tuple[int, str]]:
    """Parsed YAML/JSON: every key and every scalar value, prose keys skipped."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield 0, str(key)
            if str(key) in PROSE_KEYS and isinstance(item, str):
                continue
            yield from data_items(item, f"{path}.{key}")
    elif isinstance(value, list):
        for item in value:
            yield from data_items(item, path)
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        yield 0, str(value)


def env_items(source: str) -> Iterator[tuple[int, str]]:
    for number, raw in enumerate(source.splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        yield number, text.split("#", 1)[0] if " #" in text else text


def yaml_items(source: str) -> Iterator[tuple[int, str]]:
    import yaml

    for document in yaml.safe_load_all(source):
        yield from data_items(document)


def json_items(source: str) -> Iterator[tuple[int, str]]:
    yield from data_items(json.loads(source))


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


TREES = (
    Tree("python", ("millm",), (".py",)),
    Tree("typescript", ("admin-ui/src",), (".ts", ".tsx"),
         exclude=(r"__tests__/", r"\.test\.tsx?$", r"\.spec\.tsx?$", r"(^|/)test/",
                  r"setupTests\.ts$", r"\.d\.ts$")),
    Tree("manifests", ("k8s",), (".yaml", ".yml", ".json")),
    Tree("schemas", ("docs/schemas",), (".json",)),
    Tree("app-data", ("millm",), (".json", ".yaml", ".yml")),
    Tree("compose", (".",), (".yml", ".yaml"), exclude=(r"/", r"^(?!docker-compose)")),
    Tree("env", (".",), (".env.example",), exclude=(r"/",)),
)


def _walk(tree: Tree) -> list[str]:
    found = []
    for root in tree.roots:
        base = ROOT / root
        if not base.exists():
            continue
        if root == ".":
            entries = [p for p in ROOT.iterdir() if p.is_file()]
        else:
            entries = [p for p in base.rglob("*") if p.is_file()]
        for path in entries:
            rel = path.relative_to(ROOT).as_posix()
            if tree.wants(rel):
                found.append(rel)
    return sorted(found)


def _git_listed(tree: Tree) -> list[str] | None:
    """`git ls-files` over the tree, or None when there is no repository (the image build)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", *tree.roots], cwd=ROOT, capture_output=True, text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    listed = [rel for rel in out.stdout.splitlines() if tree.wants(rel)]
    # A tracked file deleted in the working tree is not a file to parse.
    return [rel for rel in listed if (ROOT / rel).is_file()]


def items_of(rel: str, source: str) -> Iterator[tuple[int, str]]:
    if rel.endswith(".py"):
        return python_items(source)
    if rel.endswith((".ts", ".tsx")):
        return typescript_items(source)
    if rel.endswith((".yaml", ".yml")):
        return yaml_items(source)
    if rel.endswith(".json"):
        return json_items(source)
    if rel.endswith(".env.example"):
        return env_items(source)
    raise AssertionError(f"no extractor for {rel}")


def scan(tree: Tree, names: dict[tuple[str, ...], str],
         exemptions: frozenset[tuple[str, str]] = EXEMPTIONS) -> tuple[int, list[str]]:
    """(files parsed, hits) for one tree. A file that does not parse FAILS, it is not skipped."""
    files = _walk(tree)
    hits: list[str] = []
    registered = registered_validator_ids() if tree.name == "python" else frozenset()
    for rel in files:
        source = (ROOT / rel).read_text(encoding="utf-8")
        if rel in PINNED_FOREIGN_DATA:
            continue
        in_package = rel.startswith(VALIDATOR_PACKAGE)
        for line, text in items_of(rel, source):
            if in_package and text in registered:
                continue
            for name in find(tokenize(text), names):
                if (rel, name) not in exemptions:
                    hits.append(f"{rel}:{line}: {name!r} in {text[:80]!r}")
    return len(files), hits


# ── the guard ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tree", TREES, ids=[t.name for t in TREES])
def test_no_behaviour_is_named_in_executable_or_deployment_files(tree):
    names = pack_names()
    parsed, hits = scan(tree, names)
    listed = _git_listed(tree)
    floor = len(listed) if listed is not None else len(_walk(tree))
    assert parsed >= floor, (
        f"{tree.name}: parsed {parsed} files but the index lists {floor} — the guard is "
        f"failing open on files it never read"
    )
    if tree.name not in ("compose", "env", "app-data"):
        assert parsed > 0, f"{tree.name}: parsed nothing — a broken lister must not pass"
    assert not hits, (
        "A specific behaviour is named in an executable or deployment file. miLLM serves the "
        "behaviours miStudio's catalogue defines; naming one here hard-wires it (FR-38.1). "
        "Move it to data (a probe definition, a pack, a setting) — do not exempt it:\n  "
        + "\n  ".join(hits)
    )


def test_the_packs_yield_every_name_category():
    names = set(pack_names())
    seeded = json.loads((PACKS / "seeded.json").read_text())["behaviours"]
    pilot = json.loads((PACKS / "pilot.json").read_text())
    for entry in seeded + pilot["behaviours"]:
        assert tuple(tokenize(entry["key"])) in names and tuple(tokenize(entry["name"])) in names
    assert {tuple(tokenize(w)) for w in pilot["words"]} <= names


def test_tokens_are_whole_and_case_insensitive():
    names = {tuple(tokenize("card_pan")): "card_pan"}
    for text in ("cardPan", "CARD-PAN", "card_pan", "the card pan field", "CardPAN"):
        assert find(tokenize(text), names), text
    for text in ("discard_panel", "cardpanel", "card_panel", "scard_pan"):
        assert not find(tokenize(text), names), text


# ── self-test: every token category, planted ──────────────────────────────────────

PLANT = "zq plantname"  # a name no real file carries


def _planted_python() -> str:
    return (
        '"""Docstring prose: zq plantname stays green."""\n'
        "# a comment: zq_plantname stays green\n"
        "zq_plantname = 1\n"                                      # identifier
        "x.zqPlantname\n"                                         # attribute
        "def zq_plantname_fn(zq_plantname_arg=None):\n"           # function and argument name
        '    "inner docstring zq plantname: green"\n'
        "    return {'zq-plantname': f'{x} zq plantname'}\n"      # dict key, f-string part
        "call(zq_plantname=1)\n"                                  # keyword
        "class ZqPlantname:\n    pass\n"                          # class name
        "if behaviour == 'ZQ_PLANTNAME':\n    pass\n"             # string literal compare
    )


def _planted_typescript() -> str:
    return (
        "// zq plantname in a comment stays green\n"
        "/* and zq_plantname in a block comment */\n"
        "const zqPlantname = 1;\n"                                # identifier
        "const s = 'zq-plantname';\n"                             # single-quoted string
        'const d = "zq plantname";\n'                             # double-quoted string
        "const t = `prefix ${zq_plantname} zq plantname`;\n"      # template + substitution
        "const el = <p>zq plantname as JSX text</p>;\n"          # JSX text
    )


def _planted_yaml() -> str:
    return "metadata:\n  zq-plantname: 1\nspec:\n  value: zq plantname\n"


def _planted_json() -> str:
    return json.dumps({
        "description": "zq plantname in prose stays green",
        "zq_plantname": {"examples": ["zq plantname"], "default": "ZqPlantname"},
    })


#: (extractor, source, expected executable plants found). ⚠ DELETING THE PLANTS turns this red:
#: the expected counts are of real hits, so a self-test that finds nothing cannot pass.
SELF_TEST = (
    (python_items, _planted_python, 9),
    (typescript_items, _planted_typescript, 6),
    (yaml_items, _planted_yaml, 2),
    (json_items, _planted_json, 3),
)


@pytest.mark.parametrize("extractor,source,expected", SELF_TEST,
                         ids=["python", "typescript", "yaml", "json"])
def test_the_self_test_plants_are_found_and_prose_is_not(extractor, source, expected):
    names = {tuple(tokenize(PLANT)): PLANT}
    hits = [text for _line, text in extractor(source()) if find(tokenize(text), names)]
    assert len(hits) == expected, f"expected {expected} planted hits, got {len(hits)}: {hits}"
    for text in hits:
        assert "prose" not in text and "comment" not in text and "docstring" not in text.lower()


def test_the_lister_falls_back_to_a_walk_without_git(monkeypatch):
    """The image build has no `.git`: the floor must come from the filesystem, never be 0."""
    def no_git(*_a, **_k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert _git_listed(TREES[0]) is None
    assert len(_walk(TREES[0])) > 0


def test_exemptions_are_empty_in_038():
    assert EXEMPTIONS == frozenset(), "an exemption needs its own decision and control (039)"


# ── 039 S2: the two exemptions, each proven to read its authority ───────────────────


def test_the_validator_exemption_reads_the_live_registry(monkeypatch):
    """Drop one id from the live registry and its literal inside the package is a hit again: the
    exemption is the registry, not a list (TD39-27 control)."""
    import millm.rules.validators as validators

    python = next(t for t in TREES if t.name == "python")
    names = pack_names()
    assert not scan(python, names)[1]
    kept = {k: v for k, v in validators.VALIDATORS.items() if k[0] != "iban"}
    monkeypatch.setattr(validators, "VALIDATORS", kept)
    hits = scan(python, names)[1]
    assert hits and all(h.startswith(VALIDATOR_PACKAGE) for h in hits)


def test_a_behaviour_key_planted_in_the_validator_package_is_still_found(tmp_path, monkeypatch):
    """The exemption is exact-literal only: a key, an identifier or a longer string naming a
    behaviour inside the package still fails."""
    import shutil

    root = tmp_path / "repo"
    shutil.copytree(ROOT / "millm" / "rules", root / "millm" / "rules")
    plant = root / "millm" / "rules" / "validators" / "zz_plant.py"
    key = json.loads((PACKS / "pilot.json").read_text())["behaviours"][0]["key"]
    plant.write_text(f"TABLE = {{{key!r}: 1}}\nlabel = 'the {key} format'\n")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", root)
    python = Tree("python", ("millm",), (".py",))
    hits = scan(python, pack_names())[1]
    assert any("zz_plant.py" in h for h in hits), hits


def test_pinned_foreign_data_is_pinned():
    """Every file the guard skips as foreign data is byte-pinned by the test it names."""
    for rel, pin in PINNED_FOREIGN_DATA.items():
        pin_file = ROOT / pin.split(" ")[0]
        assert pin_file.is_file(), pin
        assert Path(rel).name in pin_file.read_text(), f"{pin} does not pin {rel}"
