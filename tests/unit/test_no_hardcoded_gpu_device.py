"""No GPU is hard-coded for any function outside the placement module.

Operator directive (2026-09-13): nothing may assume index 0 or a single card,
and adding a card must need no code change. Every one of these meant GPU 0 on
the two-GPU node, whatever card that happened to be:

    device="cuda"            .to("cuda")           torch.device("cuda")
    device_map="auto"        mem_get_info()        mem_get_info(0)
    memory_allocated()       synchronize()         max_memory={0: ...}
    main_gpu=0               def f(device=0)       CUDA_VISIBLE_DEVICES=...

`millm/ml/gpu_placement.py` is the one place allowed to spell a device, because
it is where the choice is made. Everything else asks it.

The scan reads the AST, never the source text: a substring search matches the
comments that DESCRIBE these hazards (this repo's comments name them often) and
passes or fails for the wrong reason.

MUTATION CONTROLS (each must turn this file red):
  * make `_cuda_literal` return False for "cuda"   -> the snippet tests fail
  * drop the no-argument check for mem_get_info    -> the snippet tests fail
  * put `device_map="auto"` back in the draft load -> the repo scan fails
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

import millm

MILLM_ROOT = Path(millm.__file__).resolve().parent

#: Where device choices are made. Exempt by design; it is tested directly.
PLACEMENT_MODULE = "millm/ml/gpu_placement.py"

#: (module, function qualname, rule) -> why this site is correct.
#: Keep this as short as possible; every entry must still match a real finding
#: (a stale entry fails `test_every_allowlist_entry_is_still_needed`).
ALLOWLIST: dict[tuple[str, str, str], str] = {}

_CUDA_RE = re.compile(r"^cuda(:\d+)?$")

#: torch.cuda functions that act on the CURRENT device (GPU 0 unless changed)
#: when no device is passed.
_IMPLICIT_DEVICE_FUNCS = {
    "mem_get_info",
    "memory_allocated",
    "memory_reserved",
    "max_memory_allocated",
    "max_memory_reserved",
    "reset_peak_memory_stats",
    "synchronize",
    "memory_stats",
    "get_device_name",
    "get_device_properties",
    "current_device",
}
#: ...and those that pin a device when given a literal index.
_DEVICE_INDEX_FUNCS = _IMPLICIT_DEVICE_FUNCS | {"set_device", "device"}

_DEVICE_KEYWORDS = {"device", "device_map", "map_location", "main_device", "input_device"}
_DEVICE_PARAM_NAMES = {"device", "gpu", "gpu_index", "main_gpu", "device_index"}


@dataclass(frozen=True)
class Finding:
    module: str
    function: str
    rule: str
    line: int

    def __str__(self) -> str:
        return f"{self.module}:{self.line} in {self.function} [{self.rule}]"


def _cuda_literal(node: ast.AST | None) -> bool:
    """A string constant naming a CUDA device, or an expression that yields one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return bool(_CUDA_RE.match(node.value))
    if isinstance(node, ast.IfExp):
        return _cuda_literal(node.body) or _cuda_literal(node.orelse)
    if isinstance(node, ast.BoolOp):
        return any(_cuda_literal(value) for value in node.values)
    if isinstance(node, ast.Dict):
        return any(_cuda_literal(value) for value in node.values)
    return False


def _int_zero(node: ast.AST | None) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and node.value == 0
    )


def _auto_literal(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value == "auto"


def _identifier(node: ast.AST) -> str:
    """The name an assignment target or call is known by, lowercased."""
    if isinstance(node, ast.Name):
        return node.id.lower()
    if isinstance(node, ast.Attribute):
        return node.attr.lower()
    if isinstance(node, ast.Subscript):
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            return key.value.lower()
    return ""


def _is_cuda_namespace(node: ast.AST) -> bool:
    """`torch.cuda`, `_torch.cuda`, or a bare `cuda` name."""
    return (isinstance(node, ast.Attribute) and node.attr == "cuda") or (
        isinstance(node, ast.Name) and node.id == "cuda"
    )


def _has_int_card_key(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Dict) and any(
        isinstance(key, ast.Constant)
        and isinstance(key.value, int)
        and not isinstance(key.value, bool)
        for key in node.keys
    )


class _Scanner(ast.NodeVisitor):
    def __init__(self, module: str):
        self.module = module
        self.stack: list[str] = []
        self.findings: list[Finding] = []

    # -- bookkeeping ---------------------------------------------------------

    def _function(self) -> str:
        return ".".join(self.stack) or "<module>"

    def _flag(self, node: ast.AST, rule: str) -> None:
        self.findings.append(
            Finding(self.module, self._function(), rule, getattr(node, "lineno", 0))
        )

    def _enter(self, node) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._enter(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_defaults(node)
        self._enter(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_defaults(node)
        self._enter(node)

    # -- rules ---------------------------------------------------------------

    def _check_defaults(self, node) -> None:
        args = node.args
        positional = args.posonlyargs + args.args
        pairs = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
        pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
        for arg, default in pairs:
            name = arg.arg.lower()
            if name in _DEVICE_PARAM_NAMES or name.endswith("_device"):
                if _int_zero(default) or _cuda_literal(default):
                    self._flag(default, "device-default")

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        attr = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else ""
        )

        for keyword in node.keywords:
            name = (keyword.arg or "").lower()
            if name in _DEVICE_KEYWORDS and _cuda_literal(keyword.value):
                self._flag(keyword.value, "cuda-string-device")
            if name == "device_map" and _auto_literal(keyword.value):
                self._flag(keyword.value, "device-map-auto")
            if name == "main_gpu" and _int_zero(keyword.value):
                self._flag(keyword.value, "device-zero")
            if name == "max_memory" and _has_int_card_key(keyword.value):
                self._flag(keyword.value, "max-memory-literal-card")
            if name == "env" and isinstance(keyword.value, ast.Dict):
                self._check_env_dict(keyword.value)

        # .to("cuda"), torch.device("cuda"), x.to_device("cuda")
        if attr in ("to", "device", "to_device") and node.args and _cuda_literal(node.args[0]):
            self._flag(node.args[0], "cuda-string-device")

        # model.cuda() — the current device, i.e. GPU 0
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "cuda"
            and not node.args
            and not node.keywords
        ):
            self._flag(node, "bare-dot-cuda")

        if isinstance(func, ast.Attribute) and _is_cuda_namespace(func.value):
            has_device_kw = any(k.arg == "device" for k in node.keywords)
            if attr in _IMPLICIT_DEVICE_FUNCS and not node.args and not has_device_kw:
                self._flag(node, "implicit-device-query")
            if attr in _DEVICE_INDEX_FUNCS and (
                (node.args and _int_zero(node.args[0]))
                or any(k.arg == "device" and _int_zero(k.value) for k in node.keywords)
            ):
                self._flag(node, "device-zero")

        if attr in ("setdefault", "putenv") and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value == "CUDA_VISIBLE_DEVICES":
                self._flag(node, "cuda-visible-devices")

        self.generic_visit(node)

    def _check_env_dict(self, node: ast.Dict) -> None:
        for key in node.keys:
            if isinstance(key, ast.Constant) and key.value == "CUDA_VISIBLE_DEVICES":
                self._flag(key, "cuda-visible-devices")

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            name = key.value.lower()
            if key.value == "CUDA_VISIBLE_DEVICES":
                self._flag(key, "cuda-visible-devices")
            if name in _DEVICE_KEYWORDS and _cuda_literal(value):
                self._flag(value, "cuda-string-device")
            if name == "device_map" and _auto_literal(value):
                self._flag(value, "device-map-auto")
            if name == "main_gpu" and _int_zero(value):
                self._flag(value, "device-zero")
            if name == "max_memory" and _has_int_card_key(value):
                self._flag(value, "max-memory-literal-card")
        self.generic_visit(node)

    def _check_assignment(self, targets: list[ast.AST], value: ast.AST | None) -> None:
        for target in targets:
            name = _identifier(target)
            if isinstance(target, ast.Subscript) and name == "cuda_visible_devices":
                self._flag(target, "cuda-visible-devices")
            if "device" in name and _cuda_literal(value):
                self._flag(value, "cuda-string-device")
            if name == "device_map" and _auto_literal(value):
                self._flag(value, "device-map-auto")
            if "max_memory" in name and _has_int_card_key(value):
                self._flag(value, "max-memory-literal-card")

    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_assignment(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_assignment([node.target], node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_assignment([node.target], node.value)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if self.stack and "device" in self.stack[-1].lower() and _cuda_literal(node.value):
            self._flag(node.value, "cuda-string-device")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        operands = [node.left, *node.comparators]
        if any(_cuda_literal(op) for op in operands) and any(
            "device" in _identifier(op) for op in operands
        ):
            self._flag(node, "cuda-string-compare")
        self.generic_visit(node)


def scan_source(source: str, module: str = "<snippet>") -> list[Finding]:
    scanner = _Scanner(module)
    scanner.visit(ast.parse(source))
    return scanner.findings


def scan_repository() -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(MILLM_ROOT.rglob("*.py")):
        module = path.relative_to(MILLM_ROOT.parent).as_posix()
        if module == PLACEMENT_MODULE:
            continue
        findings.extend(scan_source(path.read_text(encoding="utf-8"), module))
    return findings


# ---------------------------------------------------------------------------
# The detector detects. Without these, a scanner that finds nothing passes the
# repository scan for the wrong reason.
# ---------------------------------------------------------------------------

POSITIVE = [
    ('x.to("cuda")', "cuda-string-device"),
    ('x.to("cuda:0")', "cuda-string-device"),
    ('torch.device("cuda")', "cuda-string-device"),
    ('load(p, device="cuda")', "cuda-string-device"),
    ('M.from_pretrained(p, device_map={"": "cuda"})', "cuda-string-device"),
    ('self._device = "cuda" if torch.cuda.is_available() else "cpu"', "cuda-string-device"),
    ('kwargs = {"device_map": "cuda:0"}', "cuda-string-device"),
    ('def pick_device():\n    return "cuda"', "cuda-string-device"),
    ('if device == "cuda":\n    pass', "cuda-string-compare"),
    ('def f(device="cuda"):\n    pass', "device-default"),
    ('def f(device: int = 0):\n    pass', "device-default"),
    ('def f(*, main_gpu=0):\n    pass', "device-default"),
    ('M.from_pretrained(p, device_map="auto")', "device-map-auto"),
    ('kwargs = {"device_map": "auto"}', "device-map-auto"),
    ('torch.cuda.mem_get_info()', "implicit-device-query"),
    ('_torch.cuda.memory_allocated()', "implicit-device-query"),
    ('torch.cuda.memory_reserved()', "implicit-device-query"),
    ('torch.cuda.synchronize()', "implicit-device-query"),
    ('torch.cuda.reset_peak_memory_stats()', "implicit-device-query"),
    ('torch.cuda.mem_get_info(0)', "device-zero"),
    ('torch.cuda.set_device(0)', "device-zero"),
    ('torch.cuda.memory_allocated(device=0)', "device-zero"),
    ('Llama(model_path=p, main_gpu=0)', "device-zero"),
    ('load_kwargs["max_memory"] = {0: "20GiB", "cpu": "8GiB"}', "max-memory-literal-card"),
    ('M.from_pretrained(p, max_memory={0: "20GiB"})', "max-memory-literal-card"),
    ('model.cuda()', "bare-dot-cuda"),
    ('os.environ["CUDA_VISIBLE_DEVICES"] = "1"', "cuda-visible-devices"),
    ('os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")', "cuda-visible-devices"),
    ('subprocess.run(cmd, env={"CUDA_VISIBLE_DEVICES": "0"})', "cuda-visible-devices"),
]

NEGATIVE = [
    '# model.to("cuda") and device_map="auto" are described in this comment',
    'def f():\n    """Never device_map="auto" or mem_get_info() here."""',
    "torch.cuda.is_available()",
    "torch.cuda.device_count()",
    "torch.cuda.empty_cache()",
    "torch.cuda.mem_get_info(index)",
    "torch.cuda.synchronize(index)",
    'x.to("cpu")',
    'os.environ.get("CUDA_VISIBLE_DEVICES")',
    's.startswith("cuda")',
    'label = f"cuda:{index}"',
    'M.from_pretrained(p, device_map=placement.transformers_device_map(bitsandbytes=False))',
    'def f(device):\n    pass',
    'kind = "cuda"',
]


@pytest.mark.parametrize("source, rule", POSITIVE)
def test_the_scanner_catches(source, rule):
    assert rule in {finding.rule for finding in scan_source(source)}, (
        f"the scanner does not flag {source!r} as {rule}"
    )


@pytest.mark.parametrize("source", NEGATIVE)
def test_the_scanner_does_not_flag(source):
    assert scan_source(source) == []


def test_the_scan_covers_the_package():
    """A glob that matched nothing would pass the repository test vacuously."""
    modules = {p.relative_to(MILLM_ROOT.parent).as_posix() for p in MILLM_ROOT.rglob("*.py")}
    assert "millm/ml/model_loader.py" in modules
    assert "millm/services/inference_service.py" in modules
    assert PLACEMENT_MODULE in modules, "the exempt module moved; update PLACEMENT_MODULE"


# ---------------------------------------------------------------------------
# The repository.
# ---------------------------------------------------------------------------


def test_no_hardcoded_gpu_choice_outside_the_placement_module():
    findings = [
        finding
        for finding in scan_repository()
        if (finding.module, finding.function, finding.rule) not in ALLOWLIST
    ]
    assert not findings, (
        "hard-coded GPU choice(s) outside millm/ml/gpu_placement.py — each means "
        "GPU 0 on a multi-GPU node. Ask the placement module instead:\n  "
        + "\n  ".join(str(f) for f in findings)
    )


def test_every_allowlist_entry_is_still_needed():
    present = {(f.module, f.function, f.rule) for f in scan_repository()}
    stale = [entry for entry in ALLOWLIST if entry not in present]
    assert not stale, f"allowlist entries that no longer match anything: {stale}"
