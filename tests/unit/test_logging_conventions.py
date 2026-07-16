"""
Convention sweep: modules using the STDLIB logger must not make
structlog-style keyword-argument logging calls.

Three production incidents from this one mismatch (2026-07-16): the generic
exception handler crashed on every unhandled exception, SAE attach 500'd at
hook install, and transposed-weight SAE loads would have crashed in the
loader. The stdlib Logger raises TypeError for unknown kwargs, so these bugs
only surface when the log line actually executes — this test finds them all
statically instead.
"""

import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "millm"

CALL = re.compile(
    r"logger\.(?:info|warning|error|debug|exception|critical)\((.*?)\)",
    re.S,
)
KWARG = re.compile(r"^\s*[a-z_][a-z0-9_]*\s*=", re.M)
STDLIB_SAFE_KWARGS = {"exc_info", "stacklevel", "extra", "msg"}


def test_no_structlog_kwargs_on_stdlib_loggers():
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        text = path.read_text()
        # Module-level logger binding determines the API in use.
        if not re.search(r"^logger = logging\.getLogger", text, re.M):
            continue
        for match in CALL.finditer(text):
            body = match.group(1)
            kwargs = [k.strip().rstrip("=").strip()
                      for k in KWARG.findall(body)]
            kwargs = [k for k in kwargs if k not in STDLIB_SAFE_KWARGS]
            if kwargs:
                line = text[: match.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(PACKAGE_ROOT.parent)}"
                                 f":{line} kwargs={kwargs}")
    assert not offenders, (
        "structlog-style kwargs on stdlib loggers (TypeError at runtime):\n"
        + "\n".join(offenders)
        + "\nUse %-style args, or switch the module to "
          "millm.core.logging.get_logger."
    )
