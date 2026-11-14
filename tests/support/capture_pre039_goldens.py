"""Pre-039 goldens (039 FTASKS 0.7): what every probe surface said before 039, captured from `main`
at `779efa11` with NO 039 code present, through the 038 harness (`capture_pre038_goldens`).

Two scenarios, as the task names them: two probes of two behaviours armed (`armed`, the 038
harness's own two-probe scenario) and nothing armed (`unarmed`). They pin:

* nothing armed → every surface byte-identical after 039 (FR-39.11);
* the score response an additive superset (FR-39.13): every pre-039 key present and unchanged;
* with probes armed, every pre-039 key present with its value, except the changes 039 documents
  (`test_pre039_goldens.py` lists each one with its reason).

Captured by running THIS module inside a `git archive` of `main` (the harness it imports is the one
in that tree), then copying `tests/fixtures/goldens/pre039/` here. Re-running it on a 039 tree would
record 039's own output, which is not a golden of anything.

    venv/bin/python -m tests.support.capture_pre039_goldens
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "goldens" / "pre039"
SCENARIOS = ("armed", "unarmed")


def golden_path(scenario: str) -> Path:
    return GOLDEN_DIR / f"{scenario}.json"


async def capture_all() -> dict[str, dict]:
    from tests.support import capture_pre038_goldens as cap

    return {name: await cap.capture(next(s for s in cap.SCENARIOS if s.name == name))
            for name in SCENARIOS}


def main() -> None:
    import logging

    logging.disable(logging.CRITICAL)
    captured = asyncio.run(capture_all())
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, surfaces in captured.items():
        golden_path(name).write_text(json.dumps(surfaces, indent=2, sort_keys=True) + "\n")
        print(f"{name}: {', '.join(sorted(surfaces))}")


if __name__ == "__main__":
    main()
