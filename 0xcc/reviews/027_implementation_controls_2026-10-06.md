# Feature 27 — Implementation Controls Record (2026-10-06)

Branch `feat/027-probe-scoring`, cut from `main` at `08c1c53`. Worktree `~/app/miLLM-027`.
`millm.__file__` verified as `/home/x-sean/app/miLLM-027/millm/__init__.py` (venv `~/app/miLLM/venv`).

**Baseline** (before any change, `pytest tests/unit`): **4313 passed / 3 skipped / 0 failed** (174.6 s).

Procedure for every control: back up the file, make ONE change, confirm it landed (grep), run the
named tests and require a red, restore from the backup, verify the restore by `sha256sum` against
the pre-mutation hash AND by re-grepping the mutated line. A survivor is a test finding: write the
test, re-run the control as a negative control.

## Task 1 — expected red (task 1.6)

The discovery guard (`tests/unit/services/test_probe_paths_discovered.py`) run on the code at
`08c1c53`: discovery finds **10 generation sites** and **3 entry points**; the behavioural scenario
fails on exactly the six sites FR-27.8 names, each with *"generation reached with no probe context
open"*: `_generate_batch_chunk` (batched chat), `_cbm_chat_completion`, `_cbm_text_completion`,
`_llamacpp_chat_completion`, `_llamacpp_stream_chat_completion`, `_llamacpp_text_completion`.
The four already-wired sites pass. The six are committed as `xfail(strict=True)` so the red is
recorded mechanically and each fix must remove its entry.

