"""The Batch API (Feature 26): durable JSONL batch jobs run through the one admission path.

Deliberately imports nothing at package level. `inference_service` reads `state.BATCH_ROW`, and a
package `__init__` that imported the runner would import `inference_service` back — the cycle the
FTID (§2) puts `BATCH_ROW` in `state.py` to avoid. Import the submodules directly:
`from millm.services.batch.runner import get_batch_runner`.
"""
