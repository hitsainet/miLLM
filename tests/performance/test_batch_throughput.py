"""Feature 26 acceptance 7 and FR-26.5.8 — HARDWARE ONLY (operator session on the GPU node).

Skipped unless `MILLM_BATCH_HW_URL` names a live miLLM serving JEV-9B-decision in bfloat16 and
`MILLM_BATCH_HW_FILE` names a 10,000-line scoring JSONL. Never runs in CI or in a developer tree:
a throughput figure measured anywhere else is not acceptance 7.

Records, for the same file: unpacked rows/second (must be >= 19), packed rows/second, the maximum
absolute logprob difference between packed and single, and the top-token agreement rate. If ANY top
token differs, T-63 flips BATCH_PACK_DEFAULT to false (config, .env.example, k8s) and the result is
published in the API reference and docs/mcp-contract.md §4f.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

import pytest

URL = os.environ.get("MILLM_BATCH_HW_URL")
FILE = os.environ.get("MILLM_BATCH_HW_FILE")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not (URL and FILE), reason="hardware acceptance: set MILLM_BATCH_HW_URL "
                       "and MILLM_BATCH_HW_FILE on the operator session (026 FTASKS 9.3/9.4)"),
]


def _call(method: str, path: str, body: bytes | None = None, headers: dict | None = None) -> dict:
    req = urllib.request.Request(URL + path, data=body, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.loads(response.read())


def _run(file_id: str, pack: bool) -> tuple[float, list[dict]]:
    batch = _call("POST", "/v1/batches", json.dumps({
        "input_file_id": file_id, "endpoint": "/v1/completions", "completion_window": "24h",
        "pack": pack}).encode(), {"Content-Type": "application/json"})
    started = None
    while True:
        batch = _call("GET", f"/v1/batches/{batch['id']}")
        if batch["status"] == "in_progress" and started is None:
            started = time.monotonic()
        if batch["status"] in ("completed", "failed", "cancelled", "expired"):
            break
        time.sleep(1)
    elapsed = time.monotonic() - (started or time.monotonic())
    with urllib.request.urlopen(f"{URL}/v1/files/{batch['output_file_id']}/content") as r:
        lines = [json.loads(line) for line in r.read().splitlines() if line]
    return batch["request_counts"]["completed"] / max(elapsed, 1e-9), lines


def test_acceptance_7_and_the_packing_measurement():
    boundary = "millm026"
    data = open(FILE, "rb").read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.jsonl\""
            f"\r\nContent-Type: application/jsonl\r\n\r\n").encode() + data + \
        f"\r\n--{boundary}--\r\n".encode()
    file_id = _call("POST", "/v1/files", body,
                    {"Content-Type": f"multipart/form-data; boundary={boundary}"})["id"]
    single_rate, single = _run(file_id, pack=False)
    packed_rate, packed = _run(file_id, pack=True)
    by_id = {line["custom_id"]: line for line in single}
    diffs, agree = [], 0
    for line in packed:
        a = line["response"]["body"]["choices"][0]["logprobs"]
        b = by_id[line["custom_id"]]["response"]["body"]["choices"][0]["logprobs"]
        agree += a["tokens"] == b["tokens"]
        diffs.append(abs(a["token_logprobs"][0] - b["token_logprobs"][0]))
    print(json.dumps({"single_rows_per_s": single_rate, "packed_rows_per_s": packed_rate,
                      "max_abs_logprob_diff": max(diffs), "top_token_agreement": agree / len(packed)}))
    assert single_rate >= 19.0
