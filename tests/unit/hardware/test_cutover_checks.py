"""T26 (040 FR-040.29, FR-040.43; FTDD §6.10, TID §3.10): the P2 cutover script, on the CPU, against
the fake server `tests/hardware/cutover_fake_server.py`.

What must hold, and why each test exists:

* the preflight blocks (exit 1, the condition named) on every kind of work a restart would break: a
  download, a load, an executing approval, a non-terminal batch (on any page of the list), a lease,
  an attached trainer; with everything clear it exits 0. Pending approvals warn, or block under
  `--require-no-pending` (041 uses it). A route it cannot read blocks: a check that cannot see must
  not pass.
* the snapshot lists what a restart drops (loaded model and placement, SAEs, active circuits, armed
  probes and behaviours, the active profile, telemetry switches);
* each post-window check FAILs when its fake answer is wrong, and passes on the right one;
* the re-import check passes ONLY on the exact name-conflict refusal (422, `VALIDATION_ERROR`,
  `details.code == "NAME_CONFLICT"`, the document's name) and fails on every other refusal and on a
  2xx (CF-9).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests" / "hardware"))
import cutover_checks as cc  # noqa: E402
import cutover_fake_server as fake_mod  # noqa: E402


@pytest.fixture
def fake():
    server = fake_mod.FakeServer().start()
    yield server
    server.stop()


def _by_check(results):
    return {r.check: r for r in results}


def _run(argv, capsys):
    code = cc.main(argv)
    out = capsys.readouterr().out
    return code, out


# ── preflight (FR-040.43) ────────────────────────────────────────────────────────────────────


def test_preflight_passes_when_clear(fake, capsys):
    code, out = _run(["preflight", "--base", fake.base], capsys)
    assert code == 0, out
    lines = [line for line in out.splitlines() if line.split(" ", 1)[0] in ("PASS", "FAIL", "WARN")]
    assert {line.split()[1] for line in lines} >= set(cc.PREFLIGHT_BLOCKERS), out
    assert not [line for line in lines if line.startswith("FAIL")], out


@pytest.mark.parametrize("condition", list(fake_mod.PREFLIGHT_CONDITIONS))
def test_preflight_blocks_on(fake, capsys, condition):
    fake.busy.add(condition)
    code, out = _run(["preflight", "--base", fake.base], capsys)
    assert code == 1, out
    failed = [line for line in out.splitlines() if line.startswith("FAIL ")]
    assert [line.split()[1] for line in failed] == [fake_mod.PREFLIGHT_CONDITIONS[condition]], out
    assert fake_mod.WHAT[condition] in failed[0], failed  # the blocking item is named


def test_preflight_reads_every_page_of_batches(fake, capsys):
    """The non-terminal batch sits on the LAST page; a preflight reading one page would pass."""
    fake.busy.add("non_terminal_batch")
    fake.batch_page_size = 2
    code, out = _run(["preflight", "--base", fake.base], capsys)
    assert code == 1 and "FAIL batches" in out, out
    assert fake.calls["GET /v1/batches"] >= 3, fake.calls


def test_pending_approvals_warn_or_block_with_require_no_pending(fake, capsys):
    fake.busy.add("pending_approval")
    code, out = _run(["preflight", "--base", fake.base], capsys)
    assert code == 0, out
    assert "WARN pending_approvals 1 pending" in out, out
    code, out = _run(["preflight", "--base", fake.base, "--require-no-pending"], capsys)
    assert code == 1, out
    assert "FAIL pending_approvals 1 pending" in out, out


def test_preflight_blocks_when_a_route_cannot_be_read(fake, capsys):
    fake.defects.add("approvals_500")
    code, out = _run(["preflight", "--base", fake.base], capsys)
    assert code == 1, out
    assert "FAIL executing_approvals" in out and "unreadable" in out, out


def test_preflight_blocks_when_the_weights_block_is_an_error(fake, capsys):
    fake.defects.add("weights_error")
    code, out = _run(["preflight", "--base", fake.base], capsys)
    assert code == 1 and "FAIL attached_trainer" in out, out


# ── snapshot (FR-040.43) ─────────────────────────────────────────────────────────────────────


def test_snapshot_lists_what_a_restart_drops(fake, tmp_path, capsys):
    out_file = tmp_path / "snap.json"
    code, out = _run(["snapshot", "--base", fake.base, "--out", str(out_file)], capsys)
    assert code == 0, out
    snap = json.loads(out_file.read_text())
    assert snap["taken_at"].endswith("Z")
    items = snap["items"]
    assert set(items) == set(cc.SNAPSHOT_ITEMS)
    assert items["model"] == {"name": fake_mod.MODEL, "placement": fake_mod.PLACEMENT}
    assert items["saes"] == fake_mod.SAES
    assert items["circuits"] == fake_mod.CIRCUITS
    assert items["probes"] == sorted(fake_mod.ARMED_PROBES)  # across both pages
    assert items["behaviours"] == fake_mod.ARMED_BEHAVIOURS
    assert items["profile"] == fake_mod.PROFILE
    assert items["telemetry"] == {"enabled": True, "stateless_export_permitted": False}
    assert fake.calls["GET /api/probes"] == 2, fake.calls  # it followed `meta.next`


# ── post-window checks (FR-040.29) ───────────────────────────────────────────────────────────


def _snapshot(fake, tmp_path):
    path = tmp_path / "snap.json"
    assert cc.main(["snapshot", "--base", fake.base, "--out", str(path)]) == 0
    return path


def _incluster(tmp_path, status=200, body=None):
    path = tmp_path / "incluster.txt"
    body = {"status": "healthy", "version": "x"} if body is None else body
    path.write_text(json.dumps(body) + f"\n{status}\n")
    return path


def _post_argv(fake, tmp_path, snap, incluster=None):
    return ["post", "--base", fake.base, "--old-base", fake.old_base,
            "--incluster-result", str(incluster or _incluster(tmp_path)), "--snapshot", str(snap)]


def test_post_passes_when_every_answer_is_right(fake, tmp_path, capsys):
    snap = _snapshot(fake, tmp_path)
    capsys.readouterr()
    code, out = _run(_post_argv(fake, tmp_path, snap), capsys)
    assert code == 0, out
    checks = {line.split()[1] for line in out.splitlines() if line.startswith("PASS ")}
    assert checks >= set(cc.POST_CHECKS), out
    # Payload and call count: one chat that may never load a model, one re-import of the stored
    # document with on_conflict=fail.
    assert fake.calls["POST /v1/chat/completions"] == 1
    assert fake.chat_headers[-1].get("x-millm-load-policy") == "refuse"
    assert fake.calls["POST /api/probes/import"] == 1
    assert fake.imports[-1]["query"] == {"on_conflict": ["fail"]}
    assert fake.imports[-1]["body"] == fake_mod.STORED_DEFINITION


POST_DEFECTS = {
    "headers_renamed": {"frozen_headers"},
    "headers_missing": {"frozen_headers"},
    "hf_token_unreadable": {"hf_token"},
    "old_alias_down": {"health_old_alias"},
    "new_host_down": {"health_new"},
    # No block: the switches it carried cannot be shown restored either.
    "telemetry_missing": {"telemetry", "snapshot"},
    "legacy_host_not_recorded": {"legacy_names"},
    "probes_not_rearmed": {"snapshot"},
}


@pytest.mark.parametrize("defect", list(POST_DEFECTS))
def test_post_check_fails_on_wrong_answer(fake, tmp_path, capsys, defect):
    snap = _snapshot(fake, tmp_path)
    fake.defects.add(defect)
    capsys.readouterr()
    code, out = _run(_post_argv(fake, tmp_path, snap), capsys)
    assert code == 1, out
    failed = {line.split()[1] for line in out.splitlines() if line.startswith("FAIL ")}
    assert failed == POST_DEFECTS[defect], out


@pytest.mark.parametrize("status,body", [
    (503, {"status": "unhealthy"}),
    (200, None),  # a body that is not the health answer
])
def test_post_check_fails_on_wrong_answer_incluster(fake, tmp_path, capsys, status, body):
    snap = _snapshot(fake, tmp_path)
    incluster = tmp_path / "incluster.txt"
    incluster.write_text(("<html>nginx</html>" if body is None else json.dumps(body)) + f"\n{status}\n")
    capsys.readouterr()
    code, out = _run(_post_argv(fake, tmp_path, snap, incluster), capsys)
    assert code == 1, out
    assert {line.split()[1] for line in out.splitlines() if line.startswith("FAIL ")} == {"health_incluster"}, out


def test_snapshot_item_left_down_on_purpose_is_a_warning(fake, tmp_path, capsys):
    snap = _snapshot(fake, tmp_path)
    fake.defects.add("probes_not_rearmed")
    capsys.readouterr()
    code, out = _run(_post_argv(fake, tmp_path, snap) + ["--left-down", "probes", "behaviours"], capsys)
    assert code == 0, out
    assert "WARN snapshot" in out and "probes" in out, out


# ── the re-import check (CF-9) ───────────────────────────────────────────────────────────────


NAME = fake_mod.STORED_DEFINITION["name"]


def test_reimport_check_passes_only_on_name_conflict():
    right = {"success": False, "error": {"code": "VALIDATION_ERROR", "message": "exists",
                                         "details": {"name": NAME, "code": "NAME_CONFLICT"}}}
    assert cc.check_reimport(422, right, NAME).status == "PASS"
    assert cc.check_reimport(409, right, NAME).status == "FAIL"  # the status is part of the answer
    other_name = json.loads(json.dumps(right))
    other_name["error"]["details"]["name"] = NAME + " (2)"
    assert cc.check_reimport(422, other_name, NAME).status == "FAIL"
    other_code = json.loads(json.dumps(right))
    other_code["error"]["code"] = "CONFLICT"
    assert cc.check_reimport(422, other_code, NAME).status == "FAIL"
    assert cc.check_reimport(422, "not json", NAME).status == "FAIL"


@pytest.mark.parametrize("refusal", list(fake_mod.REIMPORT_REFUSALS))
def test_reimport_check_fails_on(fake, tmp_path, capsys, refusal):
    snap = _snapshot(fake, tmp_path)
    fake.reimport = refusal
    capsys.readouterr()
    code, out = _run(_post_argv(fake, tmp_path, snap), capsys)
    assert code == 1, out
    failed = [line for line in out.splitlines() if line.startswith("FAIL ")]
    assert [line.split()[1] for line in failed] == ["reimport"], out
    status, body = fake_mod.REIMPORT_REFUSALS[refusal]
    assert cc.check_reimport(status, body, NAME).status == "FAIL"


# ── the client side ──────────────────────────────────────────────────────────────────────────


def test_a_server_that_does_not_answer_fails_every_post_check(tmp_path, capsys):
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps({"taken_at": "2026-11-10T00:00:00Z", "items": {}}))
    code, out = _run(["post", "--base", "http://127.0.0.1:9", "--old-base", "http://localhost:9",
                      "--incluster-result", str(_incluster(tmp_path)), "--snapshot", str(snap)], capsys)
    assert code == 1, out
    assert "PASS frozen_headers" not in out and "PASS reimport" not in out, out


def test_the_script_uses_a_timeout():
    client = cc.make_client()
    try:
        assert isinstance(client, httpx.Client)
        assert client.timeout.read == cc.TIMEOUT_S == 10.0
    finally:
        client.close()
