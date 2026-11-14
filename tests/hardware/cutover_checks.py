"""The P2 cutover checks (040 FR-040.29, FR-040.43; FTDD §6.10, TID §3.10).

Run by the operator or an agent around the P2 window; standard library and `httpx` only:

    python tests/hardware/cutover_checks.py preflight --base http://k8s-millm.hitsai.local [--require-no-pending]
    python tests/hardware/cutover_checks.py snapshot --base URL --out snapshot.json
    python tests/hardware/cutover_checks.py post --base http://k8s-miserver.hitsai.local \\
        --old-base http://k8s-millm.hitsai.local --incluster-result incluster.txt --snapshot snapshot.json

Each check prints one `PASS|FAIL|WARN <check> <detail>` line, then a summary; the exit code is 1 on
any FAIL, else 0.

* `preflight` is read-only. It blocks on every kind of work a restart breaks (BR-065): a model
  download or load, an executing approval, a Batch API batch not yet terminal, a lease, an attached
  trainer. Pending approvals survive a restart but an agent may be waiting on one: a warning with
  their count, or a block under `--require-no-pending` (041 uses it). A route it cannot read BLOCKS:
  a preflight that cannot see must not report clear.
* `snapshot` is read-only. It writes what a restart drops (`main.py` `STALE_STATE_RESETS`,
  `disarm_probes_on_startup`): loaded model and placement, attached SAEs, active circuits, armed
  probes and behaviours, active profile, telemetry switches; `taken_at` in UTC.
* `post` checks FR-040.29's list. It writes in two places only: one chat completion with
  `X-miLLM-Load-Policy: refuse` (it can never load a model), and one re-import with
  `on_conflict=fail` of a probe definition the server already stores, which must answer EXACTLY the
  name-conflict refusal (422, `VALIDATION_ERROR`, `details.code == "NAME_CONFLICT"`, the document's
  name; CF-9). Anything else, a 2xx included, is FAIL. The in-cluster address is called by the
  operator from a pod (`kubectl run --rm … curl -s -w '\\n%{http_code}' <url>/api/health`); the script
  checks the output it is handed: the body, then the HTTP status on the last line.

Every decision is a pure `check_*` function, unit-tested against `cutover_fake_server.py` (T26).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

TIMEOUT_S = 10.0
#: OpenAI's terminal batch statuses (`miserver/core/batch_values.py` `TERMINAL`).
TERMINAL_BATCH = frozenset({"completed", "failed", "cancelled", "expired"})
#: Headers every non-streaming chat completion carries (`miserver/api/provenance.py` `post_generation`);
#: frozen under their old names (register DR-08), never renamed.
FROZEN_CHAT_HEADERS = ("X-miLLM-Backend", "X-miLLM-Steering")
RENAMED_HEADER_PREFIX = "x-miserver-"
HEALTH_OK = ("healthy", "degraded")
PAGE_CAP = 100

PREFLIGHT_BLOCKERS = ("downloads", "loads", "executing_approvals", "batches", "lease", "attached_trainer")
SNAPSHOT_ITEMS = ("model", "saes", "circuits", "probes", "behaviours", "profile", "telemetry")
POST_CHECKS = ("health_new", "health_old_alias", "health_incluster", "frozen_headers", "reimport",
               "hf_token", "telemetry", "legacy_names", "snapshot")


@dataclass(frozen=True)
class Result:
    status: str  # PASS | FAIL | WARN
    check: str
    detail: str

    def line(self) -> str:
        return f"{self.status} {self.check} {self.detail}"


class Unreadable(Exception):
    """A route answered something the script cannot read; the check that needed it FAILs."""


def make_client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT_S, follow_redirects=False)


# ── reading ────────────────────────────────────────────────────────────────────────────────────


def _get(client: httpx.Client, url: str, **params: Any) -> Any:
    try:
        resp = client.get(url, params=params or None)
    except httpx.HTTPError as exc:
        raise Unreadable(f"{url}: {type(exc).__name__}") from exc
    if resp.status_code != 200:
        raise Unreadable(f"{url}: HTTP {resp.status_code}")
    try:
        return resp.json()
    except ValueError as exc:
        raise Unreadable(f"{url}: not JSON") from exc


def _data(envelope: Any, url: str) -> Any:
    if not isinstance(envelope, dict) or envelope.get("success") is not True:
        raise Unreadable(f"{url}: not a success envelope")
    return envelope.get("data")


def _pages(client: httpx.Client, url: str, **params: Any) -> list[Any]:
    """Every item of a `PageResponse` list, following `meta.next`."""
    items: list[Any] = []
    cursor = None
    for _ in range(PAGE_CAP):
        envelope = _get(client, url, **params, **({"cursor": cursor} if cursor else {}))
        data = _data(envelope, url)
        if not isinstance(data, list):
            raise Unreadable(f"{url}: data is not a list")
        items.extend(data)
        cursor = (envelope.get("meta") or {}).get("next")
        if not cursor:
            return items
    raise Unreadable(f"{url}: more than {PAGE_CAP} pages")


def _batches(client: httpx.Client, base: str) -> list[dict[str, Any]]:
    """Every batch, following OpenAI's `has_more` / `last_id`."""
    url = f"{base}/v1/batches"
    out: list[dict[str, Any]] = []
    after = None
    for _ in range(PAGE_CAP):
        page = _get(client, url, limit=100, **({"after": after} if after else {}))
        if not isinstance(page, dict) or not isinstance(page.get("data"), list):
            raise Unreadable(f"{url}: not an OpenAI list")
        out.extend(page["data"])
        if not page.get("has_more"):
            return out
        after = page.get("last_id")
        if not after:
            raise Unreadable(f"{url}: has_more without last_id")
    raise Unreadable(f"{url}: more than {PAGE_CAP} pages")


def _approvals(client: httpx.Client, base: str, status: str) -> list[dict[str, Any]]:
    url = f"{base}/api/approvals"
    data = _data(_get(client, url, status=status, limit=500), url)
    rows = (data or {}).get("approvals") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise Unreadable(f"{url}: no approvals list")
    return [r for r in rows if r.get("status") == status]  # a filter the server ignored still filters


# ── preflight decisions (FR-040.43) ────────────────────────────────────────────────────────────


def check_downloads(models: list[dict[str, Any]]) -> Result:
    busy = [m.get("name") for m in models if m.get("status") == "downloading"]
    return Result("FAIL" if busy else "PASS", "downloads",
                  f"downloading: {', '.join(map(str, busy))}" if busy else "none downloading")


def check_loads(models: list[dict[str, Any]]) -> Result:
    busy = [m.get("name") for m in models if m.get("status") == "loading"]
    return Result("FAIL" if busy else "PASS", "loads",
                  f"loading: {', '.join(map(str, busy))}" if busy else "none loading")


def check_executing_approvals(executing: list[dict[str, Any]]) -> Result:
    ids = [f"{a.get('approval_id')} ({a.get('action')})" for a in executing]
    return Result("FAIL" if ids else "PASS", "executing_approvals",
                  f"executing: {', '.join(ids)}" if ids else "none executing")


def check_pending_approvals(pending: list[dict[str, Any]], require_no_pending: bool) -> Result:
    if not pending:
        return Result("PASS", "pending_approvals", "0 pending")
    detail = f"{len(pending)} pending: " + ", ".join(f"{a.get('approval_id')} ({a.get('action')})" for a in pending)
    return Result("FAIL" if require_no_pending else "WARN", "pending_approvals", detail)


def check_batches(batches: list[dict[str, Any]]) -> Result:
    busy = [f"{b.get('id')} ({b.get('status')})" for b in batches if b.get("status") not in TERMINAL_BATCH]
    return Result("FAIL" if busy else "PASS", "batches",
                  f"not terminal: {', '.join(busy)}" if busy else f"{len(batches)} batches, all terminal")


def check_lease(health: dict[str, Any]) -> Result:
    lease = health.get("lease")
    if lease:
        return Result("FAIL", "lease", f"held by {lease.get('holder')} on {lease.get('model_name')} "
                                       f"({lease.get('seconds_remaining')} s left)")
    return Result("PASS", "lease", "no lease")


def check_attached_trainer(health: dict[str, Any]) -> Result:
    weights = health.get("weights")
    if not isinstance(weights, dict) or weights.get("error") or "attached" not in weights:
        reason = weights.get("error") if isinstance(weights, dict) else "no weights block"
        return Result("FAIL", "attached_trainer", f"cannot tell: {reason}")
    if weights.get("attached"):
        return Result("FAIL", "attached_trainer", f"a trainer is attached ({weights.get('attachment_state')})")
    return Result("PASS", "attached_trainer", "nothing attached")


def check_inference(health: dict[str, Any]) -> Result:
    inf = health.get("inference") or {}
    busy = {k: inf.get(k) for k in ("in_flight", "queue_waiting") if inf.get(k)}
    return Result("WARN" if busy else "PASS", "inference",
                  f"requests running: {busy}" if busy else "no request running")


def preflight(client: httpx.Client, base: str, require_no_pending: bool = False) -> list[Result]:
    out: list[Result] = []

    def run(check: str, fn: Any) -> None:
        try:
            out.append(fn())
        except Unreadable as exc:
            out.append(Result("FAIL", check, f"unreadable: {exc}"))

    health: dict[str, Any] = {}
    try:
        health = _get(client, f"{base}/api/health/detailed")
    except Unreadable as exc:
        health = {"_error": str(exc)}

    def need_health() -> dict[str, Any]:
        if "_error" in health:
            raise Unreadable(health["_error"])
        return health

    models: list[dict[str, Any]] | None = None

    def need_models() -> list[dict[str, Any]]:
        nonlocal models
        if models is None:
            data = _data(_get(client, f"{base}/api/models"), f"{base}/api/models")
            if not isinstance(data, list):
                raise Unreadable(f"{base}/api/models: data is not a list")
            models = data
        return models

    run("downloads", lambda: check_downloads(need_models()))
    run("loads", lambda: check_loads(need_models()))
    run("executing_approvals", lambda: check_executing_approvals(_approvals(client, base, "executing")))
    run("pending_approvals", lambda: check_pending_approvals(_approvals(client, base, "pending"), require_no_pending))
    run("batches", lambda: check_batches(_batches(client, base)))
    run("lease", lambda: check_lease(need_health()))
    run("attached_trainer", lambda: check_attached_trainer(need_health()))
    run("inference", lambda: check_inference(need_health()))
    return out


# ── snapshot (FR-040.43) ───────────────────────────────────────────────────────────────────────


def snapshot_items(health: dict[str, Any], armed_probes: list[dict[str, Any]],
                   behaviours: list[dict[str, Any]], circuits: Any) -> dict[str, Any]:
    telemetry = health.get("telemetry") or {}
    profile = health.get("active_profile")
    return {
        "model": ({"name": health.get("model_name"), "placement": health.get("model_placement")}
                  if health.get("model_loaded") else None),
        "saes": list(health.get("sae_ids") or []),
        "circuits": sorted(str(c.get("id")) for c in (circuits or [])),
        "probes": sorted(str(p.get("id")) for p in armed_probes),
        "behaviours": sorted(str(b.get("key")) for b in behaviours if b.get("armed")),
        "profile": {"id": profile.get("id"), "name": profile.get("name")} if profile else None,
        "telemetry": {"enabled": telemetry.get("enabled"),
                      "stateless_export_permitted": (telemetry.get("config") or {}).get("stateless_export_permitted")},
    }


def read_state(client: httpx.Client, base: str) -> dict[str, Any]:
    health = _get(client, f"{base}/api/health/detailed")
    armed = _pages(client, f"{base}/api/probes", armed="true")
    behaviours = _pages(client, f"{base}/api/probes/behaviours")
    circuits = _data(_get(client, f"{base}/api/circuits/active"), f"{base}/api/circuits/active")
    return snapshot_items(health, armed, behaviours, circuits)


# ── post-window decisions (FR-040.29) ──────────────────────────────────────────────────────────


def check_health(check: str, status: int | None, body: Any) -> Result:
    if status != 200:
        return Result("FAIL", check, f"HTTP {status}")
    if not isinstance(body, dict) or body.get("status") not in HEALTH_OK:
        return Result("FAIL", check, f"not a health answer: {str(body)[:80]!r}")
    return Result("PASS", check, f"{body.get('status')} {body.get('version', '')}".strip())


def check_frozen_headers(status: int | None, headers: dict[str, str]) -> Result:
    if status != 200:
        return Result("FAIL", "frozen_headers", f"chat completion answered HTTP {status}")
    names = {k.lower() for k in headers}
    missing = [h for h in FROZEN_CHAT_HEADERS if h.lower() not in names]
    renamed = sorted(k for k in names if k.startswith(RENAMED_HEADER_PREFIX))
    if missing or renamed:
        return Result("FAIL", "frozen_headers", f"missing {missing}, renamed {renamed}")
    return Result("PASS", "frozen_headers", ", ".join(FROZEN_CHAT_HEADERS))


def check_reimport(status: int | None, body: Any, name: str) -> Result:
    """PASS only on the exact name-conflict refusal (CF-9)."""
    error = body.get("error") if isinstance(body, dict) else None
    details = (error or {}).get("details") if isinstance(error, dict) else None
    got = (status, (error or {}).get("code") if isinstance(error, dict) else None,
           (details or {}).get("code") if isinstance(details, dict) else None,
           (details or {}).get("name") if isinstance(details, dict) else None)
    if got == (422, "VALIDATION_ERROR", "NAME_CONFLICT", name):
        return Result("PASS", "reimport", f"NAME_CONFLICT for {name!r}")
    return Result("FAIL", "reimport", f"expected 422 VALIDATION_ERROR/NAME_CONFLICT for {name!r}, got {got}")


def check_hf_token(data: Any) -> Result:
    if isinstance(data, dict) and data.get("stored") is True and data.get("stored_error") is None:
        return Result("PASS", "hf_token", "stored and readable")
    shown = {k: data.get(k) for k in ("stored", "stored_error")} if isinstance(data, dict) else data
    return Result("FAIL", "hf_token", f"not readable: {shown}")


def check_telemetry(health: dict[str, Any]) -> Result:
    block = health.get("telemetry")
    if not isinstance(block, dict) or block.get("error") or not isinstance(block.get("enabled"), bool):
        return Result("FAIL", "telemetry", f"block missing or in error: {str(block)[:80]!r}")
    return Result("PASS", "telemetry", f"enabled={block['enabled']}")


def check_legacy_names(health: dict[str, Any], old_host: str) -> Result:
    in_use = ((health.get("legacy_names") or {}).get("in_use")) or []
    if any(u.get("kind") == "host" and str(u.get("old", "")).lower() == old_host.lower() for u in in_use):
        return Result("PASS", "legacy_names", f"in_use lists host {old_host}")
    return Result("FAIL", "legacy_names", f"in_use does not list host {old_host}: {in_use}")


def check_snapshot(before: dict[str, Any], now: dict[str, Any] | None, left_down: set[str]) -> Result:
    if now is None:
        return Result("FAIL", "snapshot", "current state unreadable")
    lost: dict[str, Any] = {}
    for item, was in before.items():
        is_ = now.get(item)
        if isinstance(was, list):
            gone = sorted(set(was) - set(is_ or []))
            if gone:
                lost[item] = gone
        elif was != is_ and was is not None:
            lost[item] = was
    if not lost:
        return Result("PASS", "snapshot", f"{len(before)} items restored")
    unaccounted = {k: v for k, v in lost.items() if k not in left_down}
    detail = "; ".join(f"{k} not restored: {v}" for k, v in lost.items())
    return Result("FAIL" if unaccounted else "WARN", "snapshot",
                  detail + ("" if unaccounted else " (left down on purpose)"))


def parse_incluster(text: str) -> tuple[int | None, Any]:
    """`curl -s -w '\\n%{http_code}'` output: the body, then the status on the last line."""
    lines = text.strip().splitlines()
    if not lines:
        return None, None
    try:
        status = int(lines[-1].strip())
    except ValueError:
        return None, text
    raw = "\n".join(lines[:-1])
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw


def post(client: httpx.Client, base: str, old_base: str, incluster_text: str, snapshot: dict[str, Any],
         left_down: set[str], model: str | None = None, probe_file: Path | None = None) -> list[Result]:
    out: list[Result] = []

    def health_at(check: str, url: str) -> None:
        try:
            resp = client.get(f"{url}/api/health")
        except httpx.HTTPError as exc:
            out.append(Result("FAIL", check, f"unreachable: {type(exc).__name__}"))
            return
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        out.append(check_health(check, resp.status_code, body))

    health_at("health_new", base)
    health_at("health_old_alias", old_base)  # recorded by the host middleware: legacy_names below
    out.append(check_health("health_incluster", *parse_incluster(incluster_text)))

    items = snapshot.get("items") or {}
    model = model or (items.get("model") or {}).get("name")
    if not model:
        out.append(Result("FAIL", "frozen_headers", "no model name: pass --model or a snapshot with one"))
    else:
        try:
            resp = client.post(f"{base}/v1/chat/completions",
                               json={"model": model, "max_tokens": 1,
                                     "messages": [{"role": "user", "content": "cutover check"}]},
                               headers={"X-miLLM-Load-Policy": "refuse"})
            out.append(check_frozen_headers(resp.status_code, dict(resp.headers)))
        except httpx.HTTPError as exc:
            out.append(Result("FAIL", "frozen_headers", f"unreachable: {type(exc).__name__}"))

    try:
        if probe_file is not None:
            document = json.loads(probe_file.read_text())
        else:
            first = _data(_get(client, f"{base}/api/probes", limit=1), f"{base}/api/probes")
            if not first:
                raise Unreadable("no stored probe to re-import; pass --probe-file")
            url = f"{base}/api/probes/{first[0]['id']}"
            document = (_data(_get(client, url), url) or {}).get("definition")
        if not isinstance(document, dict) or not document.get("name"):
            raise Unreadable("the stored definition has no name")
        resp = client.post(f"{base}/api/probes/import", params={"on_conflict": "fail"}, json=document)
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        out.append(check_reimport(resp.status_code, body, document["name"]))
    except (Unreadable, httpx.HTTPError, OSError, ValueError) as exc:
        out.append(Result("FAIL", "reimport", f"could not run: {exc}"))

    try:
        url = f"{base}/api/settings/hf-token"
        out.append(check_hf_token(_data(_get(client, url), url)))
    except Unreadable as exc:
        out.append(Result("FAIL", "hf_token", f"unreadable: {exc}"))

    try:
        health = _get(client, f"{base}/api/health/detailed")  # read AFTER the old-alias call above
        out.append(check_telemetry(health))
        out.append(check_legacy_names(health, urlparse(old_base).hostname or old_base))
    except Unreadable as exc:
        out.append(Result("FAIL", "telemetry", f"unreadable: {exc}"))
        out.append(Result("FAIL", "legacy_names", f"unreadable: {exc}"))

    try:
        now = read_state(client, base)
    except Unreadable:
        now = None
    out.append(check_snapshot(items, now, left_down))
    return out


# ── the command line ───────────────────────────────────────────────────────────────────────────


def _report(results: list[Result]) -> int:
    for r in results:
        print(r.line())
    failed = [r.check for r in results if r.status == "FAIL"]
    warned = [r.check for r in results if r.status == "WARN"]
    print(f"SUMMARY {len(results)} checks, {len(failed)} FAIL {failed}, {len(warned)} WARN {warned}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--base", required=True)
    p.add_argument("--require-no-pending", action="store_true")
    s = sub.add_parser("snapshot")
    s.add_argument("--base", required=True)
    s.add_argument("--out", required=True, type=Path)
    q = sub.add_parser("post")
    q.add_argument("--base", required=True)
    q.add_argument("--old-base", required=True)
    q.add_argument("--incluster-result", required=True, type=Path)
    q.add_argument("--snapshot", required=True, type=Path)
    q.add_argument("--model")
    q.add_argument("--probe-file", type=Path)
    q.add_argument("--left-down", nargs="*", default=[], choices=SNAPSHOT_ITEMS,
                   help="snapshot items deliberately left down (reported WARN, not FAIL)")
    args = parser.parse_args(argv)
    base = args.base.rstrip("/")
    with make_client() as client:
        if args.command == "preflight":
            return _report(preflight(client, base, args.require_no_pending))
        if args.command == "snapshot":
            try:
                items = read_state(client, base)
            except Unreadable as exc:
                return _report([Result("FAIL", "snapshot", f"unreadable: {exc}")])
            taken_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
            args.out.write_text(json.dumps({"taken_at": taken_at, "base": base, "items": items}, indent=2) + "\n")
            return _report([Result("PASS", "snapshot", f"{args.out} ({', '.join(k for k, v in items.items() if v)})")])
        snapshot = json.loads(args.snapshot.read_text())
        return _report(post(client, base, args.old_base.rstrip("/"), args.incluster_result.read_text(),
                            snapshot, set(args.left_down), args.model, args.probe_file))


if __name__ == "__main__":
    sys.exit(main())
