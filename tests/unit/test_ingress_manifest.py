"""miServer's Ingress objects: the probe verdict header fits (D59), and since P7 no old host is served
(040 FR-040.24, T20, FR-040.40; CF-5 M8; R20).

**D59 (2026-10-24).** The GPU window (U3, `0xcc/reviews/gpu_window_2026-10-24.md`) measured the public
chain: ingress-nginx's default `proxy_buffer_size` (4 KB) caps the WHOLE upstream response-header
block, so a verdict header over 3,932 bytes came back as a 502; with the buffer raised, Cloudflare and
the .10 proxy passed 81,918 bytes. D59: the main Ingress carries `proxy-buffer-size: "16k"` and
`PROBE_VERDICT_HEADER_MAX_BYTES` defaults to 8,192 (its 256–65,536 range kept). Control (039 controls
record, review R2): remove the annotation → red; default back to 2,048 → red.

**Aliases (040 P2 to P7).** From P2 each old host was served beside its new twin on the same Ingress.
P7 retired them on evidence (FR-040.40): the export's `host` records are `retired`, `HOST_ALIASES` is
empty, and no Ingress host may carry an old-brand token. No Ingress host is `k8s-*.hitsai.net` (R20).
Read over the raw parsed YAML under `k8s/` (floor: three Ingress objects) and over the
`kubectl kustomize` render of the base and every overlay (that half skips loudly without kubectl).
Controls NC-77, NC-78, NC-96; P7: put `k8s-millm.hitsai.local` back on either Ingress → red.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from miserver.core.config import Settings
from tests.support import legacy_identifiers as li

REPO = Path(__file__).resolve().parents[2]
ANNOTATION = "nginx.ingress.kubernetes.io/proxy-buffer-size"
MAIN_INGRESS = "miserver-ingress"
BACKEND = "miserver-backend"
#: Everything else in the upstream response-header block at U3 (status line, Server, Date,
#: Content-Length, Content-Type, CRLFs) was 164 bytes; a real chat response carries a few more.
OTHER_HEADERS_ALLOWANCE = 2048
OLD_BRAND = re.compile(r"(?i)mi(llm|studio)")
#: Directories `kubectl kustomize` renders (what Argo CD deploys, and each opt-in overlay).
RENDERED = ["k8s/base", "k8s/overlays/hostipc", "k8s/overlays/otel-export",
            "k8s/overlays/otel-reference", "k8s/open-webui"]


def _raw_ingresses() -> list[dict]:
    out = []
    for path in sorted((REPO / "k8s").rglob("*.yaml")):
        out += [d for d in yaml.safe_load_all(path.read_text())
                if isinstance(d, dict) and d.get("kind") == "Ingress"]
    return out


def _rendered_ingresses() -> list[dict]:
    kubectl = shutil.which("kubectl")
    if kubectl is None:
        pytest.skip("kubectl not installed: the RENDERED Ingress checks did NOT run")
    out = []
    for rel in RENDERED:
        text = subprocess.run([kubectl, "kustomize", str(REPO / rel)], capture_output=True, text=True,
                              check=True).stdout
        out += [d for d in yaml.safe_load_all(text) if isinstance(d, dict) and d.get("kind") == "Ingress"]
    return out


SOURCES = {"raw": _raw_ingresses, "rendered": _rendered_ingresses}


def _ingresses() -> dict[str, dict]:
    docs = [d for d in yaml.safe_load_all((REPO / "k8s/base/ingress.yaml").read_text()) if d]
    return {d["metadata"]["name"]: d for d in docs if d["kind"] == "Ingress"}


def _routes(ingress: dict) -> dict[str, dict[str, tuple[str, int]]]:
    """`{host: {path: (service, port)}}` of one Ingress."""
    return {rule["host"]: {p["path"]: (p["backend"]["service"]["name"],
                                       p["backend"]["service"]["port"]["number"])
                           for p in rule["http"]["paths"]}
            for rule in ingress["spec"]["rules"]}


def _bytes(size: str) -> int:
    units = {"k": 1024, "m": 1024 * 1024}
    return int(size[:-1]) * units[size[-1].lower()] if size[-1].lower() in units else int(size)


def test_the_main_ingress_raises_the_proxy_buffer_to_16k():
    ingress = _ingresses()[MAIN_INGRESS]
    assert ingress["metadata"]["annotations"][ANNOTATION] == "16k"


def test_the_buffer_covers_every_host_and_route_that_serves_the_verdict_header():
    """The annotation is per Ingress object: it must be the one that routes `/v1` (chat, where
    the header is served) and `/api` (the probe routes) for the host."""
    ingress = _ingresses()[MAIN_INGRESS]
    routed = {(rule["host"], path["path"]) for rule in ingress["spec"]["rules"]
              for path in rule["http"]["paths"]
              if path["backend"]["service"]["name"] == BACKEND}
    host = "k8s-miserver.hitsai.local"
    assert {(host, "/v1"), (host, "/api")} <= routed, routed
    # Every Ingress that routes to the backend's HTTP paths carries the buffer (the WebSocket one
    # routes only /socket.io).
    for name, other in _ingresses().items():
        paths = {p["path"] for r in other["spec"]["rules"] for p in r["http"]["paths"]}
        if paths & {"/v1", "/api"}:
            assert other["metadata"]["annotations"].get(ANNOTATION) == "16k", name


def test_the_default_header_bound_is_8192_and_fits_the_ingress_buffer():
    default = Settings.model_fields["PROBE_VERDICT_HEADER_MAX_BYTES"].default
    assert default == 8192
    buffer = _bytes(_ingresses()[MAIN_INGRESS]["metadata"]["annotations"][ANNOTATION])
    assert default + OTHER_HEADERS_ALLOWANCE <= buffer


def test_the_settings_range_is_unchanged():
    from miserver.services import runtime_settings

    entry = runtime_settings.RUNTIME_SPECS["PROBE_VERDICT_HEADER_MAX_BYTES"]
    assert (entry.minimum, entry.maximum) == (256, 65536)


# ── aliases retired (040 P7, T20 inverse) ──────────────────────────────────────────


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_no_old_brand_ingress_host(source):
    """P7 (FR-040.40): every old host the export carries is `retired`, and no Ingress serves an
    old-brand host (each would answer while the server no longer watches it)."""
    ingresses = SOURCES[source]()
    assert len(ingresses) >= 3, f"{source}: only {len(ingresses)} Ingress objects: failing open"
    hosts = {h for ing in ingresses for h in _routes(ing)}
    assert "k8s-miserver.hitsai.local" in hosts, f"{source}: {hosts}"
    old = sorted(h for h in hosts if OLD_BRAND.search(h))
    assert not old, f"{source}: old-brand Ingress hosts after P7: {old}"
    retired = {r["old"] for r in li.load().hosts() if r["status"] == "retired"}
    assert not hosts & retired, f"{source}: retired hosts still served: {sorted(hosts & retired)}"


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_no_public_k8s_net_host(source):
    """R20 (2026-11-08): no public `k8s-*.hitsai.net` name exists for the new brand; the public name
    is `miserver.hitsai.net` through the tunnel."""
    hosts = {h for ing in SOURCES[source]() for h in _routes(ing)}
    bad = sorted(h for h in hosts if re.fullmatch(r"k8s-[a-z0-9-]+\.hitsai\.net", h))
    assert not bad, f"{source}: {bad}"
