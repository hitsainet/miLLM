"""miLLM's Ingress passes the probe verdict header it promises (suite decision D59, 2026-10-24).

The 2026-10-24 GPU window (U3, `0xcc/reviews/gpu_window_2026-10-24.md`) measured the public chain:
ingress-nginx's default `proxy_buffer_size` (4 KB) caps the WHOLE upstream response-header block,
so a verdict header over 3,932 bytes came back as a 502; with the buffer raised, Cloudflare and
the .10 proxy passed 81,918 bytes. D59: `millm-ingress` carries `proxy-buffer-size: "16k"` and
`PROBE_VERDICT_HEADER_MAX_BYTES` defaults to 8,192 (its 256–65,536 range kept).

Parsed from `k8s/base/ingress.yaml` (ArgoCD deploys `k8s/base`). Control (039 controls record,
review R2): remove the annotation → red; default back to 2,048 → red.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from millm.core.config import Settings

REPO = Path(__file__).resolve().parents[2]
ANNOTATION = "nginx.ingress.kubernetes.io/proxy-buffer-size"
#: Everything else in the upstream response-header block at U3 (status line, Server, Date,
#: Content-Length, Content-Type, CRLFs) was 164 bytes; a real chat response carries a few more.
OTHER_HEADERS_ALLOWANCE = 2048


def _ingresses() -> dict[str, dict]:
    docs = [d for d in yaml.safe_load_all((REPO / "k8s/base/ingress.yaml").read_text()) if d]
    return {d["metadata"]["name"]: d for d in docs if d["kind"] == "Ingress"}


def _bytes(size: str) -> int:
    units = {"k": 1024, "m": 1024 * 1024}
    return int(size[:-1]) * units[size[-1].lower()] if size[-1].lower() in units else int(size)


def test_the_millm_ingress_raises_the_proxy_buffer_to_16k():
    ingress = _ingresses()["millm-ingress"]
    assert ingress["metadata"]["annotations"][ANNOTATION] == "16k"


def test_the_buffer_covers_every_host_and_route_that_serves_the_verdict_header():
    """The annotation is per Ingress object: it must be the one that routes `/v1` (chat, where
    the header is served) and `/api` (the probe routes) for BOTH the LAN and the public host."""
    ingress = _ingresses()["millm-ingress"]
    routed = {(rule["host"], path["path"]) for rule in ingress["spec"]["rules"]
              for path in rule["http"]["paths"]
              if path["backend"]["service"]["name"] == "millm-backend"}
    for host in ("k8s-millm.hitsai.local", "k8s-millm.hitsai.net"):
        assert {(host, "/v1"), (host, "/api")} <= routed, routed


def test_the_default_header_bound_is_8192_and_fits_the_ingress_buffer():
    default = Settings.model_fields["PROBE_VERDICT_HEADER_MAX_BYTES"].default
    assert default == 8192
    buffer = _bytes(_ingresses()["millm-ingress"]["metadata"]["annotations"][ANNOTATION])
    assert default + OTHER_HEADERS_ALLOWANCE <= buffer


def test_the_settings_range_is_unchanged():
    from millm.services import runtime_settings

    entry = runtime_settings.RUNTIME_SPECS["PROBE_VERDICT_HEADER_MAX_BYTES"]
    assert (entry.minimum, entry.maximum) == (256, 65536)
