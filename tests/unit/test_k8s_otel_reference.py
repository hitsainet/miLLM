"""The reference OpenTelemetry receiver is opt-in, pinned, GPU-free and internal (038 FR-38.25).

`k8s/components/otel-reference` (rendered by `k8s/overlays/otel-reference`, deployed by its own
hand-applied ArgoCD Application) holds the Collector and the grafana/otel-lgtm backend in namespace
`millm-otel`. `k8s/components/otel-export` is the later switch-on (FTASKS 14.0). What this file
holds the line on, one test each:

* **the base is telemetry-free** (control M26): `k8s/base` — what the `millm` Application deploys —
  sets no `TELEMETRY_*`/`OTEL_*` variable and names no Collector, so a push never switches export
  on and never deploys the receiver;
* the overlay renders the namespace, the Collector, the backend, the volume and the dashboard, all
  in `millm-otel`, with only the Namespace and the one PersistentVolume cluster-scoped;
* every image is pinned by tag AND digest, the same in Kubernetes and Compose;
* nothing requests a GPU and every container sets `NVIDIA_VISIBLE_DEVICES=void` (the node's default
  runtime is nvidia, so silence means every card);
* retention is 7 days in all three stores AND the files that say so are the ones mounted;
* no public host: the only Ingress is a `k8s-*.hitsai.local` name to Grafana; the Collector is a
  ClusterIP Service with no Ingress;
* the `otel-export` render turns the three switches on, points at the reference Collector's own
  Service, keeps stateless export off, and changes nothing else;
* one Collector configuration (and one set of retention files and dashboard) for Kubernetes and
  Compose;
* the dashboard is provisioned, and its queries name only series and labels the LIVE meter provider
  registers and only record attributes the allowlist lets out;
* nothing leaves the cluster (review R1 of S3): the bundled Grafana, Loki, Tempo and Pyroscope
  report nothing and fetch nothing, Grafana has no login path to its built-in admin/admin, the
  backend pod has no egress and the Collector reaches only the backend's OTLP port and DNS.

The rendered checks run `kubectl kustomize` and skip LOUDLY without it (as the hostIPC test); the
structural checks need no binary.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "k8s" / "base"
COMPONENT = ROOT / "k8s" / "components" / "otel-reference"
OVERLAY = ROOT / "k8s" / "overlays" / "otel-reference"
EXPORT_OVERLAY = ROOT / "k8s" / "overlays" / "otel-export"
EXPORT_COMPONENT = ROOT / "k8s" / "components" / "otel-export"
APP = ROOT / "k8s" / "argocd" / "millm-otel-app.yaml"
MILLM_APP = ROOT / "k8s" / "argocd" / "millm-app.yaml"
COMPOSE = ROOT / "docker-compose.otel.yml"
NAMESPACE = "millm-otel"
PINNED = re.compile(r"^[a-z0-9./-]+:[0-9][0-9A-Za-z.\-]*@sha256:[0-9a-f]{64}$")
CLUSTER_SCOPED = {"Namespace", "PersistentVolume"}


def _render(path: Path) -> list[dict]:
    kubectl = shutil.which("kubectl")
    if kubectl is None:
        pytest.skip("kubectl not installed: the RENDERED otel-reference checks did NOT run")
    out = subprocess.run(
        [kubectl, "kustomize", str(path)], capture_output=True, text=True, check=True
    ).stdout
    return [d for d in yaml.safe_load_all(out) if d]


def _pods(docs: list[dict]) -> dict[str, dict]:
    return {f"{d['kind']}/{d['metadata']['name']}": d["spec"]["template"]["spec"]
            for d in docs if ((d.get("spec") or {}).get("template") or {}).get("spec")}


def _containers(docs: list[dict]) -> list[dict]:
    return [c for spec in _pods(docs).values()
            for c in spec.get("containers", []) + spec.get("initContainers", [])]


def _one(docs: list[dict], kind: str, name_prefix: str) -> dict:
    found = [d for d in docs if d["kind"] == kind and d["metadata"]["name"].startswith(name_prefix)]
    assert len(found) == 1, f"{kind}/{name_prefix}*: {[d['metadata']['name'] for d in found]}"
    return found[0]


def _env(container: dict) -> dict[str, str]:
    return {e["name"]: e.get("value") for e in container.get("env", [])}


def _configmap_file(docs: list[dict], prefix: str, key: str) -> str:
    return _one(docs, "ConfigMap", prefix)["data"][key]


# ── the default deployment ──────────────────────────────────────────────────────


class TestTheBaseIsTelemetryFree:
    def test_the_rendered_base_sets_no_telemetry_variable_and_names_no_collector(self):
        """M26: an endpoint or a switch in k8s/base would export from every deploy."""
        docs = _render(BASE)
        assert any(d["kind"] == "Deployment" and d["metadata"]["name"] == "millm-backend"
                   for d in docs), "the base render lost the backend: this test would see nothing"
        for c in _containers(docs):
            leaked = sorted(n for n in _env(c) if n.startswith(("TELEMETRY_", "OTEL_")))
            assert leaked == [], f"k8s/base sets {leaked} on {c['name']}: export is opt-in (14.0)"
        text = yaml.safe_dump_all(docs)
        for needle in ("otel-collector", NAMESPACE, ":4318"):
            assert needle not in text, f"k8s/base renders {needle!r}"

    def test_the_base_kustomization_references_no_otel_component(self):
        """No binary needed: the file ArgoCD's `millm` Application reads."""
        text = (BASE / "kustomization.yaml").read_text()
        assert "otel" not in text, "k8s/base/kustomization.yaml names an otel component"

    def test_the_millm_application_still_deploys_only_the_base(self):
        apps = [d for d in yaml.safe_load_all(MILLM_APP.read_text())
                if d and d.get("kind") == "Application"]
        assert [a["spec"]["source"]["path"] for a in apps] == ["k8s/base"]


# ── the reference receiver ──────────────────────────────────────────────────────


class TestTheReferenceOverlay:
    def test_it_renders_the_namespace_collector_backend_volume_and_dashboard(self):
        docs = _render(OVERLAY)
        kinds = {(d["kind"], d["metadata"]["name"].rsplit("-", 1)[0]
                  if d["kind"] == "ConfigMap" else d["metadata"]["name"]) for d in docs}
        for expected in [("Namespace", NAMESPACE), ("Deployment", "otel-collector"),
                         ("Service", "otel-collector"), ("Deployment", "otel-lgtm"),
                         ("Service", "otel-lgtm"), ("PersistentVolumeClaim", "otel-lgtm-data"),
                         ("PersistentVolume", "millm-otel-lgtm"),
                         ("ConfigMap", "otel-collector-config"),
                         ("ConfigMap", "otel-lgtm-retention"),
                         ("ConfigMap", "otel-lgtm-dashboards"), ("Ingress", "otel-lgtm-ingress")]:
            assert expected in kinds, f"{expected} missing from the overlay render"

    def test_everything_is_in_its_own_namespace_and_only_two_kinds_are_cluster_scoped(self):
        docs = _render(OVERLAY)
        for d in docs:
            if d["kind"] in CLUSTER_SCOPED:
                assert "namespace" not in d["metadata"], d["metadata"]
            else:
                assert d["metadata"].get("namespace") == NAMESPACE, (d["kind"], d["metadata"])
        assert {d["kind"] for d in docs if "namespace" not in d["metadata"]} == CLUSTER_SCOPED

    def test_the_volume_is_20gi_bound_by_name_retained_and_pinned_to_the_backend_node(self):
        docs = _render(OVERLAY)
        pv, pvc = _one(docs, "PersistentVolume", "millm-otel-lgtm"), \
            _one(docs, "PersistentVolumeClaim", "otel-lgtm-data")
        assert pv["spec"]["capacity"]["storage"] == "20Gi"
        assert pvc["spec"]["resources"]["requests"]["storage"] == "20Gi"
        assert pvc["spec"]["volumeName"] == pv["metadata"]["name"]
        assert pvc["spec"]["storageClassName"] == pv["spec"]["storageClassName"]
        assert pv["spec"]["persistentVolumeReclaimPolicy"] == "Retain"
        (term,) = pv["spec"]["nodeAffinity"]["required"]["nodeSelectorTerms"]
        (node,) = term["matchExpressions"][0]["values"]
        backend = _pods(docs)["Deployment/otel-lgtm"]
        assert backend["nodeSelector"] == {"kubernetes.io/hostname": node}
        claims = [v["persistentVolumeClaim"]["claimName"] for v in backend["volumes"]
                  if "persistentVolumeClaim" in v]
        assert claims == ["otel-lgtm-data"]

    def test_every_image_is_pinned_by_tag_and_digest(self):
        images = [c["image"] for c in _containers(_render(OVERLAY))]
        assert len(images) == 2, images
        for image in images:
            assert PINNED.match(image), f"{image} is not pinned by tag AND digest"

    def test_nothing_requests_a_gpu_and_every_container_hides_the_cards(self):
        docs = _render(OVERLAY)
        assert "nvidia.com/gpu" not in yaml.safe_dump_all(docs)
        for c in _containers(docs):
            assert _env(c).get("NVIDIA_VISIBLE_DEVICES") == "void", (
                f"{c['name']}: the node's default runtime is nvidia; unset means every card"
            )

    def test_retention_is_seven_days_in_all_three_stores_and_the_files_are_mounted(self):
        docs = _render(OVERLAY)
        loki = yaml.safe_load(_configmap_file(docs, "otel-lgtm-retention", "loki-config.yaml"))
        assert loki["compactor"]["retention_enabled"] is True
        assert loki["compactor"]["delete_request_store"] == "filesystem"
        assert loki["limits_config"]["retention_period"] == "168h"
        tempo = yaml.safe_load(_configmap_file(docs, "otel-lgtm-retention", "tempo-config.yaml"))
        assert tempo["backend_worker"]["compaction"]["block_retention"] == "168h"
        assert (tempo["backend_scheduler"]["provider"]["compaction"]["compaction"]
                ["block_retention"] == "168h")
        (lgtm,) = _pods(docs)["Deployment/otel-lgtm"]["containers"]
        args = _env(lgtm)["PROMETHEUS_EXTRA_ARGS"].split()
        assert "--storage.tsdb.retention.time=7d" in args
        assert any(a.startswith("--storage.tsdb.retention.size=") for a in args)
        # The files above do nothing unless they replace the image's own configuration.
        mounts = {m["mountPath"]: m.get("subPath") for m in lgtm["volumeMounts"]}
        assert mounts.get("/otel-lgtm/loki-config.yaml") == "loki-config.yaml"
        assert mounts.get("/otel-lgtm/tempo-config.yaml") == "tempo-config.yaml"

    def test_no_public_host_and_the_collector_is_cluster_internal(self):
        docs = _render(OVERLAY)
        ingresses = [d for d in docs if d["kind"] == "Ingress"]
        assert len(ingresses) == 1
        for rule in ingresses[0]["spec"]["rules"]:
            assert re.fullmatch(r"k8s-[a-z0-9-]+\.hitsai\.local", rule["host"]), rule["host"]
            for path in rule["http"]["paths"]:
                assert path["backend"]["service"]["name"] == "otel-lgtm"
                assert path["backend"]["service"]["port"]["number"] == 3000
        for svc in (d for d in docs if d["kind"] == "Service"):
            assert svc["spec"].get("type", "ClusterIP") == "ClusterIP", svc["metadata"]["name"]
            assert not any("nodePort" in p for p in svc["spec"]["ports"])

    def test_the_collector_mounts_the_one_config_file(self):
        docs = _render(OVERLAY)
        (collector,) = _pods(docs)["Deployment/otel-collector"]["containers"]
        assert "--config=/etc/otelcol/collector-config.yaml" in collector["args"]
        config = yaml.safe_load(_configmap_file(docs, "otel-collector-config",
                                                "collector-config.yaml"))
        for name, pipeline in config["service"]["pipelines"].items():
            assert pipeline["processors"][0] == "memory_limiter", name
            assert "batch" in pipeline["processors"], name
            assert pipeline["receivers"] == ["otlp"] and pipeline["exporters"] == ["otlp_http"]
        assert set(config["receivers"]["otlp"]["protocols"]) == {"http"}
        assert config["exporters"]["otlp_http"]["endpoint"] == "http://otel-lgtm:4318"


# ── nothing leaves the cluster (review R1 of S3) ─────────────────────────────────

#: Every phone-home of the bundled components, and the login paths to the image's admin/admin.
#: Values verified against the running 0.38.0 image (Loki `/config`, Tempo `/status/config`,
#: Pyroscope `/api/v1/status/config`, Grafana `/api/admin/settings`; review R1 of S3).
QUIET_BACKEND = {
    "GF_AUTH_DISABLE_LOGIN_FORM": "true",
    "GF_AUTH_BASIC_ENABLED": "false",
    "GF_ANALYTICS_REPORTING_ENABLED": "false",
    "GF_ANALYTICS_CHECK_FOR_UPDATES": "false",
    "GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES": "false",
    "GF_PLUGINS_PREINSTALL_DISABLED": "true",
    "GF_NEWS_NEWS_FEED_ENABLED": "false",
    "LOKI_EXTRA_ARGS": "-reporting.enabled=false",
    "TEMPO_EXTRA_ARGS": "-reporting.enabled=false",
    "PYROSCOPE_EXTRA_ARGS": "-usage-stats.enabled=false",
}


def _selects(policy: dict, labels: dict[str, str]) -> bool:
    match = policy["spec"]["podSelector"].get("matchLabels", {})
    return bool(match) and all(labels.get(k) == v for k, v in match.items())


def _policies_for(docs: list[dict], deployment: str) -> list[dict]:
    labels = _one(docs, "Deployment", deployment)["spec"]["template"]["metadata"]["labels"]
    return [d for d in docs if d["kind"] == "NetworkPolicy" and _selects(d, labels)]


class TestNothingLeavesTheCluster:
    def test_the_backend_reports_nothing_fetches_nothing_and_has_no_login(self):
        """Grafana, Loki, Tempo and Pyroscope default to usage reporting to stats.grafana.org;
        the image's start script preinstalls a plugin from grafana.com; admin/admin logs in."""
        (lgtm,) = _pods(_render(OVERLAY))["Deployment/otel-lgtm"]["containers"]
        env = _env(lgtm)
        for name, value in QUIET_BACKEND.items():
            assert env.get(name) == value, f"otel-lgtm {name}={env.get(name)!r}, want {value!r}"

    def test_compose_runs_the_backend_with_the_same_settings(self):
        service = yaml.safe_load(COMPOSE.read_text())["services"]["otel-lgtm"]
        env = dict(e.split("=", 1) for e in service["environment"])
        for name, value in QUIET_BACKEND.items():
            assert env.get(name) == value, f"compose otel-lgtm {name}={env.get(name)!r}"

    def test_the_backend_pod_has_no_egress_at_all(self):
        docs = _render(OVERLAY)
        policies = _policies_for(docs, "otel-lgtm")
        assert policies, "no NetworkPolicy selects the otel-lgtm pod: its egress is open"
        assert any("Egress" in p["spec"]["policyTypes"] for p in policies)
        for p in policies:
            assert not p["spec"].get("egress"), (
                f"{p['metadata']['name']} lets otel-lgtm send out: everything in it is loopback"
            )

    def test_the_collector_reaches_only_the_backend_otlp_port_and_dns(self):
        docs = _render(OVERLAY)
        policies = _policies_for(docs, "otel-collector")
        assert policies and all("Egress" in p["spec"]["policyTypes"] for p in policies), (
            "no egress policy selects the otel-collector pod"
        )
        backend_labels = _one(docs, "Deployment", "otel-lgtm")["spec"]["template"]["metadata"][
            "labels"]
        rules = [r for p in policies for r in p["spec"].get("egress", [])]
        to_backend = to_dns = 0
        for rule in rules:
            (peer,) = rule["to"]
            ports = {(p["protocol"], p["port"]) for p in rule["ports"]}
            if "namespaceSelector" in peer:
                assert peer["namespaceSelector"]["matchLabels"] == {
                    "kubernetes.io/metadata.name": "kube-system"}
                assert peer["podSelector"]["matchLabels"] == {"k8s-app": "kube-dns"}
                assert ports == {("UDP", 53), ("TCP", 53)}
                to_dns += 1
            else:
                assert set(peer) == {"podSelector"}, peer
                assert _selects({"spec": peer}, backend_labels), peer
                assert ports == {("TCP", 4318)}, ports
                to_backend += 1
        assert (to_backend, to_dns) == (1, 1), rules

    def test_every_collector_endpoint_is_the_backend_or_a_listener(self):
        config = yaml.safe_load((COMPONENT / "collector-config.yaml").read_text())
        found: list[str] = []

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "endpoint" and isinstance(value, str):
                        found.append(value)
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(config)
        assert found
        for endpoint in found:
            assert endpoint.startswith(("0.0.0.0:", "http://otel-lgtm:4318")), endpoint


# ── the switch-on component (14.0) ──────────────────────────────────────────────


class TestTheExportComponent:
    def test_the_render_turns_three_switches_on_at_the_reference_collector_stateless_off(self):
        docs, reference = _render(EXPORT_OVERLAY), _render(OVERLAY)
        backend = next(c for c in _pods(docs)["Deployment/millm-backend"]["containers"]
                       if c["name"] == "backend")
        env = _env(backend)
        for switch in ("TRACES", "METRICS", "LOGS"):
            assert env[f"TELEMETRY_{switch}_ENABLED"] == "true"
        assert env["TELEMETRY_STATELESS_EXPORT_PERMITTED"] == "false"
        assert env["TELEMETRY_ALLOW_PLAINTEXT"] == "true"
        svc = _one(reference, "Service", "otel-collector")
        (port,) = [p["port"] for p in svc["spec"]["ports"] if p["name"] == "otlp-http"]
        assert env["TELEMETRY_ENDPOINT"] == (
            f"http://{svc['metadata']['name']}.{svc['metadata']['namespace']}.svc:{port}"
        ), "the switch-on must point at the reference Collector's own Service"
        assert "TELEMETRY_HEADERS" not in env, "credentials belong in a Secret, never in git"

    def test_the_component_changes_nothing_but_those_variables(self):
        base, docs = _render(BASE), _render(EXPORT_OVERLAY)
        for d in docs:
            for c in _pods([d]).get(f"{d['kind']}/{d['metadata']['name']}", {}).get(
                    "containers", []):
                if "env" in c:
                    c["env"] = [e for e in c["env"] if not e["name"].startswith("TELEMETRY_")]
        assert docs == base

    def test_the_component_is_not_referenced_by_the_base_yet(self):
        component = yaml.safe_load((EXPORT_COMPONENT / "kustomization.yaml").read_text())
        assert component["kind"] == "Component"
        assert "components" not in yaml.safe_load((BASE / "kustomization.yaml").read_text())


# ── ArgoCD: its own project, applied by hand ────────────────────────────────────


class TestTheApplication:
    def _docs(self) -> dict[str, dict]:
        return {d["kind"]: d for d in yaml.safe_load_all(APP.read_text()) if d}

    def test_the_application_deploys_the_overlay_into_its_own_namespace(self):
        app = self._docs()["Application"]
        assert app["spec"]["source"]["path"] == "k8s/overlays/otel-reference"
        assert app["spec"]["destination"]["namespace"] == NAMESPACE
        assert app["spec"]["project"] == NAMESPACE
        assert (ROOT / app["spec"]["source"]["path"] / "kustomization.yaml").is_file()

    def test_the_project_reaches_only_its_namespace_and_two_cluster_kinds(self):
        project = self._docs()["AppProject"]
        assert [d["namespace"] for d in project["spec"]["destinations"]] == [NAMESPACE]
        assert {r["kind"] for r in project["spec"]["clusterResourceWhitelist"]} == CLUSTER_SCOPED

    def test_the_header_says_it_is_applied_once_by_hand(self):
        head = APP.read_text().split("---", 1)[0]
        assert "Apply once by hand" in head


# ── Compose shares the files ────────────────────────────────────────────────────


class TestComposeSharesTheFiles:
    def _services(self) -> dict:
        return yaml.safe_load(COMPOSE.read_text())["services"]

    def test_compose_mounts_the_same_collector_config_retention_and_dashboard(self):
        mounts = {v.split(":")[0]: v.split(":")[1] for s in self._services().values()
                  for v in s.get("volumes", []) if v.startswith("./")}
        component = yaml.safe_load((COMPONENT / "kustomization.yaml").read_text())
        generated = {f for g in component["configMapGenerator"] for f in g["files"]}
        expected = {f"./k8s/components/otel-reference/{f}" for f in generated}
        assert set(mounts) == expected, "Compose and Kubernetes must mount the same files"
        for source in mounts:
            assert (ROOT / source).is_file(), source

    def test_compose_runs_the_same_pinned_images(self):
        compose = sorted(s["image"] for s in self._services().values())
        kube = sorted(c["image"] for c in _containers(_render(OVERLAY)))
        assert compose == kube

    def test_compose_publishes_only_on_loopback_and_requests_no_gpu(self):
        for name, service in self._services().items():
            for port in service.get("ports", []):
                assert str(port).startswith("127.0.0.1:"), (name, port)
            assert "deploy" not in service, f"{name} reserves devices"


# ── the dashboard ───────────────────────────────────────────────────────────────


def _dashboard_queries() -> list[tuple[str, str]]:
    dash = json.loads((COMPONENT / "dashboard.json").read_text())
    return [(p["datasource"]["uid"], t["expr"]) for p in dash["panels"]
            for t in p.get("targets", [])]


class TestTheDashboard:
    def test_it_is_provisioned_from_the_path_it_is_mounted_at(self):
        docs = _render(OVERLAY)
        (lgtm,) = _pods(docs)["Deployment/otel-lgtm"]["containers"]
        mounts = {m["subPath"]: m["mountPath"] for m in lgtm["volumeMounts"] if m.get("subPath")}
        assert mounts["dashboard-provider.yaml"].startswith(
            "/otel-lgtm/grafana/conf/provisioning/dashboards/")
        provider = yaml.safe_load(_configmap_file(docs, "otel-lgtm-dashboards",
                                                  "dashboard-provider.yaml"))
        (entry,) = provider["providers"]
        assert entry["options"]["path"] == mounts["dashboard.json"]
        json.loads(_configmap_file(docs, "otel-lgtm-dashboards", "dashboard.json"))

    def test_it_has_the_four_rows_fr_38_25_names(self):
        dash = json.loads((COMPONENT / "dashboard.json").read_text())
        rows = " ".join(p["title"].lower() for p in dash["panels"] if p["type"] == "row")
        for words in ("coverage", "crossings by decision revision", "skips and failures",
                      "export health"):
            assert words in rows, words

    def test_its_metric_queries_name_only_registered_series_and_dimensions(self):
        from millm.telemetry import allowlist, counters
        from tests.support.telemetry_signal_doc import (
            HISTOGRAM_DIMS,
            live_instruments,
            prometheus_names,
        )

        series: dict[str, set[str]] = {}
        for name, kind, unit in live_instruments():
            dims = counters.INSTRUMENTS[name][2] if name in counters.INSTRUMENTS else \
                HISTOGRAM_DIMS.get(name, ())
            for prom in prometheus_names(name, kind, unit):
                series[prom] = set(dims) | {"le"}
        resource = {k.replace(".", "_") for k in allowlist.RESOURCE}
        queries = [q for uid, q in _dashboard_queries() if uid == "prometheus"]
        named = set()
        for query in queries:
            for metric, matcher in re.findall(r"\b(millm_[a-z_]+)(\{[^}]*\})?", query):
                assert metric in series, f"{metric} is not a series of any live instrument"
                named.add(metric)
                for label in re.findall(r"([a-z_]+)\s*(?:=~|!=|=)", matcher or ""):
                    assert label in series[metric] | resource, (metric, label)
            if "millm_" in query:
                for group in re.findall(r"by \(([^)]*)\)", query):
                    for label in (x.strip() for x in group.split(",")):
                        assert any(label in series[m] for m in
                                   re.findall(r"\b(millm_[a-z_]+)", query)), (label, query)
        assert {"millm_probe_evaluations_total", "millm_probe_evaluations_expected_total",
                "millm_probe_threshold_crossings_total"} <= named

    def test_its_log_queries_name_only_attributes_the_allowlist_lets_out(self):
        from millm.telemetry import allowlist

        allowed = {k.replace(".", "_") for k in (*allowlist.LOG_RESULT, *allowlist.LOG_LIFECYCLE,
                                                 *allowlist.RESOURCE)}
        queries = [q for uid, q in _dashboard_queries() if uid == "loki"]
        assert queries, "no Loki query: the crossings-by-revision panel is gone"
        for query in queries:
            for attr in re.findall(r"\b(mi_(?:probe|lifecycle)_[a-z_]+)", query):
                assert attr in allowed, f"{attr} is not an exported record attribute"
            for attr in re.findall(r"\b([a-z_]+)=\"", query):
                assert attr in allowed | {"service_name"}, attr
