---
sidebar_position: 10
title: Telemetry Export
description: "Send probe results, spans and coverage counters to an OpenTelemetry receiver: what leaves, how to switch it on, the reference receiver, queries and troubleshooting"
---

# Telemetry Export (OpenTelemetry)

miLLM can send what its [probe monitors](./probe-monitors.md) find to an OpenTelemetry (OTel)
receiver over OTLP/HTTP (the OpenTelemetry Protocol): one **log record per probe result**, one
**span per request**, and **coverage counters**. It is **off by default**, and nothing leaves
until an operator switches a signal on and names the receiver.

This page is the operator guide: what leaves and what never does, how to switch it on, the
reference receiver that ships with miLLM, the signal schemas, queries, and troubleshooting.

## What leaves, and what never does

Every attribute passes an **allowlist** before it is queued. An attribute that is not listed for
its signal, or whose value does not match its listed pattern, is dropped and counted
(`attributes_dropped` in health); it is never queued and never sent. No listed attribute can hold
free text: every string is an enumeration or an identifier pattern.

**Never sent:** prompts, replies, decoded context windows, token ids, activations, probe weights,
API keys and other credentials, exception messages (only an error's class name), and the free-text
`reason` of an arm or recalibrate call.

**Sent, when its switch is on:**

| Signal | What | Switch |
|---|---|---|
| Logs | one `mi.probe.result` record per recorded probe result (the [result contract](./probe-monitors.md#the-result-contract), flattened); one `mi.probe.lifecycle` record per import, arm, refused arm, disarm, parity run, recalibrate and delete | `TELEMETRY_LOGS_ENABLED` |
| Traces | one span per inference request (`chat`, `text_completion`, `millm.probe_score`) with two children: the admission wait and the probe evaluation | `TELEMETRY_TRACES_ENABLED` |
| Metrics | the probe coverage counters (also on the Prometheus route, export on or off) and two histograms | `TELEMETRY_METRICS_ENABLED` |

Three details that are easy to get wrong when reading the data:

- **Result records carry no sampled flag.** Every recorded live and batch result is sent whatever
  the trace sampler decided (`TELEMETRY_TRACES_SAMPLER_RATIO` samples spans, not results). A
  record's trace flags never claim a sampled trace, because a result cannot know whether its trace
  was kept; follow its `trace_id` to the trace when the trace exists.
- **`gen_ai.request.model` is sent only when it names the loaded model.** The request's model
  field is client text; when it is anything other than the resident model's own name, the attribute
  is omitted, never sent.
- **Stateless scoring results stay local** unless both the operator sets
  `TELEMETRY_STATELESS_EXPORT_PERMITTED=true` and the `POST /api/probes/score` request sends
  `export_results: true`. The response's `export` summary says which of the two held.

## Switching it on

Install and check miLLM with export **off** first; then switch on. The three switches are
independent. Set them in the deployment's environment (read at startup), or, since 2026-10-29,
**while miLLM runs on the Settings page** (below).

| Setting | Default | Meaning |
|---|---|---|
| `TELEMETRY_TRACES_ENABLED`, `TELEMETRY_METRICS_ENABLED`, `TELEMETRY_LOGS_ENABLED` | `false` | The three switches |
| `TELEMETRY_ENDPOINT` | — | The receiver's base URL; miLLM appends `/v1/traces`, `/v1/metrics`, `/v1/logs` to its path |
| `TELEMETRY_HEADERS` | — | `key=value,key=value` request headers: **where credentials go** |
| `TELEMETRY_ALLOW_PLAINTEXT` | `false` | Required for an `http://` endpoint: a recorded choice that the path is trusted and local |
| `TELEMETRY_TLS_CA_FILE`, `TELEMETRY_TLS_CLIENT_CERT_FILE`, `TELEMETRY_TLS_CLIENT_KEY_FILE` | — | A private CA, and a client certificate for mutual TLS (both files, or neither) |
| `TELEMETRY_STATELESS_EXPORT_PERMITTED` | `false` | Allows stateless scoring results to leave when a request asks |

The full list, with the queue, batch, retry and health settings, is in the
[configuration reference](../reference/configuration.md#telemetry-export).

### Changing export while miLLM runs

Open **Settings → Environment → Changeable while miLLM runs → Telemetry and logging**. Every
telemetry setting is a card there — the three switches, the endpoint, plaintext, the sampler, the
queue, batch and timing values, the result-log origins and the stateless permission — with when a
change takes effect, the value in force, the deployment's value and **Use the environment value**.

- **A save drains and restarts the export pipeline.** Records already queued are sent for up to
  `TELEMETRY_DRAIN_DEADLINE_S`, then the pipeline is built again from the values in force. A save
  of several settings at once (`PUT /api/environment`) restarts it once.
- **A saved value outlives a restart.** It is stored in miLLM's database and applied before
  telemetry starts; **Use the environment value** returns to the deployment's value.
- **What stays in the environment:** `TELEMETRY_HEADERS` and `TELEMETRY_TLS_CLIENT_KEY_FILE`
  (secrets), `TELEMETRY_TLS_CA_FILE` and `TELEMETRY_TLS_CLIENT_CERT_FILE` (a file has to be mounted
  into the pod first), and `OTEL_SDK_DISABLED`. The page refuses them.
- **An endpoint with a user name, password, query string or fragment is refused** when you save it.
  Put a receiver's token in `TELEMETRY_HEADERS`.
- **A refused configuration** (no endpoint, `http://` without plaintext allowed) is saved and shown
  on the Dashboard's Telemetry card and in `GET /api/health/detailed`; nothing is sent.
- **An AI agent's change waits for you.** An agent (the `millm_set_telemetry` tool, or any request
  carrying `X-miLLM-Agent`) cannot change a telemetry setting itself: the request waits on the
  [Approvals](./approvals.md) page as *Change telemetry settings in miLLM*, showing each setting's
  value now and after approval, and runs once when you approve it.
- Off stays free: until a switch is on **and** an endpoint is set, no OpenTelemetry SDK module is
  loaded. After export has been on once, the modules stay loaded until a restart; they cost
  nothing per request while every switch is off.

:::warning Credentials go in headers, never in the URL
Put a receiver's token in `TELEMETRY_HEADERS` (for example `authorization=Bearer …`), never in
the endpoint's path or query. miLLM strips userinfo, query and fragment from the endpoint before it
shows it, but it **shows the path**: in `GET /api/health/detailed`, on the Settings page's
environment list and through the MCP server's status tool. A receiver that takes its token in the
path would display it there. Headers are reported only as set or not set.
:::

**Docker Compose.** `docker-compose.yml` passes the switches, endpoint, headers and plaintext
setting through from your shell or `.env`, defaulting to off. With the reference receiver beside
it:

```bash
TELEMETRY_TRACES_ENABLED=true TELEMETRY_METRICS_ENABLED=true TELEMETRY_LOGS_ENABLED=true \
TELEMETRY_ENDPOINT=http://otel-collector:4318 TELEMETRY_ALLOW_PLAINTEXT=true \
docker compose -f docker-compose.yml -f docker-compose.otel.yml up -d
```

**Kubernetes.** The switch-on is the kustomize component `k8s/components/otel-export`: the three
switches, the endpoint `http://otel-collector.millm-otel.svc:4318`,
`TELEMETRY_ALLOW_PLAINTEXT=true` (the path never leaves the cluster) and
`TELEMETRY_STATELESS_EXPORT_PERMITTED=false`. It is added to `k8s/base` in its own commit; its
file header has the steps. Until then the deployed pod exports nothing.

After switching on, check `GET /api/health/detailed`: each enabled signal should read `healthy`
within a minute of traffic.

## How miLLM sends

- **OTLP/HTTP with protobuf only** (`TELEMETRY_PROTOCOL=http/protobuf`; anything else is refused).
- **Its own HTTP client.** miLLM ignores proxy, CA-bundle and netrc settings from the environment,
  and never reads `OTEL_EXPORTER_OTLP_*` variables. Only the `TELEMETRY_*` settings apply. TLS
  certificates are verified against the system bundle, or `TELEMETRY_TLS_CA_FILE` when set.
- **Redirects are never followed.** A 3xx answer is a failure (`HTTP307` and so on). The records
  and the credential headers are never re-sent to the address a redirect names.
- **The response body is not read.** Only the status matters. A small body with a declared length
  (4 KiB or less) is read raw, without decompressing, so the connection can be reused; anything
  else closes the connection unread. A huge or compressed answer cannot hold the sender.
- **Timeouts.** `TELEMETRY_EXPORT_TIMEOUT_S` (10 s) bounds connecting and each read of the
  answer, not the whole exchange: a receiver that sends its status line a byte at a time can hold
  the sender longer. Meanwhile the queues fill, further records are dropped and counted, and the
  signal reads `degraded`; nothing in a request waits.
- **Retries.** Connection errors and HTTP 408, 429, 502, 503 and 504 are retried with backoff
  (0.5 s doubling, at most 5 s apart) until `TELEMETRY_RETRY_HORIZON_S` (60 s); then the batch
  is counted `failed`. A retry re-sends the same bytes, so a result keeps its `result_id`. Any other
  status fails at once; 401 and 403 also make the signal `unavailable`.
- **Never in the request path.** Records are queued without blocking (`TELEMETRY_QUEUE_SIZE`,
  8192 per signal) and sent by one background thread. A full queue drops the record and counts it.

## The reference receiver

miLLM ships a receiver to test against: an OTel Collector (core distribution) in front of
`grafana/otel-lgtm` (Grafana with Loki for logs, Tempo for traces and Prometheus for metrics, in
one container), one provisioned dashboard, and 7-day retention. It is a reference and test target,
**not a forensic store**.

| | Kubernetes | Docker Compose |
|---|---|---|
| Files | `k8s/components/otel-reference`, rendered by `k8s/overlays/otel-reference`, deployed by its own ArgoCD Application `k8s/argocd/millm-otel-app.yaml` (namespace `millm-otel`) | `docker-compose.otel.yml`, mounting the same Collector configuration, retention files and dashboard |
| Collector | `http://otel-collector.millm-otel.svc:4318`, in-cluster only (no Ingress) | `http://otel-collector:4318` on the Compose network; `localhost:4318` on the host |
| Query UI | `http://k8s-millm-otel.hitsai.local` (internal name only, no login) | `http://localhost:3000` (bound to 127.0.0.1) |
| Storage | one 20 Gi volume on the GPU node (`/data/millm-otel/lgtm`) | the named volume `millm-otel-lgtm-data` |

Neither requests a GPU. Retention: Loki 168 h, Tempo 168 h, Prometheus 7 days with a 6 GB cap.
The 20 Gi of the Kubernetes volume is a label, not a quota: a `hostPath` volume is bounded only by
the node's `/data` disk, so the retention settings are what bound it. The images are pinned by tag
and digest; bump both together.

**Nothing leaves the cluster, and there is no login.** The bundled Grafana, Loki, Tempo and
Pyroscope would by default report anonymous usage to `stats.grafana.org`, and Grafana would check
`grafana.com` for updates and download a plugin at every start; all of that is switched off, in
Kubernetes and Compose alike. In Kubernetes a NetworkPolicy also denies the backend pod all egress
and lets the Collector reach only the backend's OTLP port (and DNS). Grafana's login form and
basic authentication are off, so the image's built-in `admin`/`admin` account cannot be used;
every visitor is an anonymous Editor, who can query and edit dashboards but not change data
sources, plugins or users.

**Installing on Kubernetes** (once, by hand, after the push that carries the files has reached
`gitops/pilot`). Run from a miLLM checkout with a kubeconfig for the cluster (or copy the file to
the node and use `microk8s kubectl`):

```bash
kubectl kustomize k8s/overlays/otel-reference                 # inspect; applies nothing
kubectl apply -f k8s/argocd/millm-otel-app.yaml               # the AppProject and the Application
kubectl -n argocd get application millm-otel -w               # until Synced and Healthy
kubectl -n millm-otel get pods,svc,pvc,ingress,networkpolicy
```

The Query UI's name, `k8s-millm-otel.hitsai.local`, has no DNS record of its own: add it where
the other `k8s-*.hitsai.local` names are resolved (a hosts entry pointing at the ingress node,
`192.168.244.52`). Check the cluster side without DNS:
`curl -H 'Host: k8s-millm-otel.hitsai.local' http://192.168.244.52/api/health`.

**Removing it:** delete the Application first, then its project. The Application's finalizer
prunes everything it created (namespace, workloads, NetworkPolicies, the PersistentVolume object);
the data stays on the node.

```bash
kubectl -n argocd delete application millm-otel     # waits for the prune
kubectl -n argocd delete appproject millm-otel
# Optional, to wipe the stored telemetry (on mcs-lnxhost02): sudo rm -rf /data/millm-otel/lgtm
```

**The dashboard** ("miLLM probes (reference)") shows coverage (completed ÷ expected) by behaviour
and probe, threshold crossings by decision revision, skips and failures by reason, evaluation cost
and score margin, and export health: results expected against result records received, the
overflow counter, and the Collector's own accepted, refused and failed counts.

## Signal schemas

Generated from miLLM's live instrument registry and allowlist. A test fails when this block and
the code disagree.

<!-- signal-schema:begin (generated by tests/support/telemetry_signal_doc.py; do not edit) -->

Semantic-conventions schema: `https://opentelemetry.io/schemas/1.44.0`. Instrumentation scope: `millm`.

#### Resource attributes

`deployment.environment.name`, `service.instance.id`, `service.name`, `service.version`

#### Spans

| Span name | What it covers |
|---|---|
| `chat` | One per `/v1/chat/completions` request (streaming: until the stream ends). `ERROR` on an exception or a client disconnect. |
| `millm.probe.evaluate` | Child: probe evaluation at the end of generation. Its span id is every result's `span_id`. |
| `millm.probe_score` | One per `POST /api/probes/score` call. |
| `millm.queue.wait` | Child: the wait for an admission slot. |
| `text_completion` | One per `/v1/completions` request. |

Span attributes (each only where it applies): `error.type`, `gen_ai.operation.name`, `gen_ai.request.model`, `millm.batch.id`, `millm.engine`, `millm.execution_mode`, `millm.probe.results.completed`, `millm.probe.results.failed`, `millm.probe.results.interrupted`, `millm.probe.results.skipped`, `millm.queue.position_at_entry`, `millm.stream`

#### Metric instruments

| Instrument | Kind | Unit | Dimensions | Name in the reference backend |
|---|---|---|---|---|
| `millm.probe.evaluation.duration` | histogram | `ms` | `probe_id`, `window`, `engine` | `millm_probe_evaluation_duration_milliseconds_bucket`, `millm_probe_evaluation_duration_milliseconds_sum`, `millm_probe_evaluation_duration_milliseconds_count` |
| `millm.probe.evaluations` | counter | `1` | `probe_id`, `behaviour`, `kind`, `window`, `engine`, `execution_mode`, `origin`, `outcome`, `reason` | `millm_probe_evaluations_total` |
| `millm.probe.evaluations.expected` | counter | `1` | `probe_id`, `behaviour`, `kind`, `window`, `engine`, `execution_mode`, `origin` | `millm_probe_evaluations_expected_total` |
| `millm.probe.score_margin` | histogram (base-2 exponential) | `1` | `probe_id`, `window` | `millm_probe_score_margin` |
| `millm.probe.series_overflow` | counter | `1` | none | `millm_probe_series_overflow_total` |
| `millm.probe.stateless.evaluations` | counter | `1` | `probe_id`, `behaviour`, `kind`, `window`, `outcome`, `reason` | `millm_probe_stateless_evaluations_total` |
| `millm.probe.threshold_crossings` | counter | `1` | `probe_id`, `behaviour`, `kind`, `window`, `origin` | `millm_probe_threshold_crossings_total` |

A series beyond `TELEMETRY_SERIES_CAP` is folded into the one series `otel.metric.overflow=true` and counted in `millm.probe.series_overflow`.

#### Log records

Event names: `mi.probe.lifecycle`, `mi.probe.result`. Each is an OpenTelemetry log record with no body; its trace and span ids are on the record.

`mi.probe.result` attributes (the result contract, flattened):

`mi.probe.basis`, `mi.probe.batch_id`, `mi.probe.batch_line`, `mi.probe.behaviour.key`, `mi.probe.behaviour.version`, `mi.probe.choice_index`, `mi.probe.comparator`, `mi.probe.completed_at`, `mi.probe.count`, `mi.probe.decision.calibration_id`, `mi.probe.decision.provisional`, `mi.probe.decision.target_fpr`, `mi.probe.decision_digest`, `mi.probe.decision_revision`, `mi.probe.definition_schema`, `mi.probe.definition_scope`, `mi.probe.detector_digest`, `mi.probe.detector_digest_source`, `mi.probe.engine`, `mi.probe.evaluation_ms`, `mi.probe.event_time`, `mi.probe.event_time_source`, `mi.probe.execution_mode`, `mi.probe.fires`, `mi.probe.form`, `mi.probe.hook`, `mi.probe.identity_algorithm`, `mi.probe.input_index`, `mi.probe.kind`, `mi.probe.layer`, `mi.probe.migrated`, `mi.probe.model_load_id`, `mi.probe.model_revision`, `mi.probe.n_scored_tokens`, `mi.probe.origin`, `mi.probe.outcome`, `mi.probe.parity_status`, `mi.probe.persistence`, `mi.probe.precision`, `mi.probe.probe_id`, `mi.probe.probe_name`, `mi.probe.producer.name`, `mi.probe.producer.version`, `mi.probe.reason`, `mi.probe.request_id`, `mi.probe.result_id`, `mi.probe.rule`, `mi.probe.rule_set_digest`, `mi.probe.rule_set_id`, `mi.probe.rule_set_version`, `mi.probe.rung`, `mi.probe.rung_language`, `mi.probe.schema`, `mi.probe.score`, `mi.probe.score_semantics`, `mi.probe.scored`, `mi.probe.served_model`, `mi.probe.session_id`, `mi.probe.set.armed_members`, `mi.probe.set.complete`, `mi.probe.set.members_digest`, `mi.probe.set.total_members`, `mi.probe.signal`, `mi.probe.source_event_id`, `mi.probe.span_id`, `mi.probe.status`, `mi.probe.steering_ref.id`, `mi.probe.steering_ref.kind`, `mi.probe.target`, `mi.probe.template_sha256`, `mi.probe.threshold`, `mi.probe.threshold_revision`, `mi.probe.trace_id`, `mi.probe.turn_index`, `mi.probe.verdict`, `mi.probe.window`

`mi.probe.lifecycle` attributes:

`mi.lifecycle.action`, `mi.lifecycle.behaviour`, `mi.lifecycle.code`, `mi.lifecycle.decision_digest`, `mi.lifecycle.decision_revision`, `mi.lifecycle.detector_digest`, `mi.lifecycle.outcome`, `mi.lifecycle.previous_decision_digest`, `mi.lifecycle.previous_threshold_revision`, `mi.lifecycle.probe_id`, `mi.lifecycle.subject_kind`, `mi.lifecycle.threshold_revision`

<!-- signal-schema:end -->

In the reference backend, Loki stores each record attribute as structured metadata with dots
replaced by underscores (`mi.probe.result_id` → `mi_probe_result_id`), and Prometheus does the
same for metric names and labels.

## Queries

Verified against the reference backend on 2026-10-23 (records and series from a real chat with
two armed probes).

**Coverage by behaviour and probe** (Prometheus):

```promql
sum by (behaviour, probe_id) (rate(millm_probe_evaluations_total{service_name="millm", outcome="completed"}[5m]))
/
sum by (behaviour, probe_id) (rate(millm_probe_evaluations_expected_total{service_name="millm"}[5m]))
```

**Threshold crossings by decision revision** (Loki; revision is not a metric dimension):

```logql
sum by (mi_probe_behaviour_key, mi_probe_threshold_revision, mi_probe_decision_revision) (
  count_over_time({service_name="millm"} | mi_probe_schema="mi.probe.result/v1"
    | mi_probe_outcome="completed" | mi_probe_verdict="true" [1h]))
```

**Skips and failures by reason** (Prometheus):

```promql
sum by (outcome, reason) (increase(millm_probe_evaluations_total{service_name="millm", outcome=~"skipped|failed|interrupted"}[1h]))
```

**From a request to its results.** Every result carries the request's `trace_id` (and the
`span_id` of its `millm.probe.evaluate` span). Send a W3C `traceparent` header with the request,
or read `trace_id` from the response's `millm_probe_verdicts`, then:

```logql
{service_name="millm"} | trace_id="0af7651916cd43dd8448eb211c80319c"
```

finds the result and lifecycle records, and Tempo shows the trace (`chat` → `millm.queue.wait`,
`millm.probe.evaluate`). By request id instead: `{service_name="millm"} |
mi_probe_request_id="chatcmpl-…"`.

**Lifecycle history of one probe:**

```logql
{service_name="millm"} | mi_lifecycle_probe_id="pr_0e236ee1d124"
```

### Prometheus and OTLP: count once

The coverage counters exist twice: on miLLM's Prometheus route
(`GET /api/health/metrics/prometheus`, always) and as OTLP metrics (when metrics export is on),
**with the same names**. Feed one store from **one** of them, never both: a store that scrapes the
route and also receives OTLP metrics counts every result twice. In Kubernetes, either leave
`TELEMETRY_METRICS_ENABLED` off and scrape, or export and do not scrape miLLM's route into that
store.

## Troubleshooting

`GET /api/health/detailed` has a `telemetry` block: the effective configuration (no secrets), and
per signal its `state`, `exported`, `failed`, `dropped`, `last_success_at`, `last_error`
(`{type, at}`, the error's class only) and queue use; plus `series_overflow`,
`attributes_dropped` and the deletion note. The Admin UI dashboard shows the same as a card.

| State | Means |
|---|---|
| `disabled` | The switch is off (or `OTEL_SDK_DISABLED=true`) |
| `healthy` | Exporting, no recent problem |
| `degraded` | A failure not yet followed by a success, a drop within `TELEMETRY_HEALTH_DEGRADED_WINDOW_S` (60 s), or a queue at least half full |
| `unavailable` | The configuration was refused, the receiver answered 401 or 403, or nothing has succeeded for `TELEMETRY_HEALTH_UNAVAILABLE_AFTER_S` (300 s) |

| `last_error.type` or refusal | Look at |
|---|---|
| `NoEndpoint`, `InvalidEndpoint`, `UnsupportedProtocol`, `InvalidHeaders` | The setting it names; the configuration is refused as a whole and nothing is sent |
| `PlaintextNotAllowed` | An `http://` endpoint without `TELEMETRY_ALLOW_PLAINTEXT=true` |
| `SSLError` | The receiver's certificate does not verify: set `TELEMETRY_TLS_CA_FILE` to its CA. For mutual TLS set both client files |
| `ConnectionError`, `ConnectTimeout`, `ReadTimeout` | The receiver is down or unreachable; retried until the horizon |
| `HTTP401`, `HTTP403` | The credential in `TELEMETRY_HEADERS` |
| `HTTP3xx` | The endpoint redirects; point it at the final address |
| `HTTP429`, `HTTP503` | The receiver is shedding load (the reference Collector's memory limiter does this); retried |

`dropped` counts records that never got a place in a queue (full, or still queued at shutdown);
`failed` counts records the receiver did not accept within the retry horizon. A rising
`attributes_dropped` means something offered an attribute the allowlist refused; the record still
left without it. A rising `series_overflow` means `TELEMETRY_SERIES_CAP` (2000 series per
instrument) was reached, or a counted probe was not armed; those measurements went to the one
overflow series.

## What a restart can lose

Export is best effort. There is no outbox and no replay:

- records still queued when the drain deadline (`TELEMETRY_DRAIN_DEADLINE_S`, 5 s) passes at
  shutdown are counted `dropped` and lost;
- when traces or logs are still sending at that deadline, up to one metric interval
  (`TELEMETRY_METRIC_INTERVAL_S`, 15 s) of counter increments is not exported;
- a receiver outage longer than the retry horizon loses the batches it covered (counted `failed`).

The counters die with the process, so whatever shutdown loses is also written to the server log, one
warning per signal, with counts only:
`telemetry_shutdown_lost signal=logs dropped=3 failed=2`. A clean shutdown writes no such line.

miLLM's own record is unaffected: results are stored in `probe_events` as before, whether or not
they were exported, and the coverage counters restart from zero with the process.

## Retention and deletion

Deleting probes or events in miLLM **does not erase copies already exported**. Retention of
exported data belongs to the receiver: the reference backend keeps 7 days. A result record says
whether miLLM stored the result (`persistence`: `persisted`, `not_persisted` or `unknown`); the
exported copy is never the authoritative one.

## Parity tolerances (three, not one)

Telemetry changes none of them, but results report a probe's parity status, so the three are
listed here: `PROBE_PARITY_TOLERANCE` (0.001) is how closely a build is expected to track miStudio;
`PROBE_PARITY_SCORE_TOLERANCE` (0.10) is the arming gate's floor on the combined score across
precisions; `PROBE_PARITY_MATCHED_DTYPE_FLOOR` (0.10, or `PROBE_PARITY_MATCHED_RELATIVE_FLOOR`
× the probe's threshold when larger) is the floor when the definition's precision equals the served
one.
