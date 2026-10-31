---
sidebar_position: 3
title: "Scripting Experiments in Python"
---

# Tutorial: Scripting Experiments in Python

Steering experiments get interesting when they're systematic: sweeps over strengths, A/B comparisons across prompts, batches of features. This tutorial builds a small experiment harness using the OpenAI SDK for inference and plain `requests` for the management API.

## Setup

```bash
pip install openai requests
```

```python
import requests
from openai import OpenAI

MILLM = "http://localhost:8000"
client = OpenAI(base_url=f"{MILLM}/v1", api_key="unused")  # key is not checked

def mgmt(method: str, path: str, **json_body):
    """Call the management API and unwrap the {success, data, error} envelope."""
    r = requests.request(method, f"{MILLM}/api{path}", json=json_body or None)
    payload = r.json()
    if not payload.get("success"):
        raise RuntimeError(f"{path}: {payload.get('error')}")
    return payload.get("data")
```

The `/api/*` endpoints all return the same envelope; raising on `success: false` up front keeps the rest of the code clean. (Error codes are documented in the [error reference](/reference/error-codes).)

## Building block: steered vs. unsteered

```python
def generate(prompt: str, max_tokens: int = 120, **extra) -> str:
    resp = client.chat.completions.create(
        model="gemma-2-2b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,          # deterministic → differences are attributable
        max_tokens=max_tokens,
        **extra,
    )
    return resp.choices[0].message.content

def set_steering(feature: int, value: float):
    mgmt("POST", "/saes/steering", feature_idx=feature, value=value)

def clear_steering():
    requests.delete(f"{MILLM}/api/saes/steering")

# A/B a single feature
prompt = "Describe your ideal weekend."
baseline = generate(prompt)
set_steering(12082, 40)
steered = generate(prompt)
clear_steering()

print("BASELINE:", baseline[:200])
print("STEERED :", steered[:200])
```

### Verify the intervention actually happened

Never trust an experiment you didn't verify. `steering_apply_count` increments on every forward pass where the delta was applied:

```python
def apply_count() -> int:
    return mgmt("GET", "/saes/attachment")["steering_apply_count"]

before = apply_count()
steered = generate(prompt)
assert apply_count() > before, "steering hook did not fire!"
```

## Strength sweep

```python
def sweep(feature: int, strengths: list[float], prompt: str) -> dict[float, str]:
    results = {}
    for s in strengths:
        set_steering(feature, s)
        results[s] = generate(prompt)
    clear_steering()
    return results

for s, text in sweep(12082, [0, 10, 40, 80, 150], prompt).items():
    print(f"\n=== strength {s} ===\n{text[:160]}")
```

## Cleaner A/B: per-request profiles

Mutating global steering between calls works, but **per-request profiles** avoid touching shared state — important if anything else is using the server:

```python
# One-time setup: save the conditions you want to compare
mgmt("POST", "/profiles", name="cond-dogs-40",
     steering={"12082": 40.0}, description="dogs @ 40")
mgmt("POST", "/profiles", name="cond-clean", steering={},
     description="explicit no-steering condition")

# Then alternate conditions per request — no global state changes:
a = generate(prompt, extra_body={"profile": "cond-dogs-40"})
b = generate(prompt, extra_body={"profile": "cond-clean"})
```

The `extra_body` hook is how the OpenAI SDK passes non-standard parameters. Failure modes are loud by design: an unknown profile name is a `404`, and a profile whose features don't fit the attached SAE is a `400` — a request is never silently served under the wrong condition. Per-request-profile calls always run on the serial backend.

## Reading the monitoring data

```python
mgmt("POST", "/monitoring/configure", enabled=True, top_k=10)

generate("Tell me about sailing across the Pacific.")

history = mgmt("GET", "/monitoring/history")
latest = history["records"][0]
print("request:", latest["request_id"])
for f in latest["top_k"]:
    print(f"  feature {f['feature_index']:>6}  activation {f['activation']:.2f}")
```

Each record reflects the final forward pass of a generation (see [monitoring semantics](/concepts/monitoring)); the per-feature statistics endpoints aggregate across requests:

```python
top = mgmt("POST", "/monitoring/statistics/top", k=10, metric="mean")
```

## Putting it together

A minimal publishable-quality experiment loop — N prompts × M conditions with verification:

```python
prompts = ["Describe your ideal weekend.",
           "What should I cook tonight?",
           "Write a short story opening."]
conditions = {"clean": "cond-clean", "dogs40": "cond-dogs-40"}

results = []
for p in prompts:
    for label, prof in conditions.items():
        before = apply_count()
        text = generate(p, extra_body={"profile": prof})
        fired = apply_count() > before
        results.append({"prompt": p, "condition": label,
                        "steered_pass_ran": fired, "output": text})
```

From here it's your analysis: keyword frequency, embedding distance between conditions (note `/v1/embeddings` is always **unsteered**, deliberately — so it's a neutral measuring stick), or human rating.

## See also

- [API Reference](/api/overview) — every endpoint used above, with full schemas
- [Profiles](/features/profiles) — export/import for sharing conditions across instances
- [Concepts: Steering](/concepts/steering) — calibration guidance for choosing strengths
