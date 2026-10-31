---
sidebar_position: 13
---

# Approvals

When an AI agent asks miLLM to do something that changes what it serves, the request waits for
you. Nothing happens until you approve it. This page explains where those requests appear, what
approving does, and what History keeps.

An agent is any client that sends the `X-miLLM-Agent` request header. miStudio's Model Context
Protocol (MCP) server sends it on every request it makes to miLLM, so a request an agent makes
through miStudio's `millm_*` tools waits **here, in miLLM**, not in miStudio. A client that does not
send the header is treated as you, the operator, and runs at once, exactly as before.

## What waits for you

| Action | What it does |
|---|---|
| Load a model in miLLM | Loads a model onto a graphics card. The model being served now is unloaded first |
| Unload a model in miLLM | Unloads the served model, with any attached sparse autoencoder (SAE) and active steering |
| Delete a model from miLLM | Deletes a downloaded model from disk. It cannot be recovered |
| Change a model's probe limits in miLLM | Sets how many probes and behaviours may be armed on a model, and the time each pass may spend on them |
| Measure a model's probe capacity in miLLM | Runs the resident model on its graphics card to measure the step time and the cost of each probe |
| Import a rule set with operator code into miLLM | Imports a rule set that names operator code. The card shows every code document in full. A rule set without code never asks |
| Change telemetry settings in miLLM | Changes the OpenTelemetry export settings (switches, endpoint, sampling, queue and timing). The card shows each setting's value now and after approval, the endpoint without credentials; approving drains and restarts the export pipeline. An agent's change of any other setting does not ask (since 2026-10-29) |

Downloads, reads, scoring and everything else an agent does run at once.

## The banner and the Approvals page

While any request waits, a banner sits at the top of every page, below the header. It shows up to
three requests, the ones that expire soonest first, each with its **Approve** and **Reject**
buttons, and a line "and N more waiting" when there are more. **Collapse** shrinks it to its
heading line for this browser tab; a new request opens it again.

The **Approvals** menu entry carries the number waiting. Its page has two sections:

- **Waiting for you**: every waiting request, the soonest to expire first.
- **History**: every decided request, the most recent decision first, with the outcome, what was
  asked, who asked and when, who decided and when, the reason, and the result. **Show older** loads
  200 more.

Each card shows the action, the agent that asked, when it asked and when the request expires, the
facts of the request, every argument the agent sent (under **All arguments the agent sent**), and
the request id. A lease the agent sent with a load or an unload is never shown: miLLM keeps it
hidden, checks it again when the request runs, and forgets it once the request is decided.

## Approving and rejecting

**Approve** carries the request out once, exactly as shown. The outcome appears in History:
**Approved and run** with the result, or **Approved, then failed** with the error, for example
`MODEL_LEASED` when another tool leased the model before you approved. A long request, such as a
capacity measurement, shows **Running** until it finishes.

**Approve all settings changes (N)** appears in the banner and above the waiting list when two
or more waiting requests change a setting (a model's probe limits, or telemetry settings): it lists
every request it will approve, with what changes and who asked, and one confirmation approves them
all; each still runs once and appears in History on its own row, marked with the shared batch.
**Approve all code rules (N)** does the same for two or more waiting imports of operator code, as a
separate button and batch (code rules and settings are never approved together); its confirmation
shows each rule set, the first 12 characters of each code document's hash, and the full code.

**Reject** opens a reason field. The reason is optional; left blank, History records "Rejected by
the operator." A rejected request never runs.

A request nobody decides expires after 24 hours (the `APPROVAL_TTL_HOURS` setting, changeable on
the Settings page) and can never run; History shows it as **Expired**, decided by nobody. The agent
must ask again.

An agent can never approve or reject: miLLM refuses it (`approval_operator_only`). If miLLM restarts
while an approved request is running, that request is recorded as failed ("interrupted by a
restart; ask again").

## For agents

An agent's gated call returns `202` with the header `X-miLLM-Approval: pending` and the request's
`approval_id`. The agent polls `millm_get_approval_status` (or `GET /api/approvals/{approval_id}`)
and never sends the request again. The full contract is in the MCP contract, section 2 "Pending
approval", and the error codes are in [Error codes](../reference/error-codes.md#approval-queue-errors).
