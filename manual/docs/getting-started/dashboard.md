---
sidebar_position: 3
title: Dashboard &amp; Navigation
---

# Dashboard &amp; Navigation

![miLLM Dashboard](/img/miLLM_Dashboard_01.jpg)

## Sidebar Navigation

The collapsible sidebar provides access to all pages. A persistent **status bar** at the top right shows the loaded model, SAE status, GPU metrics, and WebSocket connection state.

| Page | Purpose |
|------|---------|
| **Dashboard** | System overview, GPU metrics, quick start guide |
| **Models** | Download, load, unload, and manage LLMs |
| **SAEs** | Download, attach, and manage Sparse Autoencoders |
| **Profiles** | Save/load steering configurations |
| **Clusters** | Import and activate cluster definitions from miStudio |
| **Circuits** | Import and serve multi-layer circuits from miStudio |
| **Steering** | Configure feature steering values |
| **Feature Monitor** | Real-time SAE feature activation monitoring |
| **Probe Monitors** | Trained detectors scoring live traffic |
| **Help** | Short summaries of every page, each linked to this manual |
| **Settings** | Theme, connection status, Hugging Face token, server info |

Every page's title bar carries a **Help** link to that page's summary, with a **Manual** link
beside it. A summary has its own address (`/help/<topic>`, for example `/help/steering`), so it can
be bookmarked or shared. The book icon beside **Settings** opens this manual.

## Dashboard Overview

### System Status

Four status cards show the current state at a glance:

| Card | States |
|------|--------|
| **Model** | `LOADED` (green, shows model name) or `No Model` warning |
| **SAE** | `ATTACHED` (green, shows layer) or `NOT ATTACHED` (yellow) |
| **Steering** | `ACTIVE` with feature count, or `WAITING` (attach SAE first) |
| **Feature Monitor** | `LIVE` when monitoring, or `WAITING` (attach SAE first) |

### System Metrics

Four real-time GPU metric cards below the status:

| Metric | Format | Notes |
|--------|--------|-------|
| **GPU Utilization** | Percentage (e.g., 24%) | 100% = fully utilized |
| **GPU Memory** | Used/Total GB (e.g., 9.5/24.0 GB) | Watch for &gt;90% before starting jobs |
| **GPU Temperature** | Degrees C (e.g., 64°C) | Green &lt;70°C, Yellow 70–85°C, Red ≥85°C |
| **CPU Usage** | Percentage | System CPU load |

### Quick Start Guide

The dashboard includes a **step-by-step Quick Start** checklist:

1. **Load a Model** — Download and load from HuggingFace *(green checkmark when complete)*
2. **Download &amp; Attach SAE** — Get an SAE and hook it to a model layer
3. **Configure Steering** — Add features and set strength values
4. **Monitor Activations** — Watch which features fire during inference

Steps unlock progressively — each step links directly to the relevant page.

### Quick Actions

Four shortcut buttons provide direct access to:
- **Manage Model** → Models page
- **Attach SAE** → SAEs page
- **Configure Steering** → Steering page
- **View Feature Monitor** → Feature Monitor page

## Top Status Bar

The persistent bar across the top right of every page shows:
- **Model name** (teal badge) or nothing if unloaded
- **SAE status** — `No SAE` badge if not attached
- **GPU metrics** — utilization %, memory, temperature (compact)
- **Connection** — `Connected` (green) or `Disconnected` (red)

## Settings Page

![miLLM Settings Page](/img/miLLM_Settings_01.jpg)

The **Settings** page provides:
- **Theme toggle** — Switch between light and dark mode
- **Connection status** — WebSocket state and reconnect button
- **Hugging Face token** — save a token once so downloads and previews of gated or private
  repositories no longer need one typed in each time. **Save** stores it encrypted on the server
  (this needs `SETTINGS_ENCRYPTION_KEY`; see [Configuration](../reference/configuration.md#hugging-face-token)),
  **Test** asks Hugging Face which account it belongs to, and **Clear** removes it. The card shows the
  masked token (`hf_…abcd`), which source is active (saved token, the server's `HF_TOKEN`, or none)
  and when it was last verified. A token typed into a download form still overrides the saved one
  for that request; the forms say "Leave empty to use the saved token" when one is saved.
- **Environment** — every setting the server was started with. Thirty of them can be changed while
  miLLM runs, with no restart: each row shows a plain name and what it does, **when a change takes
  effect**, the environment's value next to the one in force, and who changed it last and why. Type a
  value (and, optionally, a reason) and **Save** to store an override; **Use the environment value**
  removes it. Settings that apply at the next model load, or need a restart of `millm-backend`, are
  listed read-only, and secrets show only "set" or "not set". **Recent changes** lists the last 50.
  The full list and rules are in
  [Configuration](../reference/configuration.md#changing-settings-while-millm-runs).
- **Server information** — API version, base paths, WebSocket endpoint
- **Keyboard shortcuts** reference

:::tip Keyboard Shortcuts
- `G + D` — Go to Dashboard
- `G + M` — Go to Models
- `G + S` — Go to SAE
- `G + T` — Go to Steering
- `Ctrl+Shift+T` — Toggle theme
:::
