---
sidebar_position: 3
title: Dashboard &amp; Navigation
---

# Dashboard &amp; Navigation

## Sidebar Navigation

The collapsible sidebar provides access to all pages:

| Page | Icon | Purpose |
|------|------|---------|
| **Dashboard** | Home | System overview, GPU metrics, quick actions |
| **Models** | Server | Download, load, unload, and manage LLMs |
| **SAEs** | Layers | Download, attach, and manage Sparse Autoencoders |
| **Steering** | Sliders | Configure feature steering values |
| **Probe** | Activity | Real-time activation monitoring |
| **Profiles** | FileJson | Save/load steering configurations |
| **Settings** | Settings | Theme, connection status, server info |

## Dashboard Overview

The dashboard shows four status cards at a glance:

- **Model:** Currently loaded model or "No Model" warning
- **SAE:** Attached SAE and layer or "Not Attached" warning
- **Steering:** Whether steering is enabled and feature count
- **Probe:** Whether monitoring is active

Below these, four GPU metric cards show real-time hardware status:

| Metric | Color Coding |
|--------|-------------|
| **GPU Utilization** | Percentage of compute in use |
| **GPU Memory** | Used/Total in GB |
| **GPU Temperature** | Green &lt;70°C, Yellow 70–85°C, Red ≥85°C |
| **CPU Usage** | System CPU percentage |

:::tip Keyboard Shortcuts
- `G + D` — Go to Dashboard
- `G + M` — Go to Models
- `G + S` — Go to SAE
- `G + T` — Go to Steering
- `Ctrl+Shift+T` — Toggle theme
:::
