"""
title: miLLM Cluster Dial
author: miLLM
version: 1.3.0
description: Per-chat steering-intensity dial for a miLLM backend. Injects the
  miLLM extension field `steering_intensity` (off | min | max, or a numeric
  lambda in [0, 2]) into /v1/chat/completions requests so each user can dial
  the active cluster's steering strength without touching the global state.

UX (v1.3.0)
-----------
- Appears as a TOGGLE CHIP in the message input (sliders icon). Chip OFF =
  this filter does not run = the server's stored steering governs. Chip ON =
  your `dial` valve applies to this chat's requests.
- Emits a one-line STATUS into the chat showing exactly what was sent
  ("miLLM steering: off for this reply" / "λ=1.5" / idle hint). Operators
  can silence it with the `show_status` valve.
- On older Open WebUI without toggle-filter support both attributes are
  ignored: the filter simply runs on every request, as in v1.2.

Compatibility
-------------
- No pip requirements (pydantic ships with Open WebUI). NEVER add a
  `requirements:` frontmatter key with prose in it — OWUI pip-installs the
  value VERBATIM on save and at every startup; a sentence there crashed the
  OWUI worker on save (2026-07-18). Omit the key entirely when empty.
- Open WebUI: Filter surface (class Filter, Valves / UserValves, inlet) plus
  the toggle/icon chip and __event_emitter__ status API (verified on 0.10.x;
  chip support exists since ~0.6.10). No outlet, no stream hook.
- miLLM: requires a build with Feature 10 (per-request steering dial). Older
  miLLM builds silently ignore the injected field (`extra="ignore"` on the
  request schema), so enabling this Function against an older backend is safe
  and simply has no effect — the rollout property EC-10.4 pins.
- Mixed providers: enable this Function PER MODEL (only on miLLM-served
  models), not globally, if your Open WebUI instance also talks to strict
  OpenAI-compatible providers — some reject unknown request fields with 400.

There is deliberately NO outlet hook: miLLM applies the dial inside its
request boundary and restores the previous steering state in a finally block
server-side, including on client disconnects. Nothing to undo client-side.
"""

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

DialPosition = Literal["default", "server", "off", "min", "max", "custom"]

# Sliders glyph for the input-bar chip (data URI — OWUI renders it verbatim).
_ICON = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48bGluZSB4MT0iMjEiIHkxPSI0IiB4Mj0iMTQiIHkyPSI0Ii8+PGxpbmUgeDE9IjEwIiB5MT0iNCIgeDI9IjMiIHkyPSI0Ii8+PGxpbmUgeDE9IjIxIiB5MT0iMTIiIHgyPSIxMiIgeTI9IjEyIi8+PGxpbmUgeDE9IjgiIHkxPSIxMiIgeDI9IjMiIHkyPSIxMiIvPjxsaW5lIHgxPSIyMSIgeTE9IjIwIiB4Mj0iMTYiIHkyPSIyMCIvPjxsaW5lIHgxPSIxMiIgeTE9IjIwIiB4Mj0iMyIgeTI9IjIwIi8+PGxpbmUgeDE9IjE0IiB5MT0iMiIgeDI9IjE0IiB5Mj0iNiIvPjxsaW5lIHgxPSI4IiB5MT0iMTAiIHgyPSI4IiB5Mj0iMTQiLz48bGluZSB4MT0iMTYiIHkxPSIxOCIgeDI9IjE2IiB5Mj0iMjIiLz48L3N2Zz4="


class Filter:
    class Valves(BaseModel):
        enabled: bool = Field(
            default=True,
            description="Master switch — when off, requests pass through untouched.",
        )
        default_dial: DialPosition = Field(
            default="default",
            description=(
                "Dial applied for users who leave theirs at 'default': "
                "'default' leaves steering as-is; 'custom' uses default_custom_lambda."
            ),
        )
        default_custom_lambda: float = Field(
            default=1.0,
            ge=0.0,
            le=2.0,
            description="Numeric lambda used when default_dial is 'custom' (0..2).",
        )
        show_status: bool = Field(
            default=True,
            description="Emit a one-line status into the chat showing the dial applied.",
        )

    class UserValves(BaseModel):
        dial: DialPosition = Field(
            default="default",
            description=(
                "Your steering dial: 'default' (use the operator's setting), "
                "'server' (send nothing — the server's stored state governs, "
                "even when the operator set a default), 'off', 'min', 'max', "
                "or 'custom' (uses custom_lambda)."
            ),
        )
        custom_lambda: float = Field(
            default=1.0,
            ge=0.0,
            le=2.0,
            description="Numeric lambda used when dial is 'custom' (0..2).",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        # Per-chat chip in the message input; ignored (always-on) on OWUI
        # builds without toggle-filter support.
        self.toggle = True
        self.icon = _ICON

    @staticmethod
    def _read(valves: object, key: str, fallback: object) -> object:
        """Read a valve field from a pydantic model OR a plain dict — OWUI
        versions differ in what shape __user__['valves'] arrives as, and a
        dict handed to getattr would silently drop the user's dial."""
        if isinstance(valves, dict):
            return valves.get(key, fallback)
        return getattr(valves, key, fallback)

    @classmethod
    def _resolve(cls, valves: object) -> Optional[Union[str, float]]:
        """A valve set's dial as the miLLM field value, or None for 'default'.

        Invalid shapes degrade to None — a broken valve must never break the
        user's chat."""
        dial = str(cls._read(valves, "dial", "default") or "default").strip().lower()
        if dial in ("off", "min", "max"):
            return dial
        if dial == "custom":
            try:
                lam = float(cls._read(valves, "custom_lambda", 1.0))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            return lam if 0.0 <= lam <= 2.0 else None
        return None

    async def _status(self, emitter, text: str) -> None:
        """Best-effort status line — a broken emitter must never break chat."""
        if emitter is None or not self.valves.show_status:
            return
        try:
            await emitter(
                {"type": "status", "data": {"description": text, "done": True}}
            )
        except Exception:
            pass

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__=None,
    ) -> dict:
        if not self.valves.enabled:
            return body

        user_valves = (__user__ or {}).get("valves")
        # 'server' is the explicit escape hatch: send no field, ignoring the
        # operator default — without it, once the operator sets default_dial
        # a user has no position meaning "server state governs" (US-10.1).
        if str(self._read(user_valves, "dial", "") or "").strip().lower() == "server":
            await self._status(
                __event_emitter__, "miLLM dial: server steering state governs"
            )
            return body

        # Per-user dial wins; the operator default fills in when the user
        # leaves theirs at 'default'. When a dial resolves, it REPLACES any
        # steering_intensity already present in the body (scripted clients
        # keep their field only when every valve is default/server).
        dial = self._resolve(user_valves)
        if dial is None:
            dial = self._resolve(
                {"dial": self.valves.default_dial,
                 "custom_lambda": self.valves.default_custom_lambda}
            )
        if dial is not None:
            body["steering_intensity"] = dial
            if dial == "off" or dial == 0.0:
                text = "miLLM steering: off for this reply"
            elif dial in ("min", "max"):
                text = f"miLLM steering: {dial} (cluster's declared bound)"
            else:
                text = f"miLLM steering: λ={dial:g}"
            await self._status(__event_emitter__, text)
        else:
            await self._status(
                __event_emitter__,
                "miLLM dial idle — server steering governs "
                "(set your dial in ⚙ Valves)",
            )
        return body
