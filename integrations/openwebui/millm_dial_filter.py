"""
title: miLLM Cluster Dial
author: miLLM
version: 1.2.0
description: Per-chat steering-intensity dial for a miLLM backend. Injects the
  miLLM extension field `steering_intensity` (off | min | max, or a numeric
  lambda in [0, 2]) into /v1/chat/completions requests so each user can dial
  the active cluster's steering strength without touching the global state.
requirements: none (pydantic ships with Open WebUI)

Compatibility
-------------
- Open WebUI: uses only the stable Filter surface (class Filter, Valves /
  UserValves, inlet) — no outlet, no event emitters. Verified against the
  Filter API as of OWUI 0.6.x; the surface has been stable since 0.5.
  Literal-typed valves render as dropdowns, making typos unrepresentable.
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

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        if not self.valves.enabled:
            return body

        user_valves = (__user__ or {}).get("valves")
        # 'server' is the explicit escape hatch: send no field, ignoring the
        # operator default — without it, once the operator sets default_dial
        # a user has no position meaning "server state governs" (US-10.1).
        if str(self._read(user_valves, "dial", "") or "").strip().lower() == "server":
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
        return body
