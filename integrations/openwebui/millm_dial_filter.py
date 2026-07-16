"""
title: miLLM Cluster Dial
author: miLLM
version: 1.0.0
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
- miLLM: requires a build with Feature 10 (per-request steering dial). Older
  miLLM builds silently ignore the injected field (`extra="ignore"` on the
  request schema), so enabling this Function against an older backend is safe
  and simply has no effect — the rollout property EC-10.4 pins.

There is deliberately NO outlet hook: miLLM applies the dial inside its
request boundary and restores the previous steering state in a finally block
server-side, including on client disconnects. Nothing to undo client-side.
"""

from typing import Optional

from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        enabled: bool = Field(
            default=True,
            description="Master switch — when off, requests pass through untouched.",
        )
        default_dial: str = Field(
            default="default",
            description=(
                "Dial applied when a user hasn't set their own: 'default' "
                "(leave steering as-is), 'off', 'min', 'max', or a number "
                "0..2 (e.g. '1.25')."
            ),
        )

    class UserValves(BaseModel):
        dial: str = Field(
            default="default",
            description=(
                "Your steering dial: 'default' (use the operator's setting), "
                "'off', 'min', 'max', or a number 0..2 (e.g. '0.8')."
            ),
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    @staticmethod
    def _parse_dial(raw: str) -> Optional[object]:
        """Normalize a valve string to the miLLM field value, or None.

        Returns 'off'/'min'/'max' verbatim, a float for numeric strings within
        [0, 2], and None for 'default', blanks, or anything unparseable —
        an invalid valve must never break the user's chat.
        """
        value = (raw or "").strip().lower()
        if value in ("", "default"):
            return None
        if value in ("off", "min", "max"):
            return value
        try:
            numeric = float(value)
        except ValueError:
            return None
        if 0.0 <= numeric <= 2.0:
            return numeric
        return None

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        if not self.valves.enabled:
            return body

        # Per-user dial wins; operator default fills in when the user leaves
        # theirs at 'default'.
        user_valves = (__user__ or {}).get("valves")
        dial = self._parse_dial(getattr(user_valves, "dial", "") or "")
        if dial is None:
            dial = self._parse_dial(self.valves.default_dial)
        if dial is not None:
            body["steering_intensity"] = dial
        return body
