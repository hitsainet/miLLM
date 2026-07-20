"""
title: miLLM Steering Dial
author: miLLM
version: 1.4.1
description: Per-chat steering-intensity dial for a miLLM backend. Injects the
  miLLM extension field `steering_intensity` (off | min | max, or a numeric
  lambda in [0, 2]) into /v1/chat/completions requests so each user can dial
  the active CLUSTER or CIRCUIT's steering strength without touching global
  state. When a multi-layer circuit is serving, one lambda scales every layer
  together and the status line discloses the circuit's evidence rung — a
  circuit below rung 2 is marked [UNVALIDATED] and never called causal.

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
- miLLM: requires a build with Feature 10 (per-request steering dial); the
  circuit-rung disclosure additionally needs Feature 14 and the
  `millm_base_url` valve set (it is EMPTY by default, so the disclosure is off
  until configured — "localhost" inside the OWUI container is OWUI itself).
  Older
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
        show_circuit_rung: bool = Field(
            default=True,
            description=(
                "When a CIRCUIT is steering, show its evidence rung in the status "
                "line and mark an unvalidated (rung<2) circuit. Requires "
                "millm_base_url; degrades silently when unreachable."
            ),
        )
        millm_base_url: str = Field(
            default="",
            description=(
                "miLLM base URL for the read-only circuit-status probe. EMPTY BY "
                "DEFAULT — the probe is off until you set it, because 'localhost' "
                "inside the Open WebUI container is Open WebUI itself, not miLLM. "
                "Docker: http://host.docker.internal:8000 · "
                "k8s: http://millm-backend.millm.svc.cluster.local:8000"
            ),
        )

    class UserValves(BaseModel):
        dial: DialPosition = Field(
            default="default",
            description=(
                "Your steering dial: 'default' (use the operator's setting), "
                "'server' (send nothing — the server's stored state governs, "
                "even when the operator set a default), 'off', 'min', 'max', "
                "or 'custom' (uses custom_lambda). Note for CIRCUITS: 'min' "
                "means the circuit's declared floor, and a circuit that "
                "declares none floors at 0 — so 'min' can be the same as "
                "'off'. The status line says so when it is."
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

    #: Mirrors docs/mcp-contract.md §4a VERBATIM. The server also sends the
    #: phrase in X-miLLM-Circuit-Rung; this local copy is only a fallback for
    #: the status line, and it must never be paraphrased — "causal" may not
    #: describe a circuit below rung 2.
    RUNG_LANGUAGE = {
        0: "associated",
        1: "suggested (attribution-supported)",
        2: "causally validated (edge)",
        3: "faithfulness-tested (circuit)",
    }

    #: Probe cache: the active circuit changes far less often than once per
    #: message, so a short TTL keeps steady-state messaging free.
    _CACHE_TTL_S = 10.0
    _MAX_PROBE_BYTES = 64 * 1024

    def _probe_sync(self, base: str) -> Optional[dict]:
        """Blocking probe body — always run OFF the event loop."""
        import json as _json
        import urllib.request
        from urllib.parse import urlparse

        if urlparse(base).scheme not in ("http", "https"):
            return None  # no file://, gopher://, … from an operator valve

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            # A compromised or spoofed endpoint must not redirect this
            # server-side request at an arbitrary internal address.
            def redirect_request(self, *args, **kwargs):
                return None

        opener = urllib.request.build_opener(_NoRedirect)
        req = urllib.request.Request(
            f"{base}/api/circuits/active", headers={"Accept": "application/json"}
        )
        with opener.open(req, timeout=0.8) as resp:
            payload = _json.loads(resp.read(self._MAX_PROBE_BYTES).decode("utf-8"))
        data = payload.get("data") if isinstance(payload, dict) else None
        return data if isinstance(data, dict) else None

    #: Circuit names arrive via import from shared/marketplace definitions, so
    #: they are attacker-influenced by design, and this text lands in the chat
    #: transcript (which the model may see on a later turn) rendered as
    #: markdown. R1 hardened rung_language against a spoofed endpoint and left
    #: this adjacent field from the same untrusted response unhardened.
    _MAX_NAME_LEN = 60

    @classmethod
    def _safe_name(cls, raw: object) -> str:
        """A circuit name safe to render into a chat status line."""
        if not isinstance(raw, str) or not raw.strip():
            return "circuit"
        # Collapse ALL whitespace: newlines would let a name inject what looks
        # like separate lines (or a fake system message) into the transcript.
        flat = " ".join(raw.split())
        # Neutralise markdown emphasis/link/code punctuation.
        for ch in ("*", "_", "`", "[", "]", "(", ")", "#", "|", "<", ">"):
            flat = flat.replace(ch, "")
        flat = flat.strip()
        if not flat:
            return "circuit"
        if len(flat) > cls._MAX_NAME_LEN:
            flat = flat[: cls._MAX_NAME_LEN - 1].rstrip() + "…"
        return flat

    @staticmethod
    def _min_is_off(circuit: Optional[dict]) -> bool:
        """True when "min" on this circuit resolves to zero steering.

        Circuits take a configured floor of 0.0 (clusters use 0.5), so a
        circuit whose document declares no ``budget.intensity_range`` makes
        "min" and "off" the same request.
        """
        if not circuit:
            return False
        rng = ((circuit.get("budget") or {}).get("intensity_range")) or None
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            return True  # no authored floor -> configured floor 0.0 -> off
        try:
            return float(min(rng)) == 0.0
        except (TypeError, ValueError):
            return True

    async def _circuit_status(self) -> Optional[dict]:
        """Read-only probe of GET /api/circuits/active.

        Best-effort and STRICTLY optional: it runs OFF the event loop (a
        blocking urlopen in `inlet` would stall every concurrent chat on the
        worker), is cached for a few seconds, and any failure (miLLM down,
        older build without the route, bad JSON) returns None so the dial
        degrades to the Feature 10 copy rather than blocking the message.
        """
        if not self.valves.show_circuit_rung:
            return None
        base = (self.valves.millm_base_url or "").rstrip("/")
        if not base:
            return None  # off until configured — see millm_base_url

        import asyncio
        import time

        now = time.monotonic()
        cached = getattr(self, "_probe_cache", None)
        if cached and cached[0] == base and (now - cached[1]) < self._CACHE_TTL_S:
            return cached[2]

        try:
            data = await asyncio.to_thread(self._probe_sync, base)
        except Exception:
            # Do NOT cache a failure. Caching None for the full TTL would blank
            # the [UNVALIDATED] disclosure for 10s of messages after miLLM
            # recovers — suppressing the exact safety surface this exists for.
            # A failed probe simply retries on the next message.
            return None
        self._probe_cache = (base, now, data)
        return data

    def _circuit_suffix(self, circuit: Optional[dict]) -> str:
        """' · circuit "X" — <rung phrase> [UNVALIDATED]' or ''.

        The phrase is taken from the SERVER's rung_language when present and
        only falls back to the local mirror; either way a rung<2 circuit is
        marked UNVALIDATED and never described as causal.
        """
        if not circuit:
            return ""
        try:
            rung = int(circuit.get("rung", 0))
        except (TypeError, ValueError):
            rung = 0
        # The server tells us whether this circuit is ACTUALLY steering. Only
        # it can know (slice-fallback, unparseable definition, no attached SAE
        # on a member layer all mean "active but not steering"). Deriving that
        # here from is_active overclaims — the same error the server already
        # suppresses on its own rung header. `None` means an older build that
        # does not answer, so fall back to rendering the suffix.
        if circuit.get("steering") is False:
            return ""

        # Prefer the server's phrase, but only when it MATCHES the mirrored
        # vocabulary for that rung: a spoofed or MITM'd endpoint must not be
        # able to inject "causal" for a rung-0 circuit and defeat the guarantee.
        expected = self.RUNG_LANGUAGE.get(rung, self.RUNG_LANGUAGE[0])
        server_phrase = circuit.get("rung_language")
        phrase = server_phrase if server_phrase == expected else expected
        name = self._safe_name(circuit.get("name"))
        mark = "" if rung >= 2 else " [UNVALIDATED]"
        mode = circuit.get("serving_mode")
        slice_note = " (serving a per-layer SLICE, not the whole circuit)" if (
            mode == "slice_fallback"
        ) else ""
        return f' · circuit "{name}" — {phrase}{mark}{slice_note}'

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
            circuit = await self._circuit_status()
            suffix = self._circuit_suffix(circuit)
            if dial == "off" or dial == 0.0:
                text = "miLLM steering: off for this reply"
                # R3: rendering "off" alongside a circuit attribution read as a
                # contradiction — the user could not tell whether the circuit
                # was applied. At λ=0 nothing is steering, so name nothing.
                suffix = ""
            elif dial in ("min", "max"):
                # Only name the source we actually observed — a failed probe
                # must not assert "cluster's" about an active circuit.
                bound = "circuit's declared bound" if circuit else "declared bound"
                text = f"miLLM steering: {dial} ({bound})"
                if dial == "min" and self._min_is_off(circuit):
                    # R3: a circuit with no authored intensity_range takes the
                    # configured floor, which is 0.0 for circuits — so "min" is
                    # byte-identical to "off". Saying "min (declared bound)"
                    # implies a nonzero intervention that is not happening.
                    text = "miLLM steering: min — this circuit declares no floor, so min is OFF"
            else:
                text = f"miLLM steering: λ={dial:g}"
            if (
                self.valves.show_circuit_rung
                and not (self.valves.millm_base_url or "").strip()
            ):
                # R3: with the probe unconfigured the status line looks healthy
                # while the circuit-evidence disclosure is silently off. Make
                # the missing safety surface visible rather than invisible.
                text += " · circuit evidence unavailable (set millm_base_url)"
            await self._status(__event_emitter__, text + suffix)
        else:
            await self._status(
                __event_emitter__,
                "miLLM dial idle — server steering governs "
                "(set your dial in ⚙ Valves)",
            )
        return body
