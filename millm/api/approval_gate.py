"""Agent origin and the approval gate (ENH-008 D1; BUILD_SPEC §3.1).

Pattern copied from miForge's ``backend/src/api/agent_origin.py`` (itself adapted from
miDataworks); not imported: the apps stay standalone. Changed for miLLM: async routes and sessions;
FastAPI dependencies (services, repositories) are REBUILT for the replay rather than refused;
a route that reads its raw body through ``Request`` has the body stored and replayed; values the
operator must never see (the lease id) go to ``secret_payload``.

**Origin.** A request carrying ``X-miLLM-Agent`` is an agent's; any other request is the
operator's. There is no login in the suite (operator decision 2026-10-06), so the header is
SELF-DECLARED: it binds agents that identify themselves, not adversaries. Anyone who can reach
``/api`` could already act as the operator, exactly as before this feature and exactly as in the
sibling apps. miStudio's MCP client sends the header on every request it makes to miLLM. The gate
FAILS CLOSED on the header: any value at all, even empty or malformed, marks the call an agent's.

**The gate.** :func:`requires_approval_when_agent` wraps a route function. The operator's call runs
it unchanged. An agent's call (when the ``when`` predicate holds; a predicate that raises gates)
is described (``describe=``: the route's own refusals run here, before anything is stored), stored
as a pending request and answered ``202`` with ``X-miLLM-Approval: pending``. Approving runs the
stored request through :data:`EXECUTORS`, the ORIGINAL function, in-process, exactly once
(``millm.services.approval_service``). It is never an HTTP self-call, which would have to drop the
agent header or re-enter this gate.
"""

from __future__ import annotations

import functools
import inspect
import re
from collections.abc import Awaitable, Callable, Iterator
from contextlib import AsyncExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, get_args, get_origin

from fastapi import Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.params import Depends as DependsParam
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from millm.api.dependencies import get_db
from millm.core.errors import ApprovalOperatorOnlyError, ValidationError
from millm.core.logging import get_logger

logger = get_logger(__name__)

AGENT_HEADER = "X-miLLM-Agent"
APPROVAL_HEADER = "X-miLLM-Approval"

#: The OpenAPI extension every gated route publishes (read by the coverage guard and by
#: ``GET /api/approvals/actions``).
X_APPROVAL_ACTION = "x-approval-action"

#: Where a route's raw request body is kept in the stored payload, for a route that reads it
#: through ``Request`` (the rule-set import posts the producer's document unchanged).
BODY_KEY = "request_body"

#: The longest agent name kept (the ``requested_by`` column's width).
_REQUESTED_BY_MAX = 64
_PRINTABLE = re.compile(r"[^\x21-\x7e]")


@dataclass(frozen=True)
class ApprovalAction:
    label: str
    description: str
    #: Which "Approve all" batch may approve this action (``POST /api/approvals/approve-batch``
    #: with ``kind``), or ``None``: approved one at a time only. A batch never mixes kinds.
    #: * ``settings`` (ENH-006 D79): only changes a stored setting (no GPU work, nothing deleted,
    #:   nothing imported).
    #: * ``code_rules`` (ENH-006 D81): an import of operator code; the batch confirm shows each
    #:   rule set and its code hashes.
    batch_kind: Literal["settings", "code_rules"] | None = None

    @property
    def settings_change(self) -> bool:
        return self.batch_kind == "settings"


#: Every gated action with the label the operator reads on the card and the sentence
#: ``GET /api/approvals/actions`` serves. Pinned equal to
#: ``millm.db.models.approval.APPROVAL_ACTION_NAMES`` (the CHECK constraint) by a test.
APPROVAL_ACTIONS: dict[str, ApprovalAction] = {
    "model_load": ApprovalAction(
        "Load a model in miLLM",
        "Loads a model onto a graphics card. The model being served now is unloaded first, and "
        "requests already running are given time to finish."),
    "model_unload": ApprovalAction(
        "Unload a model in miLLM",
        "Unloads the served model. Chat requests then have no model until one is loaded; an "
        "attached sparse autoencoder and any active steering are dropped with it."),
    "model_delete": ApprovalAction(
        "Delete a model from miLLM",
        "Deletes a downloaded model from disk and from miLLM's list. It cannot be recovered; it "
        "has to be downloaded again."),
    "probe_limits_write": ApprovalAction(
        "Change a model's probe limits in miLLM",
        "Sets how many probes and behaviours may be armed on a model, and the time each pass may "
        "spend on them.", batch_kind="settings"),
    "capacity_measure": ApprovalAction(
        "Measure a model's probe capacity in miLLM",
        "Runs the resident model on its graphics card to measure the step time and the cost of "
        "each probe. Chat requests wait while it runs."),
    "code_rule_import": ApprovalAction(
        "Import a rule set with operator code into miLLM",
        "Imports a rule set that names operator code. The code runs in miLLM's isolated rule "
        "runner on chat text after each response.", batch_kind="code_rules"),
    # ENH-006 D76: telemetry settings became changeable while miLLM runs (D75).
    "telemetry_write": ApprovalAction(
        "Change telemetry settings in miLLM",
        "Changes where miLLM's OpenTelemetry export goes or whether it runs (the switches, the "
        "receiver endpoint, sampling, queue and timing). Approving drains and restarts the export "
        "pipeline with the new values.", batch_kind="settings"),
}


#: The batch kinds, in the order the UI offers them.
BATCH_KINDS: tuple[str, ...] = ("settings", "code_rules")


def batch_actions(kind: str) -> frozenset[str]:
    """The actions one "Approve all" batch of ``kind`` may approve (ENH-006 D79, D81)."""
    return frozenset(name for name, a in APPROVAL_ACTIONS.items() if a.batch_kind == kind)


def settings_actions() -> frozenset[str]:
    """The actions "Approve all settings changes" may approve (ENH-006 D79)."""
    return batch_actions("settings")


@dataclass(frozen=True)
class Actor:
    """Who is calling, as far as the request says."""

    origin: Literal["operator", "agent"]
    #: ``agent:<name>``, or ``agent`` when the header carried no name.
    agent: str | None = None
    #: Set while an approved request runs, so the action can name its approval.
    approval_id: str | None = None


_executing: ContextVar[Actor | None] = ContextVar("millm_executing_actor", default=None)


@contextmanager
def executing_as(actor: Actor) -> Iterator[None]:
    """Run the block as ``actor``: the identity of a stored request while its executor runs.

    The executor replays only a route's values, so a route cannot learn through its signature who
    asked. Reset in ``finally``: later work on this task never inherits an agent's identity.
    """
    token = _executing.set(actor)
    try:
        yield
    finally:
        _executing.reset(token)


def executing_actor() -> Actor | None:
    """The approved request's requester while its executor runs; ``None`` otherwise."""
    return _executing.get()


def agent_name(raw: str) -> str:
    """The ``requested_by`` value for a header value: ``agent:<name>``, printable, at most 64.

    Never refuses: whatever an agent sent, the call is an agent's (fail closed), and the name is
    only a label for the card.
    """
    name = _PRINTABLE.sub("", raw)
    if name.startswith("agent:"):
        name = name[len("agent:"):]
    if not name:
        return "agent"
    return ("agent:" + name)[:_REQUESTED_BY_MAX]


def actor_of(request: Request) -> Actor:
    raw = request.headers.get(AGENT_HEADER)
    if raw is None:
        return Actor(origin="operator")
    return Actor(origin="agent", agent=agent_name(raw))


def refuse_agent(request: Request) -> None:
    """``403 approval_operator_only`` for any agent; nothing for the operator (OSD-56)."""
    if request.headers.get(AGENT_HEADER) is not None:
        raise ApprovalOperatorOnlyError(
            "An agent cannot approve or reject a request; the operator decides in miLLM's "
            "Approvals page.",
            details={"header": AGENT_HEADER},
        )


# --------------------------------------------------------------------------------------------
# Annotations and dependencies
# --------------------------------------------------------------------------------------------


def resolve_annotation(annotation: Any, owner: Callable[..., Any]) -> Any:
    """An annotation as an object. Route modules using ``from __future__ import annotations``
    hold strings, evaluated here against the module that declared them."""
    if isinstance(annotation, str):
        return eval(annotation, getattr(owner, "__globals__", {}))  # noqa: S307 - own source
    return annotation


def base_type(hint: Any) -> Any:
    """``Annotated[X, ...]`` -> ``X``; anything else unchanged."""
    if get_origin(hint) is Annotated:
        return get_args(hint)[0]
    return hint


def dependency_of(hint: Any, default: Any) -> Callable[..., Any] | None:
    """The callable a ``Depends`` names, from the default or the ``Annotated`` metadata."""
    if isinstance(default, DependsParam):
        return default.dependency
    if get_origin(hint) is Annotated:
        for meta in get_args(hint)[1:]:
            if isinstance(meta, DependsParam):
                return meta.dependency
    return None


def _is_request(hint: Any) -> bool:
    return base_type(hint) is Request


def check_replayable(dependency: Callable[..., Any], seen: frozenset[Any] = frozenset()) -> None:
    """Refuse at import time a dependency the executor could not rebuild.

    Every parameter must itself be a dependency, the ``Request``, or have a default. Recursive.
    """
    if dependency in seen:
        return
    for name, param in inspect.signature(dependency).parameters.items():
        hint = resolve_annotation(param.annotation, dependency)
        sub = dependency_of(hint, param.default)
        if sub is not None:
            check_replayable(sub, seen | {dependency})
        elif _is_request(hint) or param.default is not inspect.Parameter.empty:
            continue
        else:
            raise TypeError(
                f"{getattr(dependency, '__qualname__', dependency)!r}: parameter {name!r} cannot "
                f"be rebuilt when an approved request runs"
            )


class DependencyResolver:
    """Rebuild a route's FastAPI dependencies for a stored request, outside any HTTP request.

    The app's ``dependency_overrides`` apply, as they do for a live request. Each dependency is
    built once per replay (FastAPI's per-request cache), generator dependencies are closed by
    :meth:`aclose`, and a ``Request`` parameter receives a synthetic request bound to the app.
    """

    def __init__(self, app: Any, request: Request) -> None:
        self._overrides: dict[Any, Any] = dict(getattr(app, "dependency_overrides", {}) or {})
        self._request = request
        self._cache: dict[Any, Any] = {}
        self._stack = AsyncExitStack()

    async def resolve(self, dependency: Callable[..., Any]) -> Any:
        if dependency in self._cache:
            return self._cache[dependency]
        call = self._overrides.get(dependency, dependency)
        kwargs: dict[str, Any] = {}
        for name, param in inspect.signature(call).parameters.items():
            hint = resolve_annotation(param.annotation, call)
            sub = dependency_of(hint, param.default)
            if sub is not None:
                kwargs[name] = await self.resolve(sub)
            elif _is_request(hint):
                kwargs[name] = self._request
        produced = call(**kwargs)
        if inspect.isasyncgen(produced):
            value = await produced.__anext__()
            self._stack.push_async_callback(produced.aclose)
        elif inspect.isgenerator(produced):
            value = next(produced)
            self._stack.callback(produced.close)
        elif inspect.isawaitable(produced):
            value = await produced
        else:
            value = produced
        self._cache[dependency] = value
        return value

    async def aclose(self) -> None:
        await self._stack.aclose()


def synthetic_request(app: Any, body: bytes = b"") -> Request:
    """A ``Request`` for a replay: bound to the app, carrying the stored body, and with NO
    agent header (the stored request runs as approved, never back through the gate)."""
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {"type": "http", "method": "POST", "path": "/", "headers": [],
             "query_string": b"", "app": app}
    return Request(scope, receive)


# --------------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------------

#: ``values`` -> bool. Values are the route's path, query and body parameters as parsed (and the
#: raw body under :data:`BODY_KEY` for a route that reads it).
Predicate = Callable[[dict[str, Any]], bool]
#: ``(values, session)`` -> the operator-facing facts; raises the route's own refusal.
Describer = Callable[[dict[str, Any], AsyncSession], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Executor:
    """How to run a stored request once the operator approves it."""

    key: str
    action: str
    fn: Callable[..., Awaitable[Any]]
    #: name -> the annotation the stored JSON value is rebuilt with (``Annotated`` stripped).
    value_params: dict[str, Any]
    #: name -> the dependency callable rebuilt for the replay.
    dependency_params: dict[str, Callable[..., Any]]
    #: The parameter receiving a ``Request`` (with the stored body), if the route has one.
    request_param: str | None
    #: Value parameters kept in ``secret_payload``, never in ``payload``.
    secret_params: tuple[str, ...] = field(default=())


#: Every gated route, keyed by ``module.qualname``. Read by the approval service on approve.
EXECUTORS: dict[str, Executor] = {}

GATE_REQUEST_PARAM = "millm_gate_request"
GATE_SESSION_PARAM = "millm_gate_session"


def decide(when: Predicate | None, values: dict[str, Any], key: str) -> bool:
    """True when this agent call waits for the operator. A predicate that raises gates."""
    if when is None:
        return True
    try:
        return bool(when(values))
    except Exception:
        # FAIL CLOSED: a predicate that cannot decide gates the call.
        logger.warning("approval_predicate_raised", route=key, detail="gating the call")
        return True


def requires_approval_when_agent(
    action: str,
    *,
    when: Predicate | None = None,
    describe: Describer | None = None,
    secret: tuple[str, ...] = (),
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Gate a route for agent calls.

    Args:
        action: the approval action this route's agent calls wait for.
        when: the class predicate; the call is gated only when it returns true. A predicate that
            raises FAILS CLOSED (gated). Absent: always gated.
        describe: the facts shown on the card. Runs the route's own refusals (a missing model is
            the route's 404) before anything is stored.
        secret: value parameters stored apart and never returned (the lease id).

    Every route parameter must be a plain value (path, query, body, header), the ``Request``, or
    a dependency the executor can rebuild; anything else is refused at import time.
    """
    if action not in APPROVAL_ACTIONS:
        raise ValueError(f"{action!r} is not a registered approval action")

    def decorate(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        sig = inspect.signature(fn)
        hints: dict[str, Any] = {}
        value_params: dict[str, Any] = {}
        dependency_params: dict[str, Callable[..., Any]] = {}
        request_param: str | None = None
        for name, param in sig.parameters.items():
            hint = resolve_annotation(param.annotation, fn)
            hints[name] = hint
            dependency = dependency_of(hint, param.default)
            if dependency is not None:
                check_replayable(dependency)
                dependency_params[name] = dependency
            elif _is_request(hint):
                request_param = name
            else:
                value_params[name] = base_type(hint)
        unknown = set(secret) - set(value_params)
        if unknown:
            raise TypeError(f"{fn.__qualname__}: secret names no value parameter: {unknown}")

        key = f"{fn.__module__}.{fn.__qualname__}"
        EXECUTORS[key] = Executor(
            key=key, action=action, fn=fn, value_params=value_params,
            dependency_params=dependency_params, request_param=request_param,
            secret_params=tuple(secret),
        )

        @functools.wraps(fn)
        async def wrapper(**kwargs: Any) -> Any:
            # FastAPI hands the request to ONE parameter only: the route's own when it has one.
            gate_request: Request = (kwargs[request_param] if request_param is not None
                                     else kwargs.pop(GATE_REQUEST_PARAM))
            gate_session: AsyncSession = kwargs.pop(GATE_SESSION_PARAM)
            actor = actor_of(gate_request)
            if actor.origin != "agent":
                return await fn(**kwargs)
            values = {name: kwargs[name] for name in value_params}
            if request_param is not None:
                raw = await kwargs[request_param].body()
                try:
                    values[BODY_KEY] = raw.decode("utf-8")
                except UnicodeDecodeError:
                    raise ValidationError("the request body is not UTF-8 text",
                                          details={"param": "body"}) from None
            if not decide(when, values, key):
                return await fn(**kwargs)
            summary = await describe(values, gate_session) if describe is not None else {}
            from millm.services import approval_service

            approval = await approval_service.create_pending(
                gate_session,
                action=action,
                executor_key=key,
                payload={n: jsonable_encoder(v) for n, v in values.items() if n not in secret},
                secret_payload={n: jsonable_encoder(values[n]) for n in secret
                                if values.get(n) is not None} or None,
                requested_by=actor.agent or "agent",
                summary=summary,
            )
            return JSONResponse(
                status_code=202,
                content={"success": True, "data": approval_service.pending_body(approval),
                         "error": None},
                headers={APPROVAL_HEADER: "pending"},
            )

        # FastAPI reads THIS signature: the route's own parameters with their annotations
        # resolved (a string annotation would otherwise be evaluated against this module), plus
        # the request (for the agent header) and a session (to store the request). Keyword-only,
        # so parameters with and without defaults may sit in any order.
        params = [p.replace(annotation=hints[n], kind=inspect.Parameter.KEYWORD_ONLY)
                  for n, p in sig.parameters.items()]
        if request_param is None:
            params.append(inspect.Parameter(GATE_REQUEST_PARAM, inspect.Parameter.KEYWORD_ONLY,
                                            annotation=Request))
        params.append(inspect.Parameter(
            GATE_SESSION_PARAM, inspect.Parameter.KEYWORD_ONLY,
            annotation=Annotated[AsyncSession, Depends(get_db)]))
        wrapper.__signature__ = sig.replace(  # type: ignore[attr-defined]
            parameters=params, return_annotation=sig.return_annotation)
        del wrapper.__wrapped__  # FastAPI must read the signature above
        wrapper.__millm_approval_action__ = action  # type: ignore[attr-defined]
        wrapper.__millm_executor_key__ = key  # type: ignore[attr-defined]
        return wrapper

    return decorate


def gated_routes(app: Any) -> list[tuple[str, str, str]]:
    """``(METHOD, path, action)`` for every gated route the app serves, from the LIVE route
    table (``fastapi.routing.iter_route_contexts``, the walk FastAPI's own OpenAPI uses; from
    FastAPI 0.14x ``app.routes`` holds an included router as one opaque entry)."""
    from fastapi.routing import APIRoute, iter_route_contexts

    out: list[tuple[str, str, str]] = []
    for route in iter_route_contexts(app.routes):
        if not isinstance(route.original_route, APIRoute):
            continue
        action = getattr(route.endpoint, "__millm_approval_action__", None)
        if action:
            for method in sorted(route.methods or ()):
                out.append((method, str(route.path), str(action)))
    return sorted(out)


def mark_gated_routes(app: Any) -> int:
    """Publish each gated route's action as ``x-approval-action`` in the OpenAPI document.
    Returns how many routes were marked. Run on the assembled app (``register_routes``)."""
    from fastapi.routing import APIRoute, iter_route_contexts

    marked = 0
    for route in iter_route_contexts(app.routes):
        declared = route.original_route
        if not isinstance(declared, APIRoute):
            continue
        action = getattr(route.endpoint, "__millm_approval_action__", None)
        if not action:
            continue
        extra = {**(declared.openapi_extra or {}), X_APPROVAL_ACTION: action}
        declared.openapi_extra = extra
        # An included route's effective context COPIES openapi_extra when it is built, and
        # walking the table builds it; set both so the generator reads the marker.
        effective = getattr(route, "_route_context", None)
        if effective is not None:
            effective.openapi_extra = extra
        marked += 1
    return marked
