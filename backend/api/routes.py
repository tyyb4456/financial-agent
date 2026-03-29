"""
api/routes.py
-------------
FastAPI router — graph execution endpoints.

Every request gets a correlation_id (UUID) injected into structlog's
context vars so every log line during that request carries the same ID.
LangSmith tracing metadata + tags are injected via trace_graph_run().

Endpoints:
  GET  /graphs                     → list all graphs + input schemas
  POST /run/{graph_name}           → sync invoke, returns full result
  POST /stream/{graph_name}        → SSE streaming (node updates + LLM tokens)
  GET  /history/{graph_name}       → checkpoint history for a thread
"""

from __future__ import annotations

import json
import uuid
import structlog
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from config.checkpointer  import get_async_checkpointer
from config.observability  import trace_graph_run
from registry import get_registry

log    = structlog.get_logger(__name__)
router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    inputs:    dict[str, Any]
    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Thread ID for checkpointing. Auto-generated if omitted.",
    )


class RunResponse(BaseModel):
    graph:          str
    thread_id:      str
    result:         str
    correlation_id: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_entry(graph_name: str):
    registry = get_registry()
    entry    = registry.get(graph_name)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error":     f"Graph '{graph_name}' not found.",
                "available": list(registry.keys()),
            },
        )
    return entry


def _validate(entry, raw: dict) -> dict:
    try:
        return entry.schema(**raw).model_dump()
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "Input validation failed.", "detail": exc.errors()},
        )


def _correlation_id() -> str:
    return str(uuid.uuid4())


def _safe_serialise(obj: Any) -> Any:
    """Make graph state JSON-safe. Truncates large strings for SSE payloads."""
    if isinstance(obj, dict):
        return {k: _safe_serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialise(i) for i in obj]
    if isinstance(obj, str) and len(obj) > 500:
        return obj[:500] + "…[truncated]"
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/graphs", summary="List available graphs")
async def list_graphs():
    """Return all registered graphs with descriptions and input schemas."""
    registry = get_registry()
    return {
        name: {
            "description":  entry.description,
            "input_schema": entry.schema.model_json_schema(),
        }
        for name, entry in registry.items()
    }


@router.post("/run/{graph_name}", response_model=RunResponse, summary="Run a graph")
async def run_graph(graph_name: str, request: RunRequest):
    """
    Invoke a graph synchronously and return the full result.

    Every super-step is checkpointed. LangSmith traces are automatically
    tagged with graph name, thread_id, environment, and app version when
    LANGSMITH_TRACING=true.
    """
    cid    = _correlation_id()
    entry  = _get_entry(graph_name)
    inputs = _validate(entry, request.inputs)

    # Bind correlation_id to all log lines in this request
    structlog.contextvars.bind_contextvars(
        correlation_id=cid,
        graph=graph_name,
        thread_id=request.thread_id,
    )

    log.info("graph.run.start", inputs_keys=list(inputs.keys()))

    try:
        async with trace_graph_run(graph_name, request.thread_id, inputs) as lg_config:
            async with get_async_checkpointer() as saver:
                compiled = entry.with_checkpointer(saver)
                output   = await compiled.ainvoke(inputs, config=lg_config)
                result   = output.get("result", str(output))

    except Exception as exc:
        log.error("graph.run.error", error=str(exc))
        raise HTTPException(status_code=500, detail={"error": str(exc)})
    finally:
        structlog.contextvars.clear_contextvars()

    log.info("graph.run.done")
    return RunResponse(
        graph=graph_name,
        thread_id=request.thread_id,
        result=result,
        correlation_id=cid,
    )


@router.post("/stream/{graph_name}", summary="Stream a graph via SSE")
async def stream_graph(graph_name: str, request: RunRequest):
    """
    Stream graph execution as Server-Sent Events (SSE).

    Event types emitted:
      node_update  → { type, node, data }
      llm_token    → { type, token }
      done         → { type, thread_id, correlation_id }
      error        → { type, error, correlation_id }
    """
    cid    = _correlation_id()
    entry  = _get_entry(graph_name)
    inputs = _validate(entry, request.inputs)

    log.info(
        "graph.stream.start",
        graph=graph_name,
        thread_id=request.thread_id,
        correlation_id=cid,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        def _emit(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"

        try:
            async with trace_graph_run(graph_name, request.thread_id, inputs) as lg_config:
                async for chunk in entry.graph.astream(
                    inputs,
                    config=lg_config,
                    stream_mode=["updates", "messages"],
                    version="v2",
                ):
                    chunk_type = chunk.get("type")
                    data       = chunk.get("data")

                    if chunk_type == "updates":
                        for node_name, state_diff in data.items():
                            if node_name.startswith("__"):
                                continue
                            yield _emit({
                                "type": "node_update",
                                "node": node_name,
                                "data": _safe_serialise(state_diff),
                            })

                    elif chunk_type == "messages":
                        msg_chunk, _ = data
                        token = getattr(msg_chunk, "content", "")
                        if token:
                            yield _emit({"type": "llm_token", "token": token})

            yield _emit({"type": "done", "thread_id": request.thread_id, "correlation_id": cid})

        except Exception as exc:
            log.error("graph.stream.error", graph=graph_name, error=str(exc), correlation_id=cid)
            yield _emit({"type": "error", "error": str(exc), "correlation_id": cid})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
            "X-Correlation-ID":  cid,
        },
    )


@router.get("/history/{graph_name}", summary="Get checkpoint history for a thread")
async def get_history(graph_name: str, thread_id: str):
    """
    Return all saved checkpoints for a graph + thread_id pair.
    Requires that /run was previously called with the same thread_id.
    """
    cid   = _correlation_id()
    entry = _get_entry(graph_name)
    lg_config = {"configurable": {"thread_id": thread_id}}

    log.info("history.start", graph=graph_name, thread_id=thread_id, correlation_id=cid)

    try:
        async with get_async_checkpointer() as saver:
            compiled = entry.with_checkpointer(saver)
            history  = []
            async for snapshot in compiled.aget_state_history(lg_config):
                history.append({
                    "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
                    "next_nodes":    list(snapshot.next),
                    "values":        _safe_serialise(snapshot.values),
                    "created_at":    str(snapshot.created_at) if snapshot.created_at else None,
                })

    except Exception as exc:
        log.error("history.error", graph=graph_name, error=str(exc), correlation_id=cid)
        raise HTTPException(status_code=500, detail={"error": str(exc)})

    return {
        "graph":          graph_name,
        "thread_id":      thread_id,
        "history":        history,
        "correlation_id": cid,
    }