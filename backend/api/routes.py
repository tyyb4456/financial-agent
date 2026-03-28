"""
api/routes.py
-------------
FastAPI router — graph execution endpoints.

Endpoints:
  GET  /graphs                     → list all graphs + input schemas
  POST /run/{graph_name}           → sync invoke, returns full result
  POST /stream/{graph_name}        → SSE streaming with node-level updates + LLM tokens
  GET  /history/{graph_name}       → checkpoint history for a thread (requires thread_id)
"""

from __future__ import annotations

import json
import uuid
import structlog
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from config.checkpointer import get_async_checkpointer
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
    graph:     str
    thread_id: str
    result:    str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_entry(graph_name: str):
    registry = get_registry()
    entry    = registry.get(graph_name)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Graph '{graph_name}' not found.", "available": list(registry.keys())},
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


def _lg_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _safe_serialise(obj: Any) -> Any:
    """Make graph state JSON-safe. Truncates large strings to keep SSE payloads small."""
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

    Every super-step is checkpointed to SQLite. Pass the same thread_id
    on a subsequent call to resume or re-inspect a previous run.
    """
    entry  = _get_entry(graph_name)
    inputs = _validate(entry, request.inputs)
    config = _lg_config(request.thread_id)

    log.info("graph.run.start", graph=graph_name, thread_id=request.thread_id)

    try:
        async with get_async_checkpointer() as saver:
            compiled = entry.with_checkpointer(saver)
            output   = await compiled.ainvoke(inputs, config=config)
            result   = output.get("result", str(output))

    except Exception as exc:
        log.error("graph.run.error", graph=graph_name, error=str(exc))
        raise HTTPException(status_code=500, detail={"error": str(exc)})

    log.info("graph.run.done", graph=graph_name, thread_id=request.thread_id)
    return RunResponse(graph=graph_name, thread_id=request.thread_id, result=result)


@router.post("/stream/{graph_name}", summary="Stream a graph via SSE")
async def stream_graph(graph_name: str, request: RunRequest):
    """
    Stream graph execution as Server-Sent Events (SSE).

    Each event is a JSON object. Event types:
      node_update  → { type, node, data }    — node finished, state diff
      llm_token    → { type, token }         — streamed LLM token
      done         → { type, thread_id }     — graph finished
      error        → { type, error }         — unrecoverable error
    """
    entry  = _get_entry(graph_name)
    inputs = _validate(entry, request.inputs)
    config = _lg_config(request.thread_id)

    log.info("graph.stream.start", graph=graph_name, thread_id=request.thread_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        def _emit(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"

        try:
            # Bare graph (no checkpointer) for streaming — avoids SQLite context
            # manager complexity inside an async generator.
            graph = entry.graph

            async for chunk in graph.astream(
                inputs,
                config=config,
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

            yield _emit({"type": "done", "thread_id": request.thread_id})

        except Exception as exc:
            log.error("graph.stream.error", graph=graph_name, error=str(exc))
            yield _emit({"type": "error", "error": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


@router.get("/history/{graph_name}", summary="Get checkpoint history for a thread")
async def get_history(graph_name: str, thread_id: str):
    """
    Return all saved checkpoints for a graph + thread_id pair.

    Useful for debugging, audit trails, and time-travel.
    Requires that /run was previously called with the same thread_id.
    """
    entry  = _get_entry(graph_name)
    config = _lg_config(thread_id)

    try:
        async with get_async_checkpointer() as saver:
            compiled = entry.with_checkpointer(saver)
            history  = []
            async for snapshot in compiled.aget_state_history(config):
                history.append({
                    "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
                    "next_nodes":    list(snapshot.next),
                    "values":        _safe_serialise(snapshot.values),
                    "created_at":    str(snapshot.created_at) if snapshot.created_at else None,
                })

        return {"graph": graph_name, "thread_id": thread_id, "history": history}

    except Exception as exc:
        log.error("history.error", graph=graph_name, error=str(exc))
        raise HTTPException(status_code=500, detail={"error": str(exc)})