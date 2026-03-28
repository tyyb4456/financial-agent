"""
API router — graph execution
-----------------------------
Endpoints:
  POST /run/{graph_name}     → invoke a graph, return full result
  POST /stream/{graph_name}  → stream graph output via SSE (Server-Sent Events)
  GET  /graphs               → list available graphs + their input schemas
"""

from __future__ import annotations

import json
import structlog
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

from registry import get_registry

log = structlog.get_logger(__name__)
router = APIRouter()


# ── Request model ─────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    inputs: dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_entry(graph_name: str):
    """Resolve graph name → registry entry, raise 404 if missing."""
    registry = get_registry()
    entry = registry.get(graph_name)
    if entry is None:
        available = list(registry.keys())
        raise HTTPException(
            status_code=404,
            detail={"error": f"Graph '{graph_name}' not found.", "available": available},
        )
    return entry


def _validate_inputs(entry, raw_inputs: dict) -> dict:
    """Validate raw inputs against the graph's Pydantic schema."""
    try:
        validated = entry.schema(**raw_inputs)
        return validated.model_dump()
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "Input validation failed.", "detail": exc.errors()},
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/graphs", summary="List available graphs")
async def list_graphs():
    """Return all registered graphs with their input schemas."""
    registry = get_registry()
    return {
        name: {
            "description": entry.description,
            "input_schema": entry.schema.model_json_schema(),
        }
        for name, entry in registry.items()
    }


@router.post("/run/{graph_name}", summary="Run a graph synchronously")
async def run_graph(graph_name: str, request: RunRequest):
    """
    Invoke a graph and wait for the full result.
    Returns: { "graph": str, "result": str }
    """
    entry = _get_entry(graph_name)
    inputs = _validate_inputs(entry, request.inputs)

    log.info("graph.run.start", graph=graph_name, inputs=inputs)

    try:
        # version="v2" → returns GraphOutput with .value dict (LangGraph 1.1+)
        output = await entry.graph.ainvoke(inputs, version="v2")
        # GraphOutput.value is the final state dict
        state = output.value if hasattr(output, "value") else output
        result = state.get("result", str(state))
    except Exception as exc:
        log.error("graph.run.error", graph=graph_name, error=str(exc))
        raise HTTPException(status_code=500, detail={"error": str(exc)})

    log.info("graph.run.done", graph=graph_name)
    return {"graph": graph_name, "result": result}


@router.post("/stream/{graph_name}", summary="Stream a graph via SSE")
async def stream_graph(graph_name: str, request: RunRequest):
    """
    Stream graph execution via Server-Sent Events.
    Each SSE event carries a JSON payload:
      { "type": "node_update" | "done" | "error", "data": ... }
    """
    entry = _get_entry(graph_name)
    inputs = _validate_inputs(entry, request.inputs)

    log.info("graph.stream.start", graph=graph_name, inputs=inputs)

    async def event_generator():
        try:
            # version="v2" → unified StreamPart with .type / .ns / .data
            async for chunk in entry.graph.astream(inputs, version="v2"):
                payload = json.dumps({"type": chunk.type, "data": chunk.data})
                yield f"data: {payload}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as exc:
            log.error("graph.stream.error", graph=graph_name, error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'data': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )