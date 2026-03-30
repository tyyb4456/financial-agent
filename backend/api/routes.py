"""
api/routes.py
-------------
FastAPI router — graph execution endpoints.

Architecture:
  Each graph has TWO dedicated endpoints:
    POST /financial/report          → FinancialReporterInput body
    POST /financial/report/stream   → SSE streaming version
    ... one pair per graph ...

  Plus generic fallbacks:
    POST /run/{graph_name}          → RunRequest { inputs: dict }
    POST /stream/{graph_name}       → SSE version
    GET  /graphs                    → list all graphs
    GET  /history/{graph_name}      → checkpoint history

Why dedicated endpoints?
  - Clean, self-documenting Swagger UI — no "inputs: {}" guesswork
  - Each endpoint has its own typed Pydantic request model
  - Frontend knows exactly what fields to send per workflow
  - The generic endpoint is still there for programmatic/dynamic use
"""

from __future__ import annotations

import json
import uuid
import structlog
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError, ConfigDict

from config.checkpointer  import get_async_checkpointer
from config.observability  import trace_graph_run
from registry import get_registry

log    = structlog.get_logger(__name__)
router = APIRouter()


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def _cid() -> str:
    return str(uuid.uuid4())


def _get_entry(graph_name: str):
    registry = get_registry()
    entry    = registry.get(graph_name)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Graph '{graph_name}' not found.", "available": list(registry.keys())},
        )
    return entry


def _safe_serialise(obj: Any) -> Any:
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


async def _invoke(graph_name: str, inputs: dict, thread_id: str) -> dict:
    """
    Core invoke logic shared by all dedicated + generic endpoints.
    Handles checkpointing, tracing, and error translation.
    """
    cid   = _cid()
    entry = _get_entry(graph_name)

    structlog.contextvars.bind_contextvars(
        correlation_id=cid,
        graph=graph_name,
        thread_id=thread_id,
    )
    log.info("graph.run.start", input_keys=list(inputs.keys()))

    try:
        async with trace_graph_run(graph_name, thread_id, inputs) as lg_config:
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
    return {"graph": graph_name, "thread_id": thread_id, "result": result, "correlation_id": cid}


def _sse_response(graph_name: str, inputs: dict, thread_id: str) -> StreamingResponse:
    """Build a StreamingResponse for any graph. Shared by all stream endpoints."""
    cid   = _cid()
    entry = _get_entry(graph_name)

    log.info("graph.stream.start", graph=graph_name, thread_id=thread_id, correlation_id=cid)

    async def gen() -> AsyncGenerator[str, None]:
        def _emit(p: dict) -> str:
            return f"data: {json.dumps(p)}\n\n"

        try:
            async with trace_graph_run(graph_name, thread_id, inputs) as lg_config:
                async for chunk in entry.graph.astream(
                    inputs,
                    config=lg_config,
                    stream_mode=["updates", "messages"],
                    version="v2",
                ):
                    if chunk.get("type") == "updates":
                        for node, diff in chunk["data"].items():
                            if not node.startswith("__"):
                                yield _emit({"type": "node_update", "node": node,
                                             "data": _safe_serialise(diff)})
                    elif chunk.get("type") == "messages":
                        msg_chunk, _ = chunk["data"]
                        token = getattr(msg_chunk, "content", "")
                        if token:
                            yield _emit({"type": "llm_token", "token": token})

            yield _emit({"type": "done", "thread_id": thread_id, "correlation_id": cid})

        except Exception as exc:
            log.error("graph.stream.error", graph=graph_name, error=str(exc))
            yield _emit({"type": "error", "error": str(exc), "correlation_id": cid})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
            "X-Correlation-ID":  cid,
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# Typed request/response models — one per graph
# ═════════════════════════════════════════════════════════════════════════════

class _ThreadId(BaseModel):
    """Mixin: optional thread_id, auto-generated if omitted."""
    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Thread ID for checkpointing. Auto-generated if omitted.",
    )


class FinancialReportRequest(_ThreadId):
    model_config = ConfigDict(json_schema_extra={
        "example": {"symbol": "AAPL", "thread_id": "my-thread-001"}
    })
    symbol: str = Field(description="Stock ticker symbol, e.g. 'AAPL'.")


class InvestmentAdvisorRequest(_ThreadId):
    model_config = ConfigDict(json_schema_extra={
        "example": {"symbol": "MSFT", "thread_id": "my-thread-002"}
    })
    symbol: str = Field(description="Stock ticker symbol, e.g. 'MSFT'.")


class FinancialQueryRequest(_ThreadId):
    model_config = ConfigDict(json_schema_extra={
        "example": {"symbol": "GOOG", "query": "What is the P/E ratio?", "thread_id": "my-thread-003"}
    })
    symbol: str = Field(description="Stock ticker symbol.")
    query:  str = Field(description="Your specific financial question about this stock.")


class TrendingPostsRequest(_ThreadId):
    model_config = ConfigDict(json_schema_extra={
        "example": {"subreddit_name": "investing", "thread_id": "my-thread-004"}
    })
    subreddit_name: str = Field(description="Subreddit name, e.g. 'investing' or 'technology'.")


class NewsReportRequest(_ThreadId):
    model_config = ConfigDict(json_schema_extra={
        "example": {"topic": "AI regulation 2025", "thread_id": "my-thread-005"}
    })
    topic: str = Field(description="News topic to report on, e.g. 'AI regulation 2025'.")


class MedicalResearchRequest(_ThreadId):
    model_config = ConfigDict(json_schema_extra={
        "example": {"term": "type 2 diabetes", "query": "What are the latest treatments?", "thread_id": "my-thread-006"}
    })
    term:  str = Field(description="Medical search term, e.g. 'type 2 diabetes'.")
    query: str = Field(default="", description="Optional specific question. If omitted, returns a general summary.")


class GraphRunResponse(BaseModel):
    graph:          str
    thread_id:      str
    result:         str
    correlation_id: str


# ═════════════════════════════════════════════════════════════════════════════
# Dedicated endpoints — Financial
# ═════════════════════════════════════════════════════════════════════════════

@router.post(
    "/financial/report",
    response_model=GraphRunResponse,
    summary="Generate a financial report",
    tags=["Financial"],
)
async def financial_report(req: FinancialReportRequest):
    """
    Fetch Yahoo Finance data for a stock symbol and generate a structured
    financial report covering revenue, market cap, P/E, 52-week range,
    and current price insights.
    """
    return await _invoke("financial_reporter", {"symbol": req.symbol}, req.thread_id)


@router.post(
    "/financial/report/stream",
    summary="Stream a financial report (SSE)",
    tags=["Financial"],
)
async def financial_report_stream(req: FinancialReportRequest):
    """SSE streaming version of /financial/report."""
    return _sse_response("financial_reporter", {"symbol": req.symbol}, req.thread_id)


@router.post(
    "/financial/investment",
    response_model=GraphRunResponse,
    summary="Get investment recommendation",
    tags=["Financial"],
)
async def investment_advisor(req: InvestmentAdvisorRequest):
    """
    Analyse valuation metrics, earnings history, and analyst recommendations
    for a stock. Returns a structured Buy / Hold / Sell recommendation with
    bull/bear thesis and price target.
    """
    return await _invoke("investment_advisor", {"symbol": req.symbol}, req.thread_id)


@router.post(
    "/financial/investment/stream",
    summary="Stream investment recommendation (SSE)",
    tags=["Financial"],
)
async def investment_advisor_stream(req: InvestmentAdvisorRequest):
    """SSE streaming version of /financial/investment."""
    return _sse_response("investment_advisor", {"symbol": req.symbol}, req.thread_id)


@router.post(
    "/financial/query",
    response_model=GraphRunResponse,
    summary="Ask a financial question about a stock",
    tags=["Financial"],
)
async def financial_query(req: FinancialQueryRequest):
    """
    Ask any specific financial question about a stock (e.g. 'What is the P/E ratio?',
    'How does the debt-to-equity compare to sector average?').
    The answer is grounded in live Yahoo Finance data.
    """
    return await _invoke("financial_query", {"symbol": req.symbol, "query": req.query}, req.thread_id)


@router.post(
    "/financial/query/stream",
    summary="Stream financial Q&A (SSE)",
    tags=["Financial"],
)
async def financial_query_stream(req: FinancialQueryRequest):
    """SSE streaming version of /financial/query."""
    return _sse_response("financial_query", {"symbol": req.symbol, "query": req.query}, req.thread_id)


# ═════════════════════════════════════════════════════════════════════════════
# Dedicated endpoints — Reddit
# ═════════════════════════════════════════════════════════════════════════════

@router.post(
    "/reddit/trending",
    response_model=GraphRunResponse,
    summary="Analyse trending Reddit posts",
    tags=["Reddit"],
)
async def trending_posts(req: TrendingPostsRequest):
    """
    Fetch the top hot posts from a subreddit, run VADER sentiment analysis
    on each title, and return a structured report with an overview, top posts
    table, sentiment summary, and key community takeaways.
    """
    return await _invoke("trending_posts", {"subreddit_name": req.subreddit_name}, req.thread_id)


@router.post(
    "/reddit/trending/stream",
    summary="Stream Reddit trending analysis (SSE)",
    tags=["Reddit"],
)
async def trending_posts_stream(req: TrendingPostsRequest):
    """SSE streaming version of /reddit/trending."""
    return _sse_response("trending_posts", {"subreddit_name": req.subreddit_name}, req.thread_id)


# ═════════════════════════════════════════════════════════════════════════════
# Dedicated endpoints — News
# ═════════════════════════════════════════════════════════════════════════════

@router.post(
    "/news/report",
    response_model=GraphRunResponse,
    summary="Generate a news report on any topic",
    tags=["News"],
)
async def news_report(req: NewsReportRequest):
    """
    Search for recent news articles on any topic using Serper (Google News),
    then synthesise a markdown news report with headline summary, key
    developments, context, and source references.
    """
    return await _invoke("news_reporter", {"topic": req.topic}, req.thread_id)


@router.post(
    "/news/report/stream",
    summary="Stream news report (SSE)",
    tags=["News"],
)
async def news_report_stream(req: NewsReportRequest):
    """SSE streaming version of /news/report."""
    return _sse_response("news_reporter", {"topic": req.topic}, req.thread_id)


# ═════════════════════════════════════════════════════════════════════════════
# Dedicated endpoints — Medical
# ═════════════════════════════════════════════════════════════════════════════

@router.post(
    "/medical/research",
    response_model=GraphRunResponse,
    summary="Retrieve and summarise medical research",
    tags=["Medical"],
)
async def medical_research(req: MedicalResearchRequest):
    """
    Search PubMed for articles matching a medical term and either:
    - **General summary** (if `query` is empty): key findings, clinical implications, recommendations.
    - **Specific Q&A** (if `query` is provided): targeted answer grounded in the evidence.
    """
    return await _invoke(
        "medical_researcher",
        {"term": req.term, "query": req.query},
        req.thread_id,
    )


@router.post(
    "/medical/research/stream",
    summary="Stream medical research (SSE)",
    tags=["Medical"],
)
async def medical_research_stream(req: MedicalResearchRequest):
    """SSE streaming version of /medical/research."""
    return _sse_response(
        "medical_researcher",
        {"term": req.term, "query": req.query},
        req.thread_id,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Generic fallback endpoints (for programmatic / dynamic use)
# ═════════════════════════════════════════════════════════════════════════════

class RunRequest(BaseModel):
    inputs:    dict[str, Any]
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_config = ConfigDict(json_schema_extra={
        "example": {"inputs": {"symbol": "AAPL"}, "thread_id": "my-thread-001"}
    })


@router.get("/graphs", summary="List all available graphs", tags=["Meta"])
async def list_graphs():
    """Return all registered graphs with descriptions and typed input schemas."""
    registry = get_registry()
    return {
        name: {
            "description":  entry.description,
            "input_schema": entry.schema.model_json_schema(),
        }
        for name, entry in registry.items()
    }


@router.post(
    "/run/{graph_name}",
    response_model=GraphRunResponse,
    summary="Run any graph by name (generic)",
    tags=["Generic"],
)
async def run_graph(graph_name: str, request: RunRequest):
    """
    Generic endpoint — invoke any graph by name with a raw inputs dict.
    Prefer the dedicated typed endpoints above for normal use.
    """
    entry = _get_entry(graph_name)
    try:
        validated = entry.schema(**request.inputs).model_dump()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"error": "Input validation failed.", "detail": exc.errors()})

    return await _invoke(graph_name, validated, request.thread_id)


@router.post(
    "/stream/{graph_name}",
    summary="Stream any graph by name (generic SSE)",
    tags=["Generic"],
)
async def stream_graph(graph_name: str, request: RunRequest):
    """Generic SSE streaming endpoint. Prefer dedicated /*/stream endpoints above."""
    entry = _get_entry(graph_name)
    try:
        validated = entry.schema(**request.inputs).model_dump()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"error": "Input validation failed.", "detail": exc.errors()})

    return _sse_response(graph_name, validated, request.thread_id)


@router.get(
    "/history/{graph_name}",
    summary="Get checkpoint history for a thread",
    tags=["Meta"],
)
async def get_history(graph_name: str, thread_id: str):
    """Return all saved checkpoints for a graph + thread_id pair."""
    cid   = _cid()
    entry = _get_entry(graph_name)
    lg_config = {"configurable": {"thread_id": thread_id}}

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
        log.error("history.error", graph=graph_name, error=str(exc))
        raise HTTPException(status_code=500, detail={"error": str(exc)})

    return {"graph": graph_name, "thread_id": thread_id, "history": history, "correlation_id": cid}