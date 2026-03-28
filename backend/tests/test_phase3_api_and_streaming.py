"""
tests/test_phase3_api_and_streaming.py
---------------------------------------
Tests for Phase 3: FastAPI endpoints, SSE streaming format,
checkpointing lifecycle, and history endpoint.

All external I/O is mocked via conftest fixtures.
Uses httpx AsyncClient for async endpoint testing.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _parse_sse(raw_text: str) -> list[dict]:
    """Parse a raw SSE response body into a list of event dicts."""
    events = []
    for line in raw_text.strip().splitlines():
        if line.startswith("data: "):
            payload = line[len("data: "):]
            events.append(json.loads(payload))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# GET /graphs
# ─────────────────────────────────────────────────────────────────────────────

class TestListGraphs:
    @pytest.mark.asyncio
    async def test_returns_200(self, client):
        r = await client.get("/api/v1/graphs")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_returns_six_graphs(self, client):
        r = await client.get("/api/v1/graphs")
        data = r.json()
        assert len(data) == 6

    @pytest.mark.asyncio
    async def test_each_entry_has_description_and_schema(self, client):
        r = await client.get("/api/v1/graphs")
        for name, info in r.json().items():
            assert "description"  in info, f"{name} missing description"
            assert "input_schema" in info, f"{name} missing input_schema"

    @pytest.mark.asyncio
    async def test_financial_reporter_schema_has_symbol(self, client):
        r = await client.get("/api/v1/graphs")
        props = r.json()["financial_reporter"]["input_schema"]["properties"]
        assert "symbol" in props

    @pytest.mark.asyncio
    async def test_financial_query_schema_has_symbol_and_query(self, client):
        r = await client.get("/api/v1/graphs")
        props = r.json()["financial_query"]["input_schema"]["properties"]
        assert "symbol" in props
        assert "query"  in props

    @pytest.mark.asyncio
    async def test_medical_researcher_schema_has_term(self, client):
        r = await client.get("/api/v1/graphs")
        props = r.json()["medical_researcher"]["input_schema"]["properties"]
        assert "term" in props


# ─────────────────────────────────────────────────────────────────────────────
# POST /run/{graph_name} — error cases (no mocking needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunGraphErrors:
    @pytest.mark.asyncio
    async def test_unknown_graph_returns_404(self, client):
        r = await client.post("/api/v1/run/does_not_exist", json={"inputs": {}})
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_404_body_contains_available_graphs(self, client):
        r = await client.post("/api/v1/run/does_not_exist", json={"inputs": {}})
        body = r.json()
        assert "available" in body["detail"]
        assert len(body["detail"]["available"]) == 6

    @pytest.mark.asyncio
    async def test_missing_required_input_returns_422(self, client):
        # financial_reporter requires 'symbol'
        r = await client.post("/api/v1/run/financial_reporter", json={"inputs": {}})
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_422_body_has_error_and_detail(self, client):
        r = await client.post("/api/v1/run/financial_reporter", json={"inputs": {}})
        body = r.json()
        assert "error"  in body["detail"]
        assert "detail" in body["detail"]

    @pytest.mark.asyncio
    async def test_missing_query_for_financial_query_returns_422(self, client):
        r = await client.post("/api/v1/run/financial_query",
                              json={"inputs": {"symbol": "AAPL"}})
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_malformed_json_returns_422(self, client):
        r = await client.post("/api/v1/run/news_reporter",
                              content=b"not-json",
                              headers={"Content-Type": "application/json"})
        assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# POST /run/{graph_name} — success cases (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunGraphSuccess:
    @pytest.mark.asyncio
    async def test_financial_reporter_returns_200(self, client, mock_tools, mock_llm):
        r = await client.post("/api/v1/run/financial_reporter",
                              json={"inputs": {"symbol": "AAPL"}})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_response_has_graph_thread_id_result(self, client, mock_tools, mock_llm):
        r = await client.post("/api/v1/run/financial_reporter",
                              json={"inputs": {"symbol": "AAPL"}})
        body = r.json()
        assert "graph"     in body
        assert "thread_id" in body
        assert "result"    in body

    @pytest.mark.asyncio
    async def test_graph_name_echoed_in_response(self, client, mock_tools, mock_llm):
        r = await client.post("/api/v1/run/financial_reporter",
                              json={"inputs": {"symbol": "AAPL"}})
        assert r.json()["graph"] == "financial_reporter"

    @pytest.mark.asyncio
    async def test_result_is_non_empty_string(self, client, mock_tools, mock_llm):
        r = await client.post("/api/v1/run/news_reporter",
                              json={"inputs": {"topic": "AI trends"}})
        assert r.status_code == 200
        assert len(r.json()["result"]) > 0

    @pytest.mark.asyncio
    async def test_custom_thread_id_echoed(self, client, mock_tools, mock_llm):
        tid = "my-custom-thread-001"
        r = await client.post("/api/v1/run/financial_reporter",
                              json={"inputs": {"symbol": "AAPL"}, "thread_id": tid})
        assert r.json()["thread_id"] == tid

    @pytest.mark.asyncio
    async def test_auto_thread_id_generated_when_omitted(self, client, mock_tools, mock_llm):
        r = await client.post("/api/v1/run/news_reporter",
                              json={"inputs": {"topic": "space"}})
        assert "thread_id" in r.json()
        assert len(r.json()["thread_id"]) > 0

    @pytest.mark.asyncio
    async def test_medical_researcher_no_query(self, client, mock_tools, mock_llm):
        r = await client.post("/api/v1/run/medical_researcher",
                              json={"inputs": {"term": "diabetes"}})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_medical_researcher_with_query(self, client, mock_tools, mock_llm):
        r = await client.post("/api/v1/run/medical_researcher",
                              json={"inputs": {"term": "diabetes", "query": "Best treatment?"}})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_all_six_graphs_run_successfully(self, client, mock_tools, mock_llm):
        cases = [
            ("financial_reporter", {"symbol": "AAPL"}),
            ("investment_advisor", {"symbol": "GOOG"}),
            ("financial_query",    {"symbol": "TSLA", "query": "Revenue?"}),
            ("trending_posts",     {"subreddit_name": "investing"}),
            ("news_reporter",      {"topic": "AI"}),
            ("medical_researcher", {"term": "aspirin"}),
        ]
        for graph_name, inputs in cases:
            r = await client.post(f"/api/v1/run/{graph_name}",
                                  json={"inputs": inputs})
            assert r.status_code == 200, f"{graph_name} returned {r.status_code}: {r.text}"


# ─────────────────────────────────────────────────────────────────────────────
# POST /stream/{graph_name} — SSE format
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamGraph:
    @pytest.mark.asyncio
    async def test_unknown_graph_returns_404(self, client):
        r = await client.post("/api/v1/stream/no_such_graph",
                              json={"inputs": {}})
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_missing_input_returns_422(self, client):
        r = await client.post("/api/v1/stream/financial_reporter",
                              json={"inputs": {}})
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_stream_content_type_is_sse(self, client, mock_tools, mock_llm):
        r = await client.post("/api/v1/stream/news_reporter",
                              json={"inputs": {"topic": "AI"}})
        assert "text/event-stream" in r.headers["content-type"]

    @pytest.mark.asyncio
    async def test_stream_ends_with_done_event(self, client, mock_tools, mock_llm):
        r = await client.post("/api/v1/stream/news_reporter",
                              json={"inputs": {"topic": "AI"}})
        events = _parse_sse(r.text)
        assert events, "No SSE events received"
        last = events[-1]
        assert last["type"] == "done"

    @pytest.mark.asyncio
    async def test_done_event_has_thread_id(self, client, mock_tools, mock_llm):
        tid = "stream-test-thread"
        r = await client.post("/api/v1/stream/news_reporter",
                              json={"inputs": {"topic": "AI"}, "thread_id": tid})
        events = _parse_sse(r.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert done_events
        assert done_events[0]["thread_id"] == tid

    @pytest.mark.asyncio
    async def test_stream_contains_node_update_events(self, client, mock_tools, mock_llm):
        r = await client.post("/api/v1/stream/news_reporter",
                              json={"inputs": {"topic": "AI"}})
        events = _parse_sse(r.text)
        node_events = [e for e in events if e["type"] == "node_update"]
        assert len(node_events) > 0, "No node_update events in stream"

    @pytest.mark.asyncio
    async def test_node_update_has_node_and_data_keys(self, client, mock_tools, mock_llm):
        r = await client.post("/api/v1/stream/news_reporter",
                              json={"inputs": {"topic": "climate"}})
        events = _parse_sse(r.text)
        node_events = [e for e in events if e["type"] == "node_update"]
        for evt in node_events:
            assert "node" in evt, f"node_update missing 'node' key: {evt}"
            assert "data" in evt, f"node_update missing 'data' key: {evt}"

    @pytest.mark.asyncio
    async def test_every_sse_event_is_valid_json(self, client, mock_tools, mock_llm):
        r = await client.post("/api/v1/stream/financial_reporter",
                              json={"inputs": {"symbol": "AAPL"}})
        # _parse_sse will raise json.JSONDecodeError if any line is invalid
        events = _parse_sse(r.text)
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_error_event_on_graph_failure(self, client, mock_tools):
        """If the graph crashes, stream should emit an error event (not crash the server)."""
        # Make the LLM raise an exception
        bad_llm = MagicMock()
        bad_llm.invoke.side_effect = RuntimeError("LLM exploded")

        import graphs.news as news_mod
        original_llm = news_mod._llm
        news_mod._llm = bad_llm

        try:
            r = await client.post("/api/v1/stream/news_reporter",
                                  json={"inputs": {"topic": "test"}})
            events = _parse_sse(r.text)
            error_events = [e for e in events if e["type"] == "error"]
            assert len(error_events) > 0
        finally:
            news_mod._llm = original_llm


# ─────────────────────────────────────────────────────────────────────────────
# GET /history/{graph_name}
# ─────────────────────────────────────────────────────────────────────────────

class TestHistory:
    @pytest.mark.asyncio
    async def test_unknown_graph_returns_404(self, client):
        r = await client.get("/api/v1/history/no_such_graph?thread_id=abc")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_new_thread_returns_empty_history(self, client):
        r = await client.get("/api/v1/history/financial_reporter?thread_id=brand-new-thread-xyz")
        assert r.status_code == 200
        body = r.json()
        assert body["history"] == []

    @pytest.mark.asyncio
    async def test_history_response_shape(self, client):
        r = await client.get("/api/v1/history/news_reporter?thread_id=test-shape")
        body = r.json()
        assert "graph"     in body
        assert "thread_id" in body
        assert "history"   in body
        assert isinstance(body["history"], list)

    @pytest.mark.asyncio
    async def test_history_after_run_has_checkpoints(self, client, mock_tools, mock_llm):
        """
        InMemorySaver is ephemeral — each `async with get_async_checkpointer()`
        creates a fresh instance, so /run and /history cannot share state in tests.
        This test instead verifies the endpoint returns 200 and the correct shape.
        Persistence across calls is validated by the SqliteSaver integration test
        (see test_checkpointer_sqlite_persists_across_calls).
        """
        tid = "history-test-thread-001"
        await client.post("/api/v1/run/news_reporter",
                          json={"inputs": {"topic": "AI"}, "thread_id": tid})
        r = await client.get(f"/api/v1/history/news_reporter?thread_id={tid}")
        assert r.status_code == 200
        body = r.json()
        assert "history" in body
        assert isinstance(body["history"], list)  # empty is fine with InMemorySaver

    @pytest.mark.asyncio
    async def test_history_checkpoints_have_expected_keys(self, client, mock_tools, mock_llm):
        """
        Run graph and history sharing the SAME in-memory saver to test
        checkpoint shape when data is actually present.
        """
        from langgraph.checkpoint.memory import InMemorySaver
        from registry import get_registry

        saver = InMemorySaver()
        tid   = "shared-saver-test"
        entry = get_registry()["news_reporter"]

        compiled = entry.with_checkpointer(saver)
        await compiled.ainvoke(
            {"topic": "AI", "raw_articles": None, "result": None},
            config={"configurable": {"thread_id": tid}},
        )

        history = []
        async for snapshot in compiled.aget_state_history(
            {"configurable": {"thread_id": tid}}
        ):
            history.append(snapshot)

        assert len(history) > 0
        # Check the StateSnapshot shape
        snap = history[0]
        assert hasattr(snap, "values")
        assert hasattr(snap, "next")
        assert hasattr(snap, "config")

    @pytest.mark.asyncio
    async def test_different_threads_have_independent_history(self, client, mock_tools, mock_llm):
        """
        Verify thread isolation using a shared saver.
        Thread A (ran) has history; Thread B (never ran) has none.
        """
        from langgraph.checkpoint.memory import InMemorySaver
        from registry import get_registry

        saver   = InMemorySaver()
        entry   = get_registry()["news_reporter"]
        compiled = entry.with_checkpointer(saver)

        tid_a = "isolation-thread-A"
        tid_b = "isolation-thread-B"

        await compiled.ainvoke(
            {"topic": "AI", "raw_articles": None, "result": None},
            config={"configurable": {"thread_id": tid_a}},
        )

        history_a = [s async for s in compiled.aget_state_history({"configurable": {"thread_id": tid_a}})]
        history_b = [s async for s in compiled.aget_state_history({"configurable": {"thread_id": tid_b}})]

        assert len(history_a) > 0, "Thread A should have checkpoints"
        assert len(history_b) == 0, "Thread B should have no checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeSerialise:
    def test_truncates_long_strings(self):
        from api.routes import _safe_serialise
        long = "x" * 1000
        result = _safe_serialise(long)
        assert len(result) <= 514      # 500 + "…[truncated]"
        assert result.endswith("[truncated]")

    def test_short_string_unchanged(self):
        from api.routes import _safe_serialise
        assert _safe_serialise("hello") == "hello"

    def test_nested_dict_serialised(self):
        from api.routes import _safe_serialise
        data = {"a": {"b": "x" * 600}}
        result = _safe_serialise(data)
        assert result["a"]["b"].endswith("[truncated]")

    def test_list_serialised(self):
        from api.routes import _safe_serialise
        result = _safe_serialise([1, "hello", {"k": "v"}])
        assert result == [1, "hello", {"k": "v"}]

    def test_non_serialisable_object_becomes_string(self):
        from api.routes import _safe_serialise

        class Weird:
            def __repr__(self): return "WeirdObj"

        result = _safe_serialise(Weird())
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        r = await client.get("/")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_ok_status(self, client):
        r = await client.get("/")
        assert r.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_returns_version(self, client):
        r = await client.get("/")
        assert "version" in r.json()