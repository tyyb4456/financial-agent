"""
tests/test_phase4_observability.py
------------------------------------
Tests for Phase 4: LangSmith tracing setup, structured logging, correlation
IDs, request middleware, lifespan startup, /ready endpoint, and deployment
config files.

Zero live LangSmith calls — tracing is mocked or disabled throughout.
"""

import os
import json
import uuid
import importlib
import pytest
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Observability — setup_tracing()
# ─────────────────────────────────────────────────────────────────────────────

class TestSetupTracing:
    def test_tracing_disabled_when_flag_false(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        import config.observability as obs
        importlib.reload(obs)
        # Patch settings to have tracing=False
        with patch("config.observability.settings") as mock_settings:
            mock_settings.langsmith_tracing = False
            mock_settings.langsmith_api_key = ""
            result = obs.setup_tracing()
        assert result is False

    def test_tracing_disabled_when_no_api_key(self, monkeypatch):
        import config.observability as obs
        with patch("config.observability.settings") as mock_settings:
            mock_settings.langsmith_tracing = True
            mock_settings.langsmith_api_key = ""   # empty = no key
            result = obs.setup_tracing()
        assert result is False

    def test_tracing_enabled_with_valid_credentials(self, monkeypatch):
        import config.observability as obs
        with patch("config.observability.settings") as mock_settings:
            mock_settings.langsmith_tracing = True
            mock_settings.langsmith_api_key = "ls-test-key-123"
            mock_settings.langsmith_project = "test-project"
            result = obs.setup_tracing()
        assert result is True
        assert os.environ.get("LANGSMITH_TRACING") == "true"
        assert os.environ.get("LANGSMITH_API_KEY") == "ls-test-key-123"
        assert os.environ.get("LANGSMITH_PROJECT") == "test-project"
        # Cleanup env so other tests aren't affected
        for k in ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]:
            os.environ.pop(k, None)

    def test_tracing_returns_bool(self):
        import config.observability as obs
        with patch("config.observability.settings") as mock_settings:
            mock_settings.langsmith_tracing = False
            mock_settings.langsmith_api_key = ""
            result = obs.setup_tracing()
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────────────────
# Observability — trace_graph_run()
# ─────────────────────────────────────────────────────────────────────────────

class TestTraceGraphRun:
    @pytest.mark.asyncio
    async def test_yields_config_with_thread_id(self):
        from config.observability import trace_graph_run
        async with trace_graph_run("financial_reporter", "thread-xyz", {"symbol": "AAPL"}) as cfg:
            assert cfg["configurable"]["thread_id"] == "thread-xyz"

    @pytest.mark.asyncio
    async def test_no_op_when_tracing_disabled(self):
        from config.observability import trace_graph_run
        with patch("config.observability.settings") as mock_settings:
            mock_settings.langsmith_tracing = False
            mock_settings.langsmith_api_key = ""
            async with trace_graph_run("news_reporter", "t1", {"topic": "AI"}) as cfg:
                # Still yields a valid config (no-op, not None)
                assert "configurable" in cfg
                # No metadata when tracing is off
                assert "metadata" not in cfg

    @pytest.mark.asyncio
    async def test_metadata_injected_when_tracing_enabled(self):
        from config.observability import trace_graph_run
        with patch("config.observability.settings") as mock_settings:
            mock_settings.langsmith_tracing = True
            mock_settings.langsmith_api_key = "ls-test-key"
            async with trace_graph_run("medical_researcher", "t2", {"term": "cancer"}) as cfg:
                assert "metadata" in cfg
                meta = cfg["metadata"]
                assert meta["graph_name"]  == "medical_researcher"
                assert meta["thread_id"]   == "t2"
                assert "input_keys"        in meta

    @pytest.mark.asyncio
    async def test_tags_injected_when_tracing_enabled(self):
        from config.observability import trace_graph_run
        with patch("config.observability.settings") as mock_settings:
            mock_settings.langsmith_tracing = True
            mock_settings.langsmith_api_key = "ls-test-key"
            async with trace_graph_run("news_reporter", "t3", {"topic": "space"}) as cfg:
                assert "tags" in cfg
                assert any("graph:news_reporter" in t for t in cfg["tags"])

    @pytest.mark.asyncio
    async def test_input_keys_not_values_in_metadata(self):
        """Sensitive input values should not be logged, only key names."""
        from config.observability import trace_graph_run
        with patch("config.observability.settings") as mock_settings:
            mock_settings.langsmith_tracing = True
            mock_settings.langsmith_api_key = "ls-test-key"
            secret_input = {"symbol": "SUPER_SECRET_TICKER", "api_key": "should-not-appear"}
            async with trace_graph_run("financial_reporter", "t4", secret_input) as cfg:
                meta_str = json.dumps(cfg.get("metadata", {}))
                # Key names appear, but not the secret values
                assert "symbol"             in meta_str
                assert "SUPER_SECRET_TICKER" not in meta_str
                assert "should-not-appear"   not in meta_str


# ─────────────────────────────────────────────────────────────────────────────
# Structured logging
# ─────────────────────────────────────────────────────────────────────────────

class TestStructuredLogging:
    def test_configure_logging_console_mode(self, monkeypatch):
        monkeypatch.setenv("LOG_FORMAT", "console")
        import config.logging as cl
        importlib.reload(cl)
        # Should not raise
        cl.configure_logging()

    def test_configure_logging_json_mode(self, monkeypatch):
        monkeypatch.setenv("LOG_FORMAT", "json")
        import config.logging as cl
        importlib.reload(cl)
        cl.configure_logging()
        # Restore
        monkeypatch.setenv("LOG_FORMAT", "console")

    def test_logger_is_usable_after_configure(self):
        import structlog
        from config.logging import configure_logging
        configure_logging()
        log = structlog.get_logger("test")
        # Should not raise
        log.info("test.message", key="value")

    def test_log_level_env_respected(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        # Patch settings.log_level directly — don't try to reload the singleton
        import config.logging as cl
        with patch("config.logging.settings") as mock_settings:
            mock_settings.log_level = "WARNING"
            cl.configure_logging()
        import structlog
        log = structlog.get_logger("test")
        log.warning("test.warn")
        # Restore
        monkeypatch.setenv("LOG_LEVEL", "INFO")


# ─────────────────────────────────────────────────────────────────────────────
# Request middleware — correlation IDs
# ─────────────────────────────────────────────────────────────────────────────

class TestRequestMiddleware:
    @pytest.mark.asyncio
    async def test_x_correlation_id_header_present(self, client):
        r = await client.get("/")
        assert "x-correlation-id" in r.headers

    @pytest.mark.asyncio
    async def test_correlation_id_is_valid_uuid(self, client):
        r = await client.get("/api/v1/graphs")
        cid = r.headers.get("x-correlation-id", "")
        assert len(cid) > 0
        # Should be parseable as UUID
        uuid.UUID(cid)

    @pytest.mark.asyncio
    async def test_client_provided_correlation_id_echoed(self, client):
        custom_cid = str(uuid.uuid4())
        r = await client.get("/", headers={"X-Correlation-ID": custom_cid})
        assert r.headers.get("x-correlation-id") == custom_cid

    @pytest.mark.asyncio
    async def test_correlation_id_different_per_request(self, client):
        r1 = await client.get("/")
        r2 = await client.get("/")
        cid1 = r1.headers.get("x-correlation-id")
        cid2 = r2.headers.get("x-correlation-id")
        assert cid1 != cid2

    @pytest.mark.asyncio
    async def test_run_response_includes_correlation_id(self, client, mock_tools, mock_llm):
        r = await client.post("/api/v1/run/news_reporter",
                              json={"inputs": {"topic": "AI"}})
        assert r.status_code == 200
        body = r.json()
        assert "correlation_id" in body
        uuid.UUID(body["correlation_id"])   # must be valid UUID

    @pytest.mark.asyncio
    async def test_stream_done_event_has_correlation_id(self, client, mock_tools, mock_llm):
        r = await client.post("/api/v1/stream/news_reporter",
                              json={"inputs": {"topic": "AI"}})
        events = [json.loads(line[6:]) for line in r.text.splitlines()
                  if line.startswith("data: ")]
        done_events = [e for e in events if e.get("type") == "done"]
        assert done_events
        uuid.UUID(done_events[0]["correlation_id"])

    @pytest.mark.asyncio
    async def test_history_response_includes_correlation_id(self, client):
        r = await client.get("/api/v1/history/news_reporter?thread_id=test-cid")
        assert r.status_code == 200
        assert "correlation_id" in r.json()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan + startup
# ─────────────────────────────────────────────────────────────────────────────

class TestLifespan:
    @pytest.mark.asyncio
    async def test_app_starts_and_serves_requests(self, client):
        """If lifespan fails, the client fixture would error before this runs."""
        r = await client.get("/")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_ready_endpoint_returns_200(self, client):
        r = await client.get("/ready")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_ready_endpoint_lists_graphs(self, client):
        r = await client.get("/ready")
        body = r.json()
        assert body["status"] == "ready"
        assert "graphs" in body
        assert len(body["graphs"]) == 6

    @pytest.mark.asyncio
    async def test_ready_graphs_match_registry(self, client):
        from registry import get_registry
        expected = set(get_registry().keys())
        r = await client.get("/ready")
        actual = set(r.json()["graphs"])
        assert actual == expected

    @pytest.mark.asyncio
    async def test_docs_endpoint_available(self, client):
        r = await client.get("/docs")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_openapi_schema_has_all_routes(self, client):
        r = await client.get("/openapi.json")
        assert r.status_code == 200
        paths = set(r.json()["paths"].keys())
        assert "/api/v1/graphs"              in paths
        assert "/api/v1/run/{graph_name}"    in paths
        assert "/api/v1/stream/{graph_name}" in paths
        assert "/api/v1/history/{graph_name}" in paths
        assert "/ready"                      in paths


# ─────────────────────────────────────────────────────────────────────────────
# Deployment files
# ─────────────────────────────────────────────────────────────────────────────

class TestDeploymentFiles:
    BASE = os.path.dirname(os.path.dirname(__file__))

    def _path(self, *parts):
        return os.path.join(self.BASE, *parts)

    def test_langgraph_json_exists(self):
        assert os.path.isfile(self._path("langgraph.json"))

    def test_langgraph_json_is_valid(self):
        with open(self._path("langgraph.json")) as f:
            data = json.load(f)
        assert "graphs" in data
        assert "dependencies" in data

    def test_langgraph_json_has_all_six_graphs(self):
        with open(self._path("langgraph.json")) as f:
            data = json.load(f)
        graphs = data["graphs"]
        expected = {
            "financial_reporter", "investment_advisor", "financial_query",
            "trending_posts",     "news_reporter",      "medical_researcher",
        }
        assert set(graphs.keys()) == expected

    def test_langgraph_json_graph_paths_reference_real_files(self):
        with open(self._path("langgraph.json")) as f:
            data = json.load(f)
        for name, path_ref in data["graphs"].items():
            file_path = path_ref.split(":")[0]    # strip :variable_name
            full_path = self._path(file_path.lstrip("./"))
            assert os.path.isfile(full_path), \
                f"langgraph.json references missing file: {file_path} for {name}"

    def test_dockerfile_exists(self):
        assert os.path.isfile(self._path("Dockerfile"))

    def test_dockerfile_has_healthcheck(self):
        content = open(self._path("Dockerfile")).read()
        assert "HEALTHCHECK" in content

    def test_dockerfile_has_non_root_user(self):
        content = open(self._path("Dockerfile")).read()
        assert "USER appuser" in content

    def test_dockerfile_exposes_port_8000(self):
        content = open(self._path("Dockerfile")).read()
        assert "EXPOSE 8000" in content

    def test_dockerignore_excludes_tests(self):
        content = open(self._path(".dockerignore")).read()
        assert "tests/" in content

    def test_dockerignore_excludes_env(self):
        content = open(self._path(".dockerignore")).read()
        assert ".env" in content

    def test_env_example_has_all_required_vars(self):
        content = open(self._path(".env.example")).read()
        required = [
            "GOOGLE_API_KEY",
            "SERPER_API_KEY",
            "REDDIT_CLIENT_ID",
            "LANGSMITH_TRACING",
            "LANGSMITH_API_KEY",
            "ENVIRONMENT",
            "LOG_FORMAT",
            "APP_VERSION",
        ]
        for var in required:
            assert var in content, f".env.example missing: {var}"

    def test_pyproject_has_langsmith_dep(self):
        content = open(self._path("pyproject.toml")).read()
        assert "langsmith" in content

    def test_pyproject_has_aiosqlite_dep(self):
        content = open(self._path("pyproject.toml")).read()
        assert "aiosqlite" in content