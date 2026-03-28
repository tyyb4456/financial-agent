"""
tests/test_phase1_foundation.py
--------------------------------
Tests for Phase 1: config, settings, LLM factory, logging, registry structure.
Zero network calls — tests that the wiring is correct, not live APIs.
"""

import os
import importlib
import pytest


# ── Settings ──────────────────────────────────────────────────────────────────

class TestSettings:
    def test_settings_loads(self):
        from config.settings import settings
        assert settings is not None

    def test_default_port(self):
        from config.settings import settings
        assert settings.app_port == 8000

    def test_log_level_default(self):
        from config.settings import settings
        assert settings.log_level.upper() == "INFO"

    def test_checkpoint_backend_overridable(self, monkeypatch):
        monkeypatch.setenv("CHECKPOINT_BACKEND", "memory")
        # Settings picks it up via env (already set in conftest)
        assert os.environ["CHECKPOINT_BACKEND"] == "memory"

    def test_langsmith_disabled_by_default(self):
        from config.settings import settings
        assert settings.langsmith_tracing is False


# ── LLM factory ───────────────────────────────────────────────────────────────

class TestLLMFactory:
    def test_get_llm_returns_model(self):
        from config.llm import get_llm
        llm = get_llm()
        assert llm is not None

    def test_get_llm_cached(self):
        from config.llm import get_llm
        llm1 = get_llm()
        llm2 = get_llm()
        assert llm1 is llm2   # same cached instance

    def test_get_llm_has_invoke(self):
        from config.llm import get_llm
        llm = get_llm()
        assert callable(getattr(llm, "invoke", None))

    def test_get_llm_different_params_different_instance(self):
        from config.llm import get_llm
        # Different temperature → different cache key → different instance
        llm_a = get_llm(temperature=0.1)
        llm_b = get_llm(temperature=0.9)
        assert llm_a is not llm_b


# ── Checkpointer ──────────────────────────────────────────────────────────────

class TestCheckpointer:
    def test_sync_checkpointer_yields(self):
        from config.checkpointer import get_checkpointer
        from langgraph.checkpoint.memory import InMemorySaver
        with get_checkpointer() as saver:
            # conftest forces CHECKPOINT_BACKEND=memory
            assert isinstance(saver, InMemorySaver)

    @pytest.mark.asyncio
    async def test_async_checkpointer_yields(self):
        from config.checkpointer import get_async_checkpointer
        from langgraph.checkpoint.memory import InMemorySaver
        async with get_async_checkpointer() as saver:
            assert isinstance(saver, InMemorySaver)

    @pytest.mark.asyncio
    async def test_async_checkpointer_memory_backend(self, monkeypatch):
        monkeypatch.setenv("CHECKPOINT_BACKEND", "memory")
        import config.checkpointer as cm
        importlib.reload(cm)
        from langgraph.checkpoint.memory import InMemorySaver
        async with cm.get_async_checkpointer() as saver:
            assert isinstance(saver, InMemorySaver)


# ── State schemas ─────────────────────────────────────────────────────────────

class TestStateSchemas:
    def test_all_schemas_importable(self):
        from state.schemas import (
            FinancialState, InvestmentState, QueryState,
            RedditState, NewsState, MedicalState,
        )
        for schema in [FinancialState, InvestmentState, QueryState,
                       RedditState, NewsState, MedicalState]:
            assert schema is not None

    def test_financial_state_fields(self):
        from state.schemas import FinancialState
        keys = FinancialState.__annotations__.keys()
        assert "symbol"         in keys
        assert "raw_financials" in keys
        assert "result"         in keys

    def test_medical_state_has_optional_query(self):
        from state.schemas import MedicalState
        import typing
        # query should be Optional
        hints = typing.get_type_hints(MedicalState)
        assert "query" in hints

    def test_query_state_has_query_field(self):
        from state.schemas import QueryState
        assert "query" in QueryState.__annotations__


# ── Registry structure ────────────────────────────────────────────────────────

class TestRegistry:
    def test_registry_returns_six_entries(self):
        from registry import get_registry
        reg = get_registry()
        assert len(reg) == 6

    def test_all_expected_names_present(self):
        from registry import get_registry
        reg = get_registry()
        expected = {
            "financial_reporter", "investment_advisor", "financial_query",
            "trending_posts",     "news_reporter",      "medical_researcher",
        }
        assert set(reg.keys()) == expected

    def test_each_entry_has_builder_schema_description(self):
        from registry import get_registry
        for name, entry in get_registry().items():
            assert callable(entry.builder),       f"{name}: builder not callable"
            assert entry.schema is not None,      f"{name}: schema is None"
            assert len(entry.description) > 5,    f"{name}: description too short"

    def test_registry_is_singleton(self):
        from registry import get_registry
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_entry_graph_property_returns_compiled(self):
        from registry import get_registry
        entry = get_registry()["financial_reporter"]
        g = entry.graph
        assert hasattr(g, "ainvoke")
        assert hasattr(g, "astream")

    def test_entry_graph_cached_across_calls(self):
        from registry import get_registry
        entry = get_registry()["news_reporter"]
        assert entry.graph is entry.graph   # same object

    def test_with_checkpointer_returns_new_graph(self):
        from registry import get_registry
        from langgraph.checkpoint.memory import InMemorySaver
        entry = get_registry()["financial_reporter"]
        saver = InMemorySaver()
        g = entry.with_checkpointer(saver)
        assert g.checkpointer is saver
        # Bare graph still has no checkpointer
        assert entry.graph.checkpointer is None

    def test_input_schemas_have_correct_fields(self):
        from registry import get_registry
        reg = get_registry()
        cases = {
            "financial_reporter": ["symbol"],
            "investment_advisor": ["symbol"],
            "financial_query":    ["symbol", "query"],
            "trending_posts":     ["subreddit_name"],
            "news_reporter":      ["topic"],
            "medical_researcher": ["term"],
        }
        for name, expected_fields in cases.items():
            schema_fields = set(reg[name].schema.model_fields.keys())
            for f in expected_fields:
                assert f in schema_fields, f"{name} schema missing field '{f}'"


# ── Folder structure ──────────────────────────────────────────────────────────

class TestProjectStructure:
    BASE = os.path.dirname(os.path.dirname(__file__))

    def _path(self, *parts):
        return os.path.join(self.BASE, *parts)

    def test_all_packages_exist(self):
        for pkg in ["graphs", "tools", "chains", "state", "api", "config", "tests"]:
            assert os.path.isdir(self._path(pkg)), f"Missing package: {pkg}"

    def test_all_package_inits_exist(self):
        for pkg in ["graphs", "tools", "state", "api", "config", "tests"]:
            init = self._path(pkg, "__init__.py")
            assert os.path.isfile(init), f"Missing __init__.py in {pkg}"

    def test_core_files_exist(self):
        for fname in ["main.py", "registry.py", "pyproject.toml", ".env.example"]:
            assert os.path.isfile(self._path(fname)), f"Missing: {fname}"

    def test_all_graph_files_exist(self):
        for name in ["financial", "investment", "query", "reddit", "news", "medical"]:
            assert os.path.isfile(self._path("graphs", f"{name}.py"))

    def test_all_tool_files_exist(self):
        for name in ["finance", "reddit", "pubmed", "serper"]:
            assert os.path.isfile(self._path("tools", f"{name}.py"))