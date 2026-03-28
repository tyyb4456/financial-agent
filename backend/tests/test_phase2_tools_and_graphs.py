"""
tests/test_phase2_tools_and_graphs.py
--------------------------------------
Tests for Phase 2: tools (schema, contract, retry) and all 6 StateGraphs.

Strategy:
  - Tool tests verify schema shape, name, description length — NOT live API calls
  - Graph tests run end-to-end with mock_tools + mock_llm fixtures
  - Each graph test asserts on: state transitions, final `result` field, node names
"""

import pytest
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────

class TestFinanceTools:
    def test_fetch_financials_name(self):
        from tools.finance import fetch_financials
        assert fetch_financials.name == "fetch_financials"

    def test_fetch_financials_schema_fields(self):
        from tools.finance import fetch_financials
        fields = fetch_financials.args_schema.model_json_schema()["properties"]
        assert "symbol" in fields

    def test_fetch_financials_description_not_empty(self):
        from tools.finance import fetch_financials
        assert len(fetch_financials.description) > 20

    def test_fetch_investment_analysis_name(self):
        from tools.finance import fetch_investment_analysis
        assert fetch_investment_analysis.name == "fetch_investment_analysis"

    def test_fetch_investment_analysis_schema(self):
        from tools.finance import fetch_investment_analysis
        fields = fetch_investment_analysis.args_schema.model_json_schema()["properties"]
        assert "symbol" in fields

    def test_fetch_financials_returns_error_dict_on_exception(self, monkeypatch):
        from tools.finance import fetch_financials
        import tools.finance as fin_mod
        # Patch Ticker to blow up
        monkeypatch.setattr(fin_mod, "Ticker", lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
        result = fetch_financials.invoke({"symbol": "BAD"})
        assert "error" in result

    def test_fetch_investment_returns_error_dict_on_exception(self, monkeypatch):
        from tools.finance import fetch_investment_analysis
        import tools.finance as fin_mod
        monkeypatch.setattr(fin_mod, "Ticker", lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
        result = fetch_investment_analysis.invoke({"symbol": "BAD"})
        assert "error" in result


class TestRedditTool:
    def test_fetch_trending_posts_name(self):
        from tools.reddit import fetch_trending_posts
        assert fetch_trending_posts.name == "fetch_trending_posts"

    def test_fetch_trending_posts_schema(self):
        from tools.reddit import fetch_trending_posts
        fields = fetch_trending_posts.args_schema.model_json_schema()["properties"]
        assert "subreddit_name" in fields
        assert "limit" in fields

    def test_limit_field_has_bounds(self):
        from tools.reddit import TrendingPostsInput
        import pydantic
        # limit must be >= 1 and <= 50
        with pytest.raises(pydantic.ValidationError):
            TrendingPostsInput(subreddit_name="test", limit=0)
        with pytest.raises(pydantic.ValidationError):
            TrendingPostsInput(subreddit_name="test", limit=51)

    def test_fetch_trending_returns_error_list_on_exception(self, monkeypatch):
        from tools.reddit import fetch_trending_posts
        import tools.reddit as red_mod
        monkeypatch.setattr(red_mod, "_reddit_client", lambda: (_ for _ in ()).throw(RuntimeError("no creds")))
        result = fetch_trending_posts.invoke({"subreddit_name": "test", "limit": 5})
        assert isinstance(result, list)
        assert "error" in result[0]


class TestPubMedTool:
    def test_fetch_pubmed_abstracts_name(self):
        from tools.pubmed import fetch_pubmed_abstracts
        assert fetch_pubmed_abstracts.name == "fetch_pubmed_abstracts"

    def test_fetch_pubmed_schema(self):
        from tools.pubmed import fetch_pubmed_abstracts
        fields = fetch_pubmed_abstracts.args_schema.model_json_schema()["properties"]
        assert "term" in fields
        assert "max_results" in fields

    def test_max_results_bounds(self):
        from tools.pubmed import PubMedInput
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            PubMedInput(term="test", max_results=0)
        with pytest.raises(pydantic.ValidationError):
            PubMedInput(term="test", max_results=6)

    def test_returns_string_on_success(self, monkeypatch):
        import requests
        from tools.pubmed import fetch_pubmed_abstracts

        mock_esearch = MagicMock()
        mock_esearch.json.return_value = {"esearchresult": {"idlist": ["12345"]}}
        mock_esearch.raise_for_status = MagicMock()

        mock_efetch = MagicMock()
        mock_efetch.text = "Abstract: some medical text"
        mock_efetch.raise_for_status = MagicMock()

        call_count = [0]
        def fake_get(url, **kwargs):
            call_count[0] += 1
            return mock_esearch if call_count[0] == 1 else mock_efetch

        monkeypatch.setattr(requests, "get", fake_get)
        result = fetch_pubmed_abstracts.invoke({"term": "diabetes", "max_results": 1})
        assert isinstance(result, str)
        assert "Abstract" in result

    def test_returns_not_found_when_no_ids(self, monkeypatch):
        import requests
        from tools.pubmed import fetch_pubmed_abstracts

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"esearchresult": {"idlist": []}}
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr(requests, "get", lambda *a, **kw: mock_resp)

        result = fetch_pubmed_abstracts.invoke({"term": "xyznonexistent", "max_results": 1})
        assert "No PubMed articles found" in result


class TestSerperTool:
    def test_search_news_name(self):
        from tools.serper import search_news
        assert search_news.name == "search_news"

    def test_search_news_schema(self):
        from tools.serper import search_news
        fields = search_news.args_schema.model_json_schema()["properties"]
        assert "query" in fields
        assert "num_results" in fields

    def test_num_results_bounds(self):
        from tools.serper import SerperInput
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            SerperInput(query="test", num_results=0)
        with pytest.raises(pydantic.ValidationError):
            SerperInput(query="test", num_results=11)

    def test_returns_list_on_success(self, monkeypatch):
        import requests, json as json_mod
        from tools.serper import search_news

        fake_articles = [{"title": "T", "snippet": "S", "link": "http://x.com",
                          "date": "2025-01-01", "source": "AP"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"news": fake_articles}
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr(requests, "post", lambda *a, **kw: mock_resp)

        result = search_news.invoke({"query": "AI news", "num_results": 1})
        assert isinstance(result, list)
        assert result[0]["title"] == "T"
        assert "url" in result[0]     # normalised key

    def test_returns_error_list_on_request_failure(self, monkeypatch):
        import requests
        from tools.serper import search_news
        monkeypatch.setattr(requests, "post",
                            lambda *a, **kw: (_ for _ in ()).throw(requests.RequestException("timeout")))
        result = search_news.invoke({"query": "test", "num_results": 1})
        assert isinstance(result, list)
        assert "error" in result[0]


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHS
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphStructure:
    """Tests the compiled graph topology — no execution needed."""

    def _nodes(self, graph):
        return [n for n in graph.get_graph().nodes.keys()
                if not n.startswith("__")]

    def test_financial_graph_nodes(self):
        from graphs.financial import _build
        g = _build()
        assert self._nodes(g) == ["fetch_data", "write_report"]

    def test_investment_graph_nodes(self):
        from graphs.investment import _build
        g = _build()
        assert self._nodes(g) == ["fetch_data", "fetch_analysis", "write_recommendation"]

    def test_query_graph_nodes(self):
        from graphs.query import _build
        g = _build()
        assert self._nodes(g) == ["fetch_data", "fetch_analysis", "answer_query"]

    def test_reddit_graph_nodes(self):
        from graphs.reddit import _build
        g = _build()
        assert self._nodes(g) == ["fetch_posts", "summarise"]

    def test_news_graph_nodes(self):
        from graphs.news import _build
        g = _build()
        assert self._nodes(g) == ["search_articles", "write_report"]

    def test_medical_graph_nodes(self):
        from graphs.medical import _build
        g = _build()
        nodes = set(self._nodes(g))
        assert nodes == {"fetch_abstracts", "summarise", "answer_query"}

    def test_medical_graph_has_conditional_edges(self):
        from graphs.medical import _build
        g = _build()
        # fetch_abstracts should branch to both summarise AND answer_query
        edges = [(s, d) for s, d, *_ in g.get_graph().edges]
        targets_of_fetch = [d for s, d in edges if s == "fetch_abstracts"]
        assert "summarise"    in targets_of_fetch
        assert "answer_query" in targets_of_fetch

    def test_all_graphs_accept_checkpointer(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from graphs.financial  import _build as fb
        from graphs.investment import _build as ib
        from graphs.query      import _build as qb
        from graphs.reddit     import _build as rb
        from graphs.news       import _build as nb
        from graphs.medical    import _build as mb
        for name, builder in [("financial",fb),("investment",ib),("query",qb),
                               ("reddit",rb),("news",nb),("medical",mb)]:
            saver = InMemorySaver()
            g = builder(checkpointer=saver)
            assert g.checkpointer is saver, f"{name} checkpointer not injected"

    def test_bare_graph_has_no_checkpointer(self):
        from graphs.financial import _build
        g = _build()
        assert g.checkpointer is None


class TestFinancialGraphExecution:
    @pytest.mark.asyncio
    async def test_full_run_returns_result(self, mock_tools, mock_llm):
        from graphs.financial import _build
        g = _build()
        out = await g.ainvoke({"symbol": "AAPL", "raw_financials": None, "result": None})
        assert out["result"] is not None
        assert isinstance(out["result"], str)
        assert len(out["result"]) > 0

    @pytest.mark.asyncio
    async def test_raw_financials_populated(self, mock_tools, mock_llm):
        from graphs.financial import _build
        g = _build()
        out = await g.ainvoke({"symbol": "AAPL", "raw_financials": None, "result": None})
        assert out["raw_financials"] is not None

    @pytest.mark.asyncio
    async def test_llm_called_with_symbol_in_prompt(self, mock_tools, mock_llm):
        from graphs.financial import _build
        g = _build()
        await g.ainvoke({"symbol": "TSLA", "raw_financials": None, "result": None})
        call_args = mock_llm.invoke.call_args[0][0]
        assert "TSLA" in call_args


class TestInvestmentGraphExecution:
    @pytest.mark.asyncio
    async def test_full_run_returns_result(self, mock_tools, mock_llm):
        from graphs.investment import _build
        g = _build()
        out = await g.ainvoke({"symbol": "MSFT", "raw_financials": None,
                               "raw_investment": None, "result": None})
        assert isinstance(out["result"], str)

    @pytest.mark.asyncio
    async def test_both_data_fields_populated(self, mock_tools, mock_llm):
        from graphs.investment import _build
        g = _build()
        out = await g.ainvoke({"symbol": "MSFT", "raw_financials": None,
                               "raw_investment": None, "result": None})
        assert out["raw_financials"] is not None
        assert out["raw_investment"] is not None


class TestQueryGraphExecution:
    @pytest.mark.asyncio
    async def test_query_in_prompt(self, mock_tools, mock_llm):
        from graphs.query import _build
        g = _build()
        await g.ainvoke({"symbol": "GOOG", "query": "What is the P/E ratio?",
                         "raw_financials": None, "raw_investment": None, "result": None})
        prompt = mock_llm.invoke.call_args[0][0]
        assert "P/E ratio" in prompt

    @pytest.mark.asyncio
    async def test_result_is_string(self, mock_tools, mock_llm):
        from graphs.query import _build
        g = _build()
        out = await g.ainvoke({"symbol": "GOOG", "query": "Revenue?",
                               "raw_financials": None, "raw_investment": None, "result": None})
        assert isinstance(out["result"], str)


class TestRedditGraphExecution:
    @pytest.mark.asyncio
    async def test_full_run_returns_result(self, mock_tools, mock_llm):
        from graphs.reddit import _build
        g = _build()
        out = await g.ainvoke({"subreddit_name": "investing", "raw_posts": None, "result": None})
        assert isinstance(out["result"], str)

    @pytest.mark.asyncio
    async def test_raw_posts_populated(self, mock_tools, mock_llm):
        from graphs.reddit import _build
        g = _build()
        out = await g.ainvoke({"subreddit_name": "technology", "raw_posts": None, "result": None})
        assert isinstance(out["raw_posts"], list)
        assert len(out["raw_posts"]) > 0


class TestNewsGraphExecution:
    @pytest.mark.asyncio
    async def test_full_run_returns_result(self, mock_tools, mock_llm):
        from graphs.news import _build
        g = _build()
        out = await g.ainvoke({"topic": "AI regulation", "raw_articles": None, "result": None})
        assert isinstance(out["result"], str)

    @pytest.mark.asyncio
    async def test_topic_in_prompt(self, mock_tools, mock_llm):
        from graphs.news import _build
        g = _build()
        await g.ainvoke({"topic": "quantum computing", "raw_articles": None, "result": None})
        prompt = mock_llm.invoke.call_args[0][0]
        assert "quantum computing" in prompt


class TestMedicalGraphExecution:
    @pytest.mark.asyncio
    async def test_no_query_routes_to_summarise(self, mock_tools, mock_llm):
        from graphs.medical import _build
        g = _build()
        out = await g.ainvoke({"term": "diabetes", "query": "",
                               "raw_abstract": None, "result": None})
        # summarise prompt contains the term, not a question
        assert isinstance(out["result"], str)

    @pytest.mark.asyncio
    async def test_with_query_routes_to_answer_query(self, mock_tools, mock_llm):
        from graphs.medical import _build
        g = _build()
        await g.ainvoke({"term": "diabetes", "query": "What is the best treatment?",
                         "raw_abstract": None, "result": None})
        prompt = mock_llm.invoke.call_args[0][0]
        assert "What is the best treatment?" in prompt

    @pytest.mark.asyncio
    async def test_raw_abstract_populated(self, mock_tools, mock_llm):
        from graphs.medical import _build
        g = _build()
        out = await g.ainvoke({"term": "cancer", "query": "",
                               "raw_abstract": None, "result": None})
        assert out["raw_abstract"] is not None

    def test_conditional_router_no_query(self):
        from graphs.medical import _route
        assert _route({"term": "x", "query": "",    "raw_abstract": None, "result": None}) == "summarise"
        assert _route({"term": "x", "query": None,  "raw_abstract": None, "result": None}) == "summarise"
        assert _route({"term": "x", "query": "  ",  "raw_abstract": None, "result": None}) == "summarise"

    def test_conditional_router_with_query(self):
        from graphs.medical import _route
        result = _route({"term": "x", "query": "What dose?", "raw_abstract": None, "result": None})
        assert result == "answer_query"