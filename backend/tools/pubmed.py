"""
tools/pubmed.py
---------------
PubMed E-utilities tool for fetching medical article abstracts.

Uses NCBI's free ESearch + EFetch endpoints — no API key required.
Fetches up to `max_results` article IDs then retrieves their abstracts.
"""

from __future__ import annotations

import structlog
import requests
from pydantic import BaseModel, Field
from langchain.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = structlog.get_logger(__name__)

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

_http_retry = retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


# ── Arg schema ────────────────────────────────────────────────────────────────

class PubMedInput(BaseModel):
    """Input for fetching PubMed abstracts."""
    term: str = Field(
        description="Medical search term or condition, e.g. 'type 2 diabetes treatment' or 'mRNA vaccines'."
    )
    max_results: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of articles to retrieve (1–5).",
    )


# ── Tool ──────────────────────────────────────────────────────────────────────

@tool("fetch_pubmed_abstracts", args_schema=PubMedInput)
def fetch_pubmed_abstracts(term: str, max_results: int = 2) -> str:
    """
    Search PubMed for medical articles matching a term and return their abstracts.

    Uses NCBI E-utilities (ESearch + EFetch). Returns plain-text abstracts
    concatenated together. Use this for evidence-based medical research summaries.
    """
    log.info("tool.fetch_pubmed_abstracts", term=term, max_results=max_results)

    try:
        # ── Step 1: search for article IDs ────────────────────────────────────
        @_http_retry
        def _esearch() -> list[str]:
            resp = requests.get(
                _ESEARCH_URL,
                params={"db": "pubmed", "term": term, "retmax": str(max_results), "retmode": "json"},
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json().get("esearchresult", {}).get("idlist", [])

        ids = _esearch()

        if not ids:
            return f"No PubMed articles found for: {term}"

        # ── Step 2: fetch abstracts ────────────────────────────────────────────
        @_http_retry
        def _efetch(article_ids: list[str]) -> str:
            resp = requests.get(
                _EFETCH_URL,
                params={
                    "db":      "pubmed",
                    "id":      ",".join(article_ids),
                    "retmode": "text",
                    "rettype": "abstract",
                },
                timeout=20,
            )
            resp.raise_for_status()
            return resp.text

        abstracts = _efetch(ids)
        log.info("tool.fetch_pubmed_abstracts.done", term=term, articles=len(ids))
        return abstracts

    except requests.RequestException as exc:
        log.error("tool.fetch_pubmed_abstracts.error", term=term, error=str(exc))
        return f"Failed to fetch PubMed data: {exc}"
    except Exception as exc:
        log.error("tool.fetch_pubmed_abstracts.error", term=term, error=str(exc))
        return f"Unexpected error: {exc}"