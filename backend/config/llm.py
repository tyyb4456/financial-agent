"""
LLM factory
-----------
One place to create LLM instances.
Supports Google Gemini (default) and Groq as a fallback.

Usage:
    from config.llm import get_llm
    llm = get_llm()                          # default: gemini-1.5-flash
    llm = get_llm("groq/llama-3.3-70b-versatile")
"""

from functools import lru_cache
from langchain.chat_models import init_chat_model
from config.settings import settings


# ── Default model identifiers ─────────────────────────────────────────────────
DEFAULT_GEMINI = "google_genai/gemini-2.5-flash"
DEFAULT_GROQ   = "groq/llama-3.3-70b-versatile"


@lru_cache(maxsize=8)
def get_llm(model_id: str = DEFAULT_GEMINI, temperature: float = 0.3):
    """
    Return a cached ChatModel instance.

    Args:
        model_id:    LangChain model string.
                     Format: "<provider>/<model-name>"
                     e.g.  "google_genai/gemini-1.5-flash"
                           "groq/llama-3.3-70b-versatile"
        temperature: Sampling temperature (0.0 – 1.0).

    Returns:
        A LangChain BaseChatModel instance, ready for .invoke() / .stream().
    """
    return init_chat_model(
        model_id,
        temperature=temperature,
        # init_chat_model picks up API keys from environment automatically
        # (GOOGLE_API_KEY, GROQ_API_KEY, etc.)
    )