import httpx
import logging
from typing import List, Dict
from duckduckgo_search import DDGS
from config import settings
from langfuse import observe

logger = logging.getLogger(__name__)

@observe(as_type="retriever")
async def web_search(query: str, max_results: int = 3) -> str:
    """Performs a web search using Tavily (if configured) or DuckDuckGo Search (open source)."""
    # 1. Try Tavily first if API key exists
    if settings.TAVILY_API_KEY:
        logger.info(f"Using Tavily for query: '{query}'")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": settings.TAVILY_API_KEY,
                        "query": query,
                        "search_depth": "basic",
                        "max_results": max_results
                    },
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    formatted = []
                    for idx, r in enumerate(results):
                        formatted.append(f"Source {idx+1}: {r.get('title')}\nSnippet: {r.get('content')}")
                    return "\n\n".join(formatted)
        except Exception as e:
            logger.warning(f"Tavily search error: {str(e)}. Falling back to DuckDuckGo...")
 
    # 2. Fall back to DuckDuckGo (free, open source)
    logger.info(f"Using DuckDuckGo for query: '{query}'")
    try:
        # Run blocking duckduckgo call in executor
        return await asyncio_ddg(query, max_results)
    except Exception as e:
        logger.error(f"DuckDuckGo search error: {str(e)}")
        return f"Could not retrieve search results for {query} due to connection error."

import asyncio
async def asyncio_ddg(query: str, max_results: int) -> str:
    """Wrapper to run DDGS synchronously in an async thread pool."""
    def run_sync():
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "No search results found."
            formatted = []
            for idx, r in enumerate(results):
                formatted.append(f"Source {idx+1}: {r.get('title')}\nSnippet: {r.get('body')}")
            return "\n\n".join(formatted)
    return await asyncio.to_thread(run_sync)
