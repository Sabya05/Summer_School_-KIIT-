"""
services/tavily_service.py

Responsible ONLY for talking to the Tavily API and returning
clean, structured search results. No Gemini logic lives here.
"""

from tavily import TavilyClient
from config import TAVILY_API_KEY, MAX_SEARCH_RESULTS


def search_news(query: str) -> list[dict]:
    """
    Search the web using Tavily for the given query.

    Returns a list of dictionaries like:
        {"title": ..., "url": ..., "content": ...}

    Raises:
        ValueError: if the Tavily API key is missing.
        RuntimeError: if the Tavily API call fails.
    """

    if not TAVILY_API_KEY:
        raise ValueError(
            "Tavily API key is missing. Please add it to your .env file."
        )

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)

        response = client.search(
            query=query,

            # More thorough search
            search_depth="advanced",

            # Focus on current news
            topic="news",

            # Retrieve more sources
            max_results=MAX_SEARCH_RESULTS,

            # We want Gemini to do the analysis
            include_answer=False,

            # Get more complete page information
            include_raw_content=True,
        )

    except Exception as error:
        raise RuntimeError(
            "Unable to retrieve web results. Please check your Tavily API key "
            "and internet connection."
        ) from error

    raw_results = response.get("results", [])

    clean_results = []

    for item in raw_results:
        url = item.get("url", "").strip()
        title = item.get("title", "Untitled source").strip()

        # Prefer raw content when available
        content = item.get("raw_content") or item.get("content") or ""
        content = content.strip()

        if not url or not content:
            continue

        clean_results.append({
            "title": title,
            "url": url,
            "content": content,
        })

    return clean_results