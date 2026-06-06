import json
from typing import Any
from urllib import request

from config.settings import TAVILY_API_KEY, TAVILY_SEARCH_URL


def search_tavily(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not configured.")

    payload = json.dumps(
        {
            "query": query,
            "max_results": max_results,
        }
    ).encode("utf-8")

    req = request.Request(
        TAVILY_SEARCH_URL,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {TAVILY_API_KEY}",
            "Content-Type": "application/json",
        },
    )

    with request.urlopen(req, timeout=30) as response:
        response_body = response.read().decode("utf-8")

    data = json.loads(response_body)
    results = data.get("results", [])

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "score": item.get("score"),
        }
        for item in results[:max_results]
    ]
