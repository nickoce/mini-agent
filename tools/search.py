import json
import os
from typing import Any
from urllib import request

from dotenv import load_dotenv


load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_SEARCH_URL = os.getenv("TAVILY_SEARCH_URL", "https://api.tavily.com/search")


def tavily_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    if not TAVILY_API_KEY:
        raise RuntimeError("Missing TAVILY_API_KEY.")

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
        data = json.loads(response.read().decode("utf-8"))

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
