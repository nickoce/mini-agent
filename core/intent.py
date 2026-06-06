from typing import Any

from core.json_utils import parse_json_object
from core.llm import chat_with_llm, chat_with_llm_with_usage


INTENT_SYSTEM_PROMPT = """
You are an intent detection module for a simple question-answering agent.

Decide whether the user's question requires internet search.

Return only valid JSON. Do not include markdown, comments, or extra text.

If search is needed, return:
{
  "need_search": true,
  "query": "a concise search query",
  "reason": "brief reason"
}

If search is not needed, return:
{
  "need_search": false,
  "reason": "brief reason"
}

Search is needed only when the user asks for recent, current, real-time, price,
news, policy, version, company status, person status, or anything likely to
have changed.

Search is not needed for stable general knowledge, coding basics, writing,
translation, math, or reasoning tasks.

Important examples that do not need search:
- "TCP三次握手是什么？"
- "Python list 和 tuple 有什么区别？"
- "HTTP 和 HTTPS 的区别是什么？"
- "什么是 Transformer？"

Do not search merely to verify accuracy. Stable technical concepts should be
answered directly from model knowledge.
""".strip()


NO_SEARCH_KEYWORDS = [
    "是什么",
    "什么意思",
    "区别",
    "原理",
    "解释",
    "介绍",
    "怎么理解",
]

SEARCH_KEYWORDS = [
    "今天",
    "昨日",
    "昨天",
    "最新",
    "当前",
    "现在",
    "实时",
    "新闻",
    "价格",
    "股价",
    "政策",
    "法规",
    "版本",
    "发布",
    "最近",
    "本周",
    "本月",
    "2026",
]


def detect_intent(question: str) -> dict[str, Any]:
    intent, _tokens = detect_intent_with_usage(question)
    return intent


def detect_intent_with_usage(question: str) -> tuple[dict[str, Any], int]:
    rule_based_intent = _detect_stable_knowledge(question)
    if rule_based_intent is not None:
        return rule_based_intent, 0

    result = chat_with_llm_with_usage(INTENT_SYSTEM_PROMPT, question)
    intent = parse_json_object(result["content"])

    need_search = bool(intent.get("need_search"))
    if not need_search:
        return {
            "need_search": False,
            "reason": str(intent.get("reason") or "这是稳定知识问题，不需要联网搜索。"),
        }, result["total_tokens"]

    query = str(intent.get("query") or question).strip()
    return {
        "need_search": True,
        "query": query,
        "reason": str(intent.get("reason") or "这是时效性问题，需要联网搜索。"),
    }, result["total_tokens"]


def _detect_stable_knowledge(question: str) -> dict[str, Any] | None:
    normalized = question.strip().lower()

    if any(keyword in normalized for keyword in SEARCH_KEYWORDS):
        return None

    if any(keyword in normalized for keyword in NO_SEARCH_KEYWORDS):
        return {
            "need_search": False,
            "reason": "这是稳定知识问题，不需要联网搜索。",
        }

    return None
