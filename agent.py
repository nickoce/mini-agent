import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from tools.search import tavily_search
from utils.logger import add_trace


load_dotenv()

MODEL = os.getenv("OPENAI_MODEL") or os.getenv("BAILIAN_MODEL", "qwen-plus")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
LLM_API_KEY = OPENAI_API_KEY or DASHSCOPE_API_KEY
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("BAILIAN_BASE_URL")

if not LLM_BASE_URL and DASHSCOPE_API_KEY and not OPENAI_API_KEY:
    LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


INTENT_PROMPT = """
You are an intent detection module for a web-search agent.

Decide whether the user's question requires internet search.

Return only valid JSON.

If search is needed:
{
  "need_search": true,
  "query": "concise search query",
  "reason": "brief reason"
}

If search is not needed:
{
  "need_search": false,
  "reason": "brief reason"
}

Search is needed for recent, current, real-time, news, price, policy, version,
company status, person status, or information likely to have changed.

Search is not needed for stable general knowledge, math, coding basics,
networking basics, writing, translation, or reasoning.

Do not search merely to verify accuracy.
Use plain text only. Do not use emoji.
""".strip()


ANSWER_PROMPT = """
You are a helpful assistant.

Answer clearly and concisely.
Use plain text only. Do not use emoji.
""".strip()


SEARCH_ANSWER_PROMPT = """
You are a helpful assistant that answers using provided search results.

Use the search results for factual or current claims.
If the search results are insufficient, say so clearly.
Include useful source URLs when appropriate.
Use plain text only. Do not use emoji.
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
    "昨天",
    "昨日",
    "最新",
    "最近",
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
    "本周",
    "本月",
]


def run_agent(question: str) -> dict[str, Any]:
    started_at = time.perf_counter()
    trace: list[dict[str, Any]] = []
    total_tokens = 0
    search_calls = 0

    add_trace(trace, "User Input", {"question": question})

    intent, tokens = detect_intent(question)
    total_tokens += tokens
    add_trace(trace, "Intent Detection", intent)

    if intent["need_search"]:
        thought = intent.get("reason") or "这是时效性问题，需要联网搜索。"
        add_trace(trace, "Thought 1", thought)

        query = intent.get("query") or question
        action = {
            "tool": "search",
            "input": query,
        }
        add_trace(trace, "Action 1", action)

        search_results = tavily_search(query=query, max_results=5)
        search_calls += 1
        observation = summarize_search_results(search_results)
        add_trace(trace, "Observation 1", observation)

        add_trace(trace, "Thought 2", "搜索结果已返回，开始基于结果生成最终答案。")
        answer, tokens = answer_with_search_results(question, search_results)
        total_tokens += tokens
    else:
        add_trace(trace, "Thought 1", intent.get("reason") or "无需联网搜索。")
        add_trace(
            trace,
            "Action 1",
            {
                "tool": "final_answer",
                "input": "finish",
            },
        )
        answer, tokens = direct_answer(question)
        total_tokens += tokens

    add_trace(trace, "Final Answer", answer)

    return {
        "intent": intent,
        "trace": trace,
        "final_answer": answer,
        "metrics": {
            "tokens": total_tokens,
            "latency": round(time.perf_counter() - started_at, 2),
            "search_calls": search_calls,
        },
    }


def detect_intent(question: str) -> tuple[dict[str, Any], int]:
    rule_result = rule_based_intent(question)
    if rule_result is not None:
        return rule_result, 0

    content, tokens = call_llm(INTENT_PROMPT, question)
    intent = parse_json_object(content)

    if not bool(intent.get("need_search")):
        return {
            "need_search": False,
            "reason": str(intent.get("reason") or "这是稳定问题，不需要联网搜索。"),
        }, tokens

    return {
        "need_search": True,
        "query": str(intent.get("query") or question),
        "reason": str(intent.get("reason") or "这是时效性问题，需要联网搜索。"),
    }, tokens


def rule_based_intent(question: str) -> dict[str, Any] | None:
    normalized = question.strip().lower()

    if any(keyword in normalized for keyword in SEARCH_KEYWORDS):
        return None

    if any(keyword in normalized for keyword in NO_SEARCH_KEYWORDS):
        return {
            "need_search": False,
            "reason": "这是稳定知识问题，不需要联网搜索。",
        }

    return None


def direct_answer(question: str) -> tuple[str, int]:
    return call_llm(ANSWER_PROMPT, question)


def answer_with_search_results(
    question: str,
    search_results: list[dict[str, Any]],
) -> tuple[str, int]:
    user_prompt = f"""
User question:
{question}

Search results:
{json.dumps(search_results, ensure_ascii=False, indent=2)}

Generate the final answer.
""".strip()

    return call_llm(SEARCH_ANSWER_PROMPT, user_prompt)


def call_llm(system_prompt: str, user_prompt: str) -> tuple[str, int]:
    if not LLM_API_KEY:
        raise RuntimeError("Missing LLM API key. Set OPENAI_API_KEY or DASHSCOPE_API_KEY.")

    client_kwargs = {"api_key": LLM_API_KEY}
    if LLM_BASE_URL:
        client_kwargs["base_url"] = LLM_BASE_URL

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
    return response.choices[0].message.content or "", total_tokens or 0


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Model did not return JSON: {text}")
        value = json.loads(match.group(0))

    if not isinstance(value, dict):
        raise ValueError(f"Model returned non-object JSON: {text}")

    return value


def summarize_search_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "result_count": len(results),
        "results": [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": truncate(item.get("content", ""), 220),
                "score": item.get("score"),
            }
            for item in results
        ],
    }


def truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
