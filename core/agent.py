import json
import time
from typing import Any

from config.settings import CURRENT_DATE
from core.intent import detect_intent_with_usage
from core.json_utils import parse_json_object
from core.llm import chat_with_llm_with_usage
from tools.tavily_search import search_tavily


MAX_AGENT_LOOPS = 3

AGENT_SYSTEM_PROMPT = """
You are a small search agent.

At each step, decide whether to search again or finish with a final answer.

Return only valid JSON. Do not include markdown or extra text.

Allowed actions:
1. Search
2. Final Answer

When searching, return:
{
  "thought": "brief reason visible to the user",
  "action": "Search",
  "query": "search query"
}

When finishing, return:
{
  "thought": "brief reason visible to the user",
  "action": "Final Answer",
  "final_answer": "answer to the user",
  "termination_reason": "information is sufficient"
}

Rules:
- Maximum total steps is 3.
- Search only if current or external information is needed.
- If Intent Detection says need_search is false, you must choose Final Answer.
- If observations already contain enough information, choose Final Answer.
- If this is the last step, choose Final Answer.
- Keep thought short and do not reveal hidden chain-of-thought.
- Use plain text only. Do not use emoji.
""".strip()


def run_agent(question: str) -> dict[str, Any]:
    started_at = time.perf_counter()
    trace_log: list[dict[str, Any]] = []
    trace_steps: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    total_tokens = 0
    search_calls = 0

    _add_trace(trace_log, "User Input", {"question": question})

    intent, intent_tokens = detect_intent_with_usage(question)
    total_tokens += intent_tokens
    _add_trace(trace_log, "Intent Detection", intent)

    final_answer = ""
    termination_reason = ""

    for step in range(1, MAX_AGENT_LOOPS + 1):
        _add_trace(trace_log, f"Loop {step}", {"loop": step})
        loop_record: dict[str, Any] = {"loop": step}

        decision, decision_tokens = _decide_next_step(question, intent, observations, step)
        total_tokens += decision_tokens

        thought = str(decision.get("thought") or "").strip()
        action = str(decision.get("action") or "").strip()

        loop_record["thought"] = thought
        _add_trace(trace_log, "Thought", thought)

        if action == "Search" and step < MAX_AGENT_LOOPS:
            query = str(decision.get("query") or intent.get("query") or question).strip()
            action_record = {
                "tool": "search",
                "input": query,
            }
            loop_record["action"] = action_record
            _add_trace(trace_log, "Action", action_record)

            results = search_tavily(query, max_results=5)
            search_calls += 1

            observation = {
                "query": query,
                "results": results,
            }
            observations.append(observation)
            observation_summary = _summarize_observation(observation)
            loop_record["observation"] = observation_summary
            _add_trace(trace_log, "Observation", observation_summary)

            trace_steps.append(loop_record)
            continue

        final_answer = str(decision.get("final_answer") or "").strip()
        termination_reason = str(
            decision.get("termination_reason") or "信息已足够"
        ).strip()

        if action == "Search" and step >= MAX_AGENT_LOOPS:
            termination_reason = "达到最大循环次数"

        action_record = {
            "tool": "final_answer",
            "input": "finish",
        }
        loop_record["action"] = action_record
        _add_trace(trace_log, "Action", action_record)

        if not final_answer:
            final_answer, answer_tokens = _force_final_answer(question, observations)
            total_tokens += answer_tokens

        loop_record["final_answer"] = final_answer
        trace_steps.append(loop_record)

        _add_trace(trace_log, "Termination Reason", termination_reason)
        _add_trace(trace_log, "Final Answer", final_answer)
        return _build_result(
            intent=intent,
            trace_log=trace_log,
            trace_steps=trace_steps,
            final_answer=final_answer,
            termination_reason=termination_reason,
            total_tokens=total_tokens,
            search_calls=search_calls,
            started_at=started_at,
        )

    final_answer, answer_tokens = _force_final_answer(question, observations)
    total_tokens += answer_tokens
    termination_reason = "达到最大循环次数"
    _add_trace(trace_log, "Termination Reason", termination_reason)
    _add_trace(trace_log, "Final Answer", final_answer)

    return _build_result(
        intent=intent,
        trace_log=trace_log,
        trace_steps=trace_steps,
        final_answer=final_answer,
        termination_reason=termination_reason,
        total_tokens=total_tokens,
        search_calls=search_calls,
        started_at=started_at,
    )


def format_trace_log(trace_log: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    for entry in trace_log:
        label = entry["label"]
        value = entry["value"]
        lines.append(f"[{entry['index']}] {label}")

        if isinstance(value, str):
            lines.append(value)
        else:
            lines.append(json.dumps(value, ensure_ascii=False, indent=2))

        lines.append("")

    return "\n".join(lines).strip()


def _decide_next_step(
    question: str,
    intent: dict[str, Any],
    observations: list[dict[str, Any]],
    step: int,
) -> tuple[dict[str, Any], int]:
    user_prompt = f"""
User question:
{question}

Current date:
{CURRENT_DATE}

Intent detection result:
{json.dumps(intent, ensure_ascii=False, indent=2)}

Current loop:
Loop {step}

Maximum loops:
{MAX_AGENT_LOOPS}

Previous observations:
{json.dumps(observations, ensure_ascii=False, indent=2)}
""".strip()

    result = chat_with_llm_with_usage(AGENT_SYSTEM_PROMPT, user_prompt)
    decision = parse_json_object(result["content"])

    action = str(decision.get("action") or "").strip()
    if action not in {"Search", "Final Answer"}:
        raise ValueError(f"Agent returned unsupported action: {action}")

    if not intent.get("need_search") and action == "Search":
        decision["action"] = "Final Answer"
        decision["thought"] = "Intent Detection 判断这是稳定知识问题，不需要联网搜索。"
        decision.setdefault("termination_reason", "信息已足够")

    if step >= MAX_AGENT_LOOPS:
        decision["action"] = "Final Answer"
        decision.setdefault("termination_reason", "达到最大循环次数")

    return decision, result["total_tokens"]


def _force_final_answer(
    question: str,
    observations: list[dict[str, Any]],
) -> tuple[str, int]:
    system_prompt = """
You are a helpful assistant generating the final answer for a search agent.

Use the observations if provided. If they are insufficient, say so clearly.
Answer the user's question directly and concisely.
Use plain text only. Do not use emoji.
""".strip()

    user_prompt = f"""
User question:
{question}

Current date:
{CURRENT_DATE}

Observations:
{json.dumps(observations, ensure_ascii=False, indent=2)}

Generate Final Answer only.
""".strip()

    result = chat_with_llm_with_usage(system_prompt, user_prompt)
    return result["content"], result["total_tokens"]


def _build_result(
    intent: dict[str, Any],
    trace_log: list[dict[str, Any]],
    trace_steps: list[dict[str, Any]],
    final_answer: str,
    termination_reason: str,
    total_tokens: int,
    search_calls: int,
    started_at: float,
) -> dict[str, Any]:
    return {
        "intent_analysis": intent,
        "trace_log": trace_log,
        "trace_steps": trace_steps,
        "final_answer": final_answer,
        "termination_reason": termination_reason,
        "metrics": {
            "tokens": total_tokens,
            "latency": round(time.perf_counter() - started_at, 2),
            "search_calls": search_calls,
        },
    }


def _add_trace(trace_log: list[dict[str, Any]], label: str, value: Any) -> None:
    trace_log.append(
        {
            "index": len(trace_log) + 1,
            "label": label,
            "value": value,
        }
    )


def _summarize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": observation.get("query", ""),
        "result_count": len(observation.get("results", [])),
        "results": [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": _truncate(item.get("content", ""), max_length=180),
                "score": item.get("score"),
            }
            for item in observation.get("results", [])
        ],
    }


def _truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."
