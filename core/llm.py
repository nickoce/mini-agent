from openai import OpenAI

from config.settings import BAILIAN_BASE_URL, BAILIAN_MODEL, DASHSCOPE_API_KEY


def _get_client() -> OpenAI:
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured.")

    return OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=BAILIAN_BASE_URL,
    )


def chat_with_llm(system_prompt: str, user_prompt: str) -> str:
    result = chat_with_llm_with_usage(system_prompt, user_prompt)
    return result["content"]


def chat_with_llm_with_usage(system_prompt: str, user_prompt: str) -> dict:
    client = _get_client()

    completion = client.chat.completions.create(
        model=BAILIAN_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ],
    )

    usage = getattr(completion, "usage", None)
    total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

    return {
        "content": completion.choices[0].message.content or "",
        "total_tokens": total_tokens or 0,
    }


def ask_llm(question: str) -> str:
    return chat_with_llm(
        "You are a helpful assistant. Answer the user's question clearly and concisely.",
        question,
    )


def answer_with_search_results(question: str, search_results: str) -> str:
    system_prompt = """
You are a helpful assistant that answers questions using provided search results.

Use only the information in the search results when answering current or factual claims.
If the search results are insufficient, say so clearly.
Keep the answer concise and include source URLs when useful.
Use plain text only. Do not use emoji.
""".strip()

    user_prompt = f"""
User question:
{question}

Search results:
{search_results}

Generate the final answer.
""".strip()

    return chat_with_llm(system_prompt, user_prompt)
