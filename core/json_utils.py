import json
import re
from typing import Any


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
