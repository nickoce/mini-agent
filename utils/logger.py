from typing import Any


def add_trace(trace: list[dict[str, Any]], trace_type: str, value: Any) -> None:
    trace.append(
        {
            "index": len(trace) + 1,
            "type": trace_type,
            "value": value,
        }
    )
