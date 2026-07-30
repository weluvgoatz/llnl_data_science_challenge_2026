"""The generic OpenAI-compatible tool-calling loop shared by the orchestrator
and every subagent. This module has no domain knowledge of lattices, defects,
or plots -- it only knows how to run a system prompt + tool roster against
the OpenAI Chat Completions API, dispatch tool calls to real Python
callables, and feed the results back, logging every call so the conversation
is auditable.

Tool schemas are authored elsewhere (subagents.py, orchestrator.py) in the
simple {name, description, input_schema} shape; `_to_provider_tools` adapts
them to OpenAI's {"type": "function", "function": {...}} wrapper here, so
swapping the underlying provider again later only touches this file.
"""

from __future__ import annotations

import json
from typing import Any, Callable


class ToolDispatchError(Exception):
    """Raised when the model calls a tool name that isn't in the dispatch table."""


def _to_provider_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in tools
    ]


def run_tool_loop(
    client: Any,
    model: str,
    system: str,
    tools: list[dict[str, Any]],
    dispatch: dict[str, Callable[..., Any]],
    messages: list[dict[str, Any]],
    max_turns: int = 8,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Run `client.chat.completions.create` in a loop until the model stops
    calling tools (or `max_turns` is hit), executing every tool call against
    `dispatch` and feeding the result back as a "tool" message.

    `messages` holds only user/assistant/tool turns (no system message --
    `system` is prepended fresh on every call so it's never persisted
    redundantly into the stored conversation). Mutates `messages` in place
    so the caller can persist or resume the conversation. Returns
    {"final_text", "tool_calls", "stop_reason"} -- `tool_calls` is the full,
    ordered log of every tool invoked this turn: {tool, input, output,
    is_error}, which is exactly the evidence trail an answer can be audited
    against.
    """
    tool_calls_log: list[dict[str, Any]] = []
    provider_tools = _to_provider_tools(tools)

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            tools=provider_tools,
            messages=[{"role": "system", "content": system}] + messages,
        )
        choice = response.choices[0]
        message = choice.message
        tool_calls = list(getattr(message, "tool_calls", None) or [])

        assistant_entry: dict[str, Any] = {"role": "assistant", "content": message.content}
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in tool_calls
            ]
        messages.append(assistant_entry)

        if not tool_calls:
            return {
                "final_text": message.content or "",
                "tool_calls": tool_calls_log,
                "stop_reason": choice.finish_reason,
            }

        for call in tool_calls:
            name = call.function.name
            args: dict[str, Any] = {}
            try:
                if call.function.arguments:
                    args = json.loads(call.function.arguments)
                fn = dispatch.get(name)
                if fn is None:
                    raise ToolDispatchError(f"Unknown tool: {name!r}. Available: {sorted(dispatch)}")
                result = fn(**args)
                is_error = False
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: tell the model, don't crash the turn
                result = {"error": str(exc)}
                is_error = True
            tool_calls_log.append({"tool": name, "input": args, "output": result, "is_error": is_error})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": _to_tool_result_text(result),
                }
            )

    return {
        "final_text": "(reached the tool-call limit for this turn without a final answer)",
        "tool_calls": tool_calls_log,
        "stop_reason": "max_turns",
    }


def _to_tool_result_text(result: Any) -> str:
    try:
        return json.dumps(result, default=str)
    except TypeError:
        return str(result)
