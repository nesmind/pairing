"""Fits a tool result into the admin's character limit without breaking it.

A JSON result is shortened structurally (long strings clipped, long lists cut, still valid JSON), because a
model that reads half a JSON document often gives up. Anything else is cut at a line or word boundary."""

import json
from typing import Any

_STRING_STEPS = (400, 200, 120, 80, 50, 30, 15)
_LIST_STEPS = (10, 6, 4, 3, 2, 1)


class ToolResultClip:
    def __init__(self, limit: int) -> None:
        self._limit = limit

    def fit(self, text: str) -> str:
        if len(text) <= self._limit:
            return text
        structured = self._structured(text)
        if structured is not None:
            return structured
        return self._cut(text)

    def _structured(self, text: str) -> str | None:
        try:
            data = json.loads(text)
        except ValueError:
            return None
        if not isinstance(data, (dict, list)):
            return None
        for strings, items in zip(_STRING_STEPS, _LIST_STEPS + (_LIST_STEPS[-1],) * len(_STRING_STEPS), strict=False):
            shrunk = json.dumps(self._shrink(data, strings, items), ensure_ascii=False, separators=(",", ":"))
            if len(shrunk) <= self._limit:
                return shrunk
        return None

    def _shrink(self, value: Any, strings: int, items: int) -> Any:
        if isinstance(value, str):
            return value if len(value) <= strings else value[:strings].rstrip() + "…"
        if isinstance(value, list):
            return [self._shrink(v, strings, items) for v in value[:items]]
        if isinstance(value, dict):
            return {k: self._shrink(v, strings, items) for k, v in value.items()}
        return value

    def _cut(self, text: str) -> str:
        head = text[: self._limit]
        boundary = max(head.rfind("\n"), head.rfind(" "))
        if boundary > self._limit // 2:
            head = head[:boundary]
        return head + f"\n[truncated: result was longer than {self._limit} characters]"
