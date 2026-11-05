"""Pieces the turn handlers of the multi-turn experiments (3, 6, 7, 8) share."""
from __future__ import annotations

import json
import re
from pathlib import Path


def code_blocks(answer: str, kinds: str, info: str = r"[ \t]*") -> list[re.Match]:
    """The reply's ```<kind> blocks in order, as matches of (kind, code). A fence counts only at the start of a line;
    `info` is what may follow the kind on the opening line."""
    return list(re.finditer(rf"^[ \t]*```({kinds}){info}\n(.*?)^[ \t]*```", answer, re.S | re.M))


def explained(answer: str, block: re.Match) -> bool:
    """True if the reply has text before the block."""
    return bool(answer[:block.start()].strip())


class ToolLog:
    """tool_calls.jsonl: one JSON line per tool call or harness event."""

    def __init__(self, path: Path):
        self.path = path

    def __call__(self, rec: dict):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
