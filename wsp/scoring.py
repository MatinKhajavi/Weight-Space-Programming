"""Scorers for experiments 1 and 2 (single-turn, open answer).

The original scorers are not released; their READMEs say only that the answer line is mapped to a label by keywords
(exp1: Frogger counts as Freeway; exp2: circle/disc/round/oval, square/rectangle/box, triangle/pyramid/wedge).
The rules below are ours and are documented in docs/protocol.md. They are frozen before any Opus answer is scored.

Rules
1. The answer field is the LAST line that starts with the field label (`Game` / `Images`), allowing a markdown
   heading (`#`), bold (`**`) and an optional colon. Its text is what follows the label on that line, or, if nothing
   follows, the next non-empty line.
2. The text is lowercased and markdown emphasis (`*`, `_`, backticks) is removed.
3. Keywords are matched as whole words (so "round" does not match "background").
4. Exactly one label must be named. No label, or two different labels, counts as wrong (recorded as "unmapped" or
   "hedged").
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

GAME_KEYWORDS = {
    "breakout": ["breakout", "break out", "arkanoid", "brick breaker", "brick-breaker", "brick breaking",
                 "brick-breaking"],
    "pong": ["pong"],
    "pacman": ["pac-man", "pacman", "pac man"],
    "freeway": ["freeway", "frogger"],
}

SHAPE_KEYWORDS = {
    "circle": ["circle", "circles", "circular", "disc", "discs", "disk", "disks", "round", "oval", "ovals"],
    "square": ["square", "squares", "rectangle", "rectangles", "rectangular", "box", "boxes"],
    "triangle": ["triangle", "triangles", "triangular", "pyramid", "pyramids", "wedge", "wedges"],
}

_EMPHASIS = re.compile(r"[*_`]")


def answer_field(text: str, label: str) -> str | None:
    """Return the text of the last `label` field, or None if the field is missing."""
    head = re.compile(rf"^\s*#*\s*(?:\*\*)?\s*{label}\s*(?:\*\*)?\s*(?::|$)\s*(?:\*\*)?\s*(.*)$", re.IGNORECASE)
    lines = text.splitlines()
    found = None
    for i, line in enumerate(lines):
        m = head.match(line)
        if not m:
            continue
        rest = m.group(1).strip()
        if not rest:
            rest = next((ln.strip() for ln in lines[i + 1:] if ln.strip()), "")
        found = rest
    return found


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", _EMPHASIS.sub("", s)).strip().lower()


def map_label(answer: str, keywords: dict[str, list[str]]) -> tuple[str | None, str, list[str]]:
    """Map normalized answer text to a label. Returns (label or None, status, labels named)."""
    named = []
    for label, words in keywords.items():
        if any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", answer) for w in words):
            named.append(label)
    if len(named) == 1:
        return named[0], "mapped", named
    return None, ("unmapped" if not named else "hedged"), named


def confidence(text: str) -> int | None:
    m = re.findall(r"confidence\s*(?:\*\*)?\s*:?\s*(?:\*\*)?\s*(\d{1,3})\s*%", text, re.IGNORECASE)
    return int(m[-1]) if m else None


def score(text: str, label: str, keywords: dict[str, list[str]], truth: str) -> dict:
    field = answer_field(text, label)
    answer = normalize(field) if field is not None else ""
    guess, status, named = map_label(answer, keywords) if field is not None else (None, "no_field", [])
    return {"true": truth, "guess": guess, "correct": guess == truth, "status": status, "named": named,
            "answer_line": answer[:200], "confidence": confidence(text)}


def score_exp1(text: str, truth: str) -> dict:
    return score(text, "game", GAME_KEYWORDS, truth)


def score_exp2(text: str, truth: str) -> dict:
    return score(text, "images", SHAPE_KEYWORDS, truth)


if __name__ == "__main__":
    exp, path, truth = sys.argv[1:4]
    fn = {"exp1": score_exp1, "exp2": score_exp2}[exp]
    print(fn(Path(path).read_text(encoding="utf-8"), truth))
