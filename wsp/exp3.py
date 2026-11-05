"""Experiment 3: Python sandbox for the multi-turn conversation, and the scorer for its final answer.

Sandbox (from upstream exp3_language/runs/*/message.txt): each run is a fresh process with numpy as np and W, the dict
of block weights; 600 s per run; output cut at 20,000 characters; at most 100 runs; no network.
Truth: upstream eval/key.json (corpus) and eval/heads.json; copying heads are heads with induct > 0.2, as in the post.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from wsp import sandbox
from wsp.turns import ToolLog, code_blocks

UP = Path(__file__).resolve().parents[1] / "upstream" / "exp3_language"
NETS = {"python": "python_L4_H6_d384_s0", "romaji": "romaji_L4_H6_d384_s0", "english": "english_L4_H6_d384_s0"}
MAX_RUNS, TIMEOUT, MAX_OUT = 100, 600, 20_000
PRELUDE = 'import numpy as np\nW = dict(np.load("/data/blocks.npz"))\n'
LABELS = ("previous", "local", "broad", "copy", "unclear")


def message() -> str:
    return (UP / "runs" / "python_r0" / "message.txt").read_text(encoding="utf-8")


def run_code(code: str, language: str) -> str:
    blocks = UP / "networks" / language / "blocks.npz"
    return sandbox.run(code, PRELUDE, {"/data/blocks.npz": (blocks, "ro")}, timeout=TIMEOUT, max_out=MAX_OUT)


class Turns:
    """Turn handler: run the reply's code block and return the output message; None when the reply has no block.

    The original replies always had exactly one block and nothing else, and never reached the run limit, so the
    handling of several blocks (run the first) and of the limit (ask for the final answer) is ours."""

    def __init__(self, language: str, out: Path):
        self.language, self.runs = language, 0
        self.log = ToolLog(out / "tool_calls.jsonl")

    def __call__(self, answer: str) -> str | None:
        blocks = code_blocks(answer, "python", info=r"[^\n]*")
        if not blocks:
            return None
        if self.runs >= MAX_RUNS:
            return f"You have used all {MAX_RUNS} runs. Give your final answer now, without a code block."
        self.runs += 1
        code, t0 = blocks[0].group(2), time.time()
        output = run_code(code, self.language)
        self.log({"run": self.runs, "blocks_in_reply": len(blocks), "seconds": round(time.time() - t0, 1),
                  "code": code, "out": output})
        return f"OUTPUT of run {self.runs} of at most {MAX_RUNS}:\n```\n{output}\n```"


HEAD_LIST = r"L\d\s*H\d(?:\s*(?:[-–—,/&]+|to|and)\s*(?:L\d\s*)?H\d)*"


def _heads(spec: str) -> list[str]:
    """'L2H1', 'L1H0 – L1H5', 'L2H0 to L2H5', 'L2H0, L2H2', 'L0H0/H1' -> head names (a range stays in its layer)."""
    out, layer, last, span = [], None, None, False
    for lay, h, sep in re.findall(r"(?:L(\d)\s*)?H(\d)|([-–—]+|\bto\b)", spec):
        if sep:
            span = True
            continue
        layer, h = int(lay) if lay else layer, int(h)
        new = range(last[1] + 1, h + 1) if span and last and last[0] == layer else [h]
        out += [f"L{layer}H{x}" for x in new]
        last, span = (layer, h), False
    return out


def parse(answer: str) -> dict:
    labels = "|".join(LABELS)
    heads = {}
    # The prompt asks for one line per head; answers also group them ("L1H0 – L1H5 local", "L2H0, L2H2: broad") or put
    # several on one line ("L2H3: unclear. L2H4: unclear.").
    for line in answer.splitlines():
        for part in re.split(r"(?<=[.;])\s+(?=\W*L\d\s*H\d)", line):
            m = re.match(rf"^\W*({HEAD_LIST})\W*:?\W*({labels})\b", part, re.I)
            if m:
                heads.update({k: m.group(2).lower() for k in _heads(m.group(1))})

    def field(name: str) -> str | None:
        return (re.findall(rf"^\W*{name}\W*:\s*\**\s*([a-z ]+?)\W*$", answer, re.I | re.M) or [None])[-1]

    probs = re.findall(r"^\W*probabilities\W*:.*$", answer, re.I | re.M)
    corpus = field("corpus")
    return {"corpus_guess": corpus.lower() if corpus else None, "order_guess": (field("order") or "").lower() or None,
            "probabilities": probs[-1].strip() if probs else None, "head_labels": heads,
            "copy_named": sorted(k for k, v in heads.items() if v == "copy")}


def score(answer: str, language: str) -> dict:
    heads = json.loads((UP / "eval" / "heads.json").read_text())[NETS[language]]
    true_copy = sorted(k for k, v in heads.items() if v["induct"] > 0.2)
    res = parse(answer)
    named = set(res["copy_named"])
    res.update({"corpus": language, "corpus_correct": res["corpus_guess"] == language, "true_copy_heads": true_copy,
                "copy_found": sorted(named & set(true_copy)), "copy_false": sorted(named - set(true_copy)),
                "copy_missed": sorted(set(true_copy) - named)})
    return res
