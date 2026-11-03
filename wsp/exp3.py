"""Experiment 3: Python sandbox for the multi-turn conversation, and the scorer for its final answer.

Sandbox (from upstream exp3_language/runs/*/message.txt): each run is a fresh process with numpy as np and W, the dict of
block weights; 600 s per run; output cut at 20,000 characters; at most 100 runs; no network.
Truth: upstream eval/key.json (corpus) and eval/heads.json; copying heads are heads with induct > 0.2, as in the post.
"""
from __future__ import annotations

import fcntl
import json
import re
import tempfile
import subprocess
import time
import uuid
from pathlib import Path

UP = Path(__file__).resolve().parents[1] / "upstream" / "exp3_language"
NETS = {"python": "python_L4_H6_d384_s0", "romaji": "romaji_L4_H6_d384_s0", "english": "english_L4_H6_d384_s0"}
MAX_RUNS, TIMEOUT, MAX_OUT = 100, 600, 20_000
IMAGE = "wsp-sandbox"
LOCK = Path(tempfile.gettempdir()) / "wsp-sandbox.lock"
PRELUDE = 'import numpy as np\nW = dict(np.load("/data/blocks.npz"))\n'
LABELS = ("previous", "local", "broad", "copy", "unclear")


def message() -> str:
    return (UP / "runs" / "python_r0" / "message.txt").read_text(encoding="utf-8")


def run_code(code: str, language: str) -> str:
    """Run one code block in a fresh container, as in upstream harness/sandbox.py: stdout, then stderr's tail."""
    blocks = UP / "networks" / language / "blocks.npz"
    name = f"wsp-{uuid.uuid4().hex[:12]}"
    cmd = ["docker", "run", "--rm", "-i", "--name", name, "--network", "none", "--cpus", "8", "--memory", "7g",
           "-v", f"{blocks}:/data/blocks.npz:ro", IMAGE, "python", "-I", "-"]
    with open(LOCK, "w") as lock:   # one run at a time across conversations, so each gets the whole sandbox
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            p = subprocess.run(cmd, input=PRELUDE + code, capture_output=True, text=True, timeout=TIMEOUT)
            out = p.stdout + (("\n[stderr]\n" + p.stderr[-6000:]) if p.stderr.strip() else "")
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            out = f"[timed out after {TIMEOUT} s]"
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + f"\n[output cut at {MAX_OUT:,} characters]"
    return out or "(no output)"


def step(language: str, out: Path):
    """Turn handler: run the reply's code block and return the output message; None when the reply has no block.

    The original replies always had exactly one block and nothing else, and never reached the run limit, so the
    handling of several blocks (run the first) and of the limit (ask for the final answer) is ours."""
    runs = 0

    def handle(answer: str) -> str | None:
        nonlocal runs
        blocks = re.findall(r"```python[^\n]*\n(.*?)```", answer, re.S)
        if not blocks:
            return None
        if runs >= MAX_RUNS:
            return f"You have used all {MAX_RUNS} runs. Give your final answer now, without a code block."
        runs += 1
        t0 = time.time()
        output = run_code(blocks[0], language)
        with open(out / "tool_calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"run": runs, "blocks_in_reply": len(blocks), "seconds": round(time.time() - t0, 1),
                                "code": blocks[0], "out": output}) + "\n")
        return f"OUTPUT of run {runs} of at most {MAX_RUNS}:\n```\n{output}\n```"

    return handle


def parse(answer: str) -> dict:
    heads = {f"L{l}H{h}": lab.lower() for l, h, lab in
             re.findall(r"^\W*L(\d)\s*H(\d)\W*:?\W*(" + "|".join(LABELS) + r")\b", answer, re.I | re.M)}
    field = lambda k: (re.findall(rf"^\W*{k}\W*:\s*\**\s*([a-z ]+?)\s*\**\s*$", answer, re.I | re.M) or [None])[-1]
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
