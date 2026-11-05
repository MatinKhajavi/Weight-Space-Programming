"""Replay every logged multi-turn conversation through its turn handler and check that each message the handler writes
is the logged one. The tools are not run: they answer from the run's tool_calls.jsonl (and, for exp8's sandbox state at
a handoff or restart, from the logged message itself)."""
import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wsp import exp3, exp6, exp7, exp8, sandbox

RUNS = Path(__file__).resolve().parents[1] / "runs"
CONVERSATIONS = sorted([*RUNS.glob("*/exp3/*/s*"), *RUNS.glob("*/exp6/budget_*/s*"), *RUNS.glob("*/exp7/*/pair*/s*"),
                        *RUNS.glob("*/exp8/*/s*")])


def records(conv: Path, *kinds: str) -> list[dict]:
    path = conv / "tool_calls.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [r for r in rows if not kinds or r.get("kind", "python") in kinds]


def untimed(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in ("t", "s", "seconds")} for r in rows]


def turns(conv: Path) -> list[Path]:
    return sorted(d for d in (conv / "turns").iterdir() if d.is_dir())


def fake_runs(monkeypatch, conv: Path):
    """The Docker sandbox answers each Python run with its logged output."""
    outs = iter(r["out"] for r in records(conv, "python"))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: next(outs))


def exp3_handler(conv, monkeypatch, tmp_path):
    fake_runs(monkeypatch, conv)
    return exp3.message(), exp3.Turns(conv.parent.name, tmp_path)


def _move(name: str) -> int:
    if name == "pass":
        return exp6.PASS
    r, c = map(int, name.strip("()").split(","))
    return 9 * r + c


def _pass_result(rec: dict) -> dict:
    """What the Modal pass function returned, rebuilt from the logged result (only the parts the handler reads)."""
    res = rec["result"]
    if rec["kind"] == "games":
        return {"per_opponent": res["per_opponent"], "games": [], "boards": np.zeros(1), "valid": np.zeros(1)}
    probs = np.zeros((len(res["top3_per_position"]), exp6.PASS + 1))
    for i, top3 in enumerate(res["top3_per_position"]):
        for j, (name, p) in enumerate(top3):
            probs[i, _move(name)] = p + (3 - j) * 1e-9   # the logged order, ties included
    return {"probs": probs}


def exp6_handler(conv, monkeypatch, tmp_path):
    fake_runs(monkeypatch, conv)
    local = ("AssertionError", "JSONDecodeError", "KeyError", "TypeError")   # raised before the pass function runs
    passes = iter(r for r in records(conv, "games", "positions") if "result" in r or not r["error"].startswith(local))

    def remote(fn, *args):
        rec = next(passes)
        if "error" in rec:
            name, _, text = rec["error"].partition(": ")
            raise type(name, (Exception,), {})(text)
        return _pass_result(rec)

    monkeypatch.setattr(exp6, "_remote", remote)
    monkeypatch.setattr(exp6, "saved_net", lambda work: None)
    monkeypatch.setattr(exp6, "practice_opponents", lambda: dict.fromkeys(["random", "weak", "strong"]))
    budget = int(conv.parent.name.split("_")[1])
    return exp6.message(budget), exp6.Turns(budget, tmp_path)


def exp7_handler(conv, monkeypatch, tmp_path):
    fake_runs(monkeypatch, conv)
    calls = iter(r["out"] for r in records(conv, "attention") if not r["out"].startswith("The attention block is not"))
    monkeypatch.setattr(exp7, "attention", lambda *a: next(calls))
    monkeypatch.setattr(exp7, "case", lambda rnd, pair: tmp_path / "case")
    rnd = conv.parents[1].name
    return exp7.message(rnd), exp7.Turns(rnd, conv.parent.name, tmp_path)


class FakeKernel:
    """exp8's sandbox kernel: Python runs answer from the log; the state shown at a restart or handoff comes from the
    logged message of the next turn."""

    def __init__(self, conv: Path):
        self.outs = iter(r["out"] for r in records(conv, "python"))
        self.prompts = [(d / "prompt.txt").read_text(encoding="utf-8") for d in turns(conv)]
        self.turn = 0
        self.sb = SimpleNamespace(object_id="sb", filesystem=SimpleNamespace(write_text=lambda *a: None))

    def run(self, code: str) -> str:
        return next(self.outs)

    def fresh(self) -> bool:   # called once per non-final turn
        self.turn += 1
        return exp8.RESTARTED in self.prompts[self.turn]

    def ls(self, path: str) -> list[str]:
        nxt = self.prompts[self.turn]
        if path == "/work/edited":
            ed = re.search(r"Tensors currently edited: (.*?)\. Files in /work", nxt).group(1)
            return [] if ed == "none" else [e + ".npy" for e in ast.literal_eval(ed)]
        return ast.literal_eval(re.search(r"Files in /work: (\[.*?\])\.\nYOUR NOTEBOOK", nxt, re.S).group(1))

    def text(self, path: str) -> str:
        return re.search(r"last 8,000 characters\):\n```\n(.*?)\n```\nContinue from here", self.prompts[self.turn],
                         re.S).group(1)


def exp8_handler(conv, monkeypatch, tmp_path):
    results = {r["k"]: r["result"] for r in records(conv, "forward")}
    monkeypatch.setattr(exp8, "Kernel", lambda volume: FakeKernel(conv))
    monkeypatch.setattr(exp8, "run_forward", lambda sandbox_id, k, req: results[k])
    run = conv.parent.name
    return exp8.message(run), exp8.Turns(run, tmp_path, 1)


HANDLERS = {"exp3": exp3_handler, "exp6": exp6_handler, "exp7": exp7_handler, "exp8": exp8_handler}


@pytest.mark.parametrize("conv", CONVERSATIONS, ids=lambda c: str(c.relative_to(RUNS)))
def test_replay(conv, monkeypatch, tmp_path):
    exp = conv.relative_to(RUNS).parts[1]
    first, handle = HANDLERS[exp](conv, monkeypatch, tmp_path)
    ds = turns(conv)
    assert (ds[0] / "prompt.txt").read_text(encoding="utf-8") == first
    for i, d in enumerate(ds):
        got = handle((d / "answer.md").read_text(encoding="utf-8"))
        got = got[0] if isinstance(got, tuple) else got
        want = (ds[i + 1] / "prompt.txt").read_text(encoding="utf-8") if i + 1 < len(ds) else None
        if exp == "exp8" and i + 2 == exp8.EXPLAIN_AT[conv.parent.name]:
            # fixed after the runs: upstream's switch turn carries the reminder; all three of our replies before the
            # switch had no explanation, so all three get it (protocol.md)
            want += exp8.REMINDER
        assert got == want, f"{d.name}: the handler's next message differs from the logged one"
    assert untimed(records(tmp_path)) == untimed(records(conv)), "tool_calls.jsonl differs"
    if exp == "exp7":
        prediction = [json.loads((d / "prediction.json").read_text()) for d in (tmp_path, conv)]
        assert prediction[0] == prediction[1]
