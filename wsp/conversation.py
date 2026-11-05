"""Multi-turn runner: one Agent SDK call per turn, each resuming the previous one (the original used
`codex exec resume`).

  python -m wsp.conversation exp3 <language> [--model M] [--sample K]
  python -m wsp.conversation exp6 budget_<0, 10, 25 or 50> [--model M] [--sample K]
  python -m wsp.conversation exp7 <round>/<pair> [--model M] [--sample K]     e.g. attention_tool/pair2
  python -m wsp.conversation exp8 <run> [--model M] [--sample K]              run: p10c, p25c or p50c

Run folder: runs/<model>/<exp>/<item>/s<K>/ with turns/tNNN/{prompt.txt, answer.md, thinking.md, messages.jsonl,
run.json}, tool_calls.jsonl, final_answer.md and state.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from wsp.runner import ITEMS, MODELS, RUNS, _versions, api_key, call, save_reply, scrub_inherited_env


def converse(model: str, exp: str, item: str, sample: int, first_message: str, make_step) -> Path:
    """`make_step()` is called once the run folder exists; the `step(answer)` it returns gives the next message (or
    (message, True) to start a new conversation with it), or None when the answer is final; `step.close()`, if
    present, is called at the end."""
    effort = ITEMS[exp][item][1]
    key = api_key()
    out = RUNS / model / exp / item / f"s{sample}"
    if out.exists():
        sys.exit(f"{out} already exists; refusing to overwrite.")
    out.mkdir(parents=True)
    step = make_step()
    state = {"experiment": exp, "item": item, "sample": sample, "model": model, "effort": effort,
             "versions": _versions(), "turns": 0, "done": False}
    tmp = Path(tempfile.mkdtemp(prefix="wsp_conv_"))
    (tmp / "cwd").mkdir()
    session = last_uuid = None
    session_cost = spent = 0.0
    message = first_message
    try:
        while True:
            state["turns"] += 1
            d = out / "turns" / f"t{state['turns']:03d}"
            d.mkdir(parents=True)
            (d / "prompt.txt").write_text(message, encoding="utf-8")
            msgs, result, answer, attempts = call(model, effort, message, key, tmp / "cwd", tmp / "home",
                                                  session, last_uuid)
            rec = {"attempts": attempts, **save_reply(d, msgs, answer)}
            # Claude Code reports the cost of the whole resumed session; keep it and store this turn's own cost. A retry
            # forks the session, and the fork reports everything this conversation spent so far (p10c turn 42).
            base = spent if len(attempts) > 1 and last_uuid else session_cost
            rec["session_cost_usd"] = rec["total_cost_usd"] or 0
            rec["total_cost_usd"] = rec["session_cost_usd"] - base if rec["session_cost_usd"] >= base \
                else rec["session_cost_usd"]
            session_cost = rec["session_cost_usd"]
            spent += rec["total_cost_usd"]
            (d / "run.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
            session, last_uuid = rec["session_id"], rec["last_uuid"]
            print(f"turn {state['turns']}: ${rec['total_cost_usd'] or 0:.2f}, {attempts[-1]['wall_s']} s", flush=True)
            message = step(answer)
            if isinstance(message, tuple):
                message, fresh = message
                if fresh:
                    session = last_uuid = None
                    session_cost = 0.0
            if message is None:
                (out / "final_answer.md").write_text(answer, encoding="utf-8")
                state["done"] = True
                break
    except RuntimeError as e:
        state["stopped"] = str(e)
        print("stopped:", e, flush=True)
    finally:
        if hasattr(step, "close"):
            step.close()
        turns = sorted((out / "turns").glob("t*/run.json"))
        state["cost_usd"] = round(sum(json.loads(p.read_text()).get("total_cost_usd") or 0 for p in turns), 4)
        (out / "state.json").write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")
        shutil.rmtree(tmp, ignore_errors=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["exp3", "exp6", "exp7", "exp8"])
    ap.add_argument("item")
    ap.add_argument("--model", choices=MODELS, default=MODELS[0])
    ap.add_argument("--sample", type=int, default=1)
    a = ap.parse_args()
    if a.item not in ITEMS[a.exp]:
        sys.exit(f"unknown item {a.item!r}; choose from {list(ITEMS[a.exp])}")
    scrub_inherited_env()
    out = RUNS / a.model / a.exp / a.item / f"s{a.sample}"
    if a.exp == "exp3":
        from wsp import exp3
        first, step = exp3.message(), lambda: exp3.Turns(a.item, out)
    elif a.exp == "exp6":
        from wsp import exp6
        budget = int(a.item.split("_")[1])
        first, step = exp6.message(budget), lambda: exp6.Turns(budget, out)
    elif a.exp == "exp7":
        from wsp import exp7
        rnd, pair = a.item.split("/")
        first, step = exp7.message(rnd), lambda: exp7.Turns(rnd, pair, out)
    else:
        from wsp import exp8
        first, step = exp8.message(a.item), lambda: exp8.Turns(a.item, out, a.sample)
    converse(a.model, a.exp, a.item, a.sample, first, step)
