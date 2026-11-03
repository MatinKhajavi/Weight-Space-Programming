"""Single-turn runner: send one upstream prompt to Claude Opus 5.5 through the Agent SDK, set up as in
docs/protocol.md, and log the run.

  python -m wsp.runner run <exp> <item> [--sample K]
  python -m wsp.runner score <exp> <item> [--sample K]

Run folder: runs/<exp>/<item>/s<K>/ with answer.md, thinking.md, messages.jsonl, run.json (and score.json).
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream"
RUNS = ROOT / "runs"
MODEL = "claude-opus-5-5"
SYSTEM_PROMPT = "Answer the user's request."
BUDGET_USD_PER_CALL = 20.0

# experiment -> item -> (prompt file in upstream, Astra's effort)
ITEMS = {
    "exp1": {g: (f"exp1_which_game/runs/{g}/prompt.txt", "high") for g in ["breakout", "pong", "pacman", "freeway"]},
    "exp2": {s: (f"exp2_diffusion/runs/{s}/prompt.txt", "high") for s in ["circle", "square", "triangle"]},
    "exp4": {"oneshot_repair": ("exp4_fix_car/runs/oneshot_repair/reply/prompt.txt", "high")},
    "exp5": {g: (f"exp5_write_from_scratch/runs/{g}/reply/prompt.txt", "xhigh") for g in
             ["breakout", "seaquest", "asterix"]},
}

# Retry waits observed in the original harness logs.
WAITS_FAILURE = [60, 120, 240, 480]
WAITS_LIMIT = [900, 1350, 2025, 3038]
NOISE = {"thinking_tokens", "ui_invalidate"}  # streaming progress events, no content


def scrub_inherited_env() -> list[str]:
    """The SDK passes its parent's environment to Claude Code; drop everything Claude/Anthropic-related."""
    gone = sorted(k for k in os.environ if k.startswith(("CLAUDE", "ANTHROPIC")))
    for k in gone:
        del os.environ[k]
    return gone


def read_dotenv(path: Path = ROOT / ".env") -> dict[str, str]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def options(effort: str, cwd: Path, home: Path, env_extra: dict[str, str]):
    from claude_agent_sdk import ClaudeAgentOptions
    return ClaudeAgentOptions(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        tools=[],
        setting_sources=[],
        strict_mcp_config=True,
        verbatim_prompts=True,
        extra_args={"bare": None},
        effort=effort,
        thinking={"type": "adaptive", "display": "summarized"},
        max_buffer_size=16 * 1024 * 1024,
        max_turns=1,
        max_budget_usd=BUDGET_USD_PER_CALL,
        cwd=str(cwd),
        env={"CLAUDE_CONFIG_DIR": str(home), "DISABLE_AUTO_COMPACT": "1",
             "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000", "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1",
             "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", **env_extra},
    )


def _plain(m) -> dict:
    try:
        return {"type": type(m).__name__, **dataclasses.asdict(m)}
    except Exception:
        return {"type": type(m).__name__, "repr": repr(m)[:5000]}


async def _ask(prompt: str, opts) -> list[dict]:
    from claude_agent_sdk import query
    return [_plain(m) async for m in query(prompt=prompt, options=opts)]


def _versions() -> dict:
    import claude_agent_sdk
    from claude_agent_sdk._cli_version import __cli_version__
    return {"claude_agent_sdk": claude_agent_sdk.__version__, "claude_code_cli": __cli_version__}


def _prompt(exp: str, item: str) -> tuple[str, str, str]:
    rel, effort = ITEMS[exp][item]
    return (UP / rel).read_text(encoding="utf-8"), effort, rel


def _count_tokens(prompt: str, key: str) -> int | None:
    try:
        import anthropic
        r = anthropic.Anthropic(api_key=key).messages.count_tokens(
            model=MODEL, system=SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}])
        return r.input_tokens
    except Exception as e:
        print("count_tokens failed:", type(e).__name__, str(e)[:200])
        return None


def run(exp: str, item: str, sample: int) -> Path:
    prompt, effort, rel = _prompt(exp, item)
    key = read_dotenv().get("ANTHROPIC_API_KEY", "")
    if not key.startswith("sk-ant-"):
        sys.exit("No ANTHROPIC_API_KEY in .env (expected a value starting with sk-ant-). Nothing was sent.")
    out = RUNS / exp / item / f"s{sample}"
    if out.exists():
        sys.exit(f"{out} already exists; refusing to overwrite.")
    record = {"experiment": exp, "item": item, "sample": sample, "model": MODEL, "effort": effort,
              "system_prompt": SYSTEM_PROMPT, "prompt_file": f"upstream/{rel}",
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_chars": len(prompt),
              "prompt_tokens": _count_tokens(prompt, key), "versions": _versions(), "attempts": []}
    for attempt in range(len(WAITS_FAILURE) + 1):
        tmp = Path(tempfile.mkdtemp(prefix="wsp_run_"))
        (tmp / "cwd").mkdir()
        t0 = time.time()
        msgs, err = [], None
        try:
            msgs = asyncio.run(_ask(prompt, options(effort, tmp / "cwd", tmp / "home", {"ANTHROPIC_API_KEY": key})))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:2000]
        shutil.rmtree(tmp, ignore_errors=True)
        wall = round(time.time() - t0, 1)
        result = next((m for m in msgs if m["type"] == "ResultMessage"), {})
        texts = [b["text"] for m in msgs if m["type"] == "AssistantMessage" for b in m.get("content", []) if "text" in b]
        answer = result.get("result") or "\n".join(texts)
        stop = [m.get("stop_reason") for m in msgs if m["type"] == "AssistantMessage"]
        record["attempts"].append({"wall_s": wall, "error": err, "is_error": result.get("is_error"),
                                   "stop_reasons": stop, "answer_chars": len(answer or "")})
        refused = "refusal" in stop
        ok = err is None and not result.get("is_error") and (answer.strip() or refused)
        if ok:
            break
        reason = err or json.dumps(result.get("errors") or result.get("result") or "empty reply")[:500]
        status = re.search(r"API Error: (\d{3})", reason)
        if status and status.group(1).startswith("4") and status.group(1) != "429":
            sys.exit(f"request rejected, not retrying: {reason}")
        if attempt == len(WAITS_FAILURE):
            sys.exit(f"giving up after {attempt + 1} attempts: {reason}")
        limit = re.search(r"rate.?limit|429|usage limit|overloaded", reason, re.I)
        wait = (WAITS_LIMIT if limit else WAITS_FAILURE)[attempt]
        print(f"attempt {attempt + 1} failed ({reason}); new conversation in {wait} s", flush=True)
        time.sleep(wait)

    out.mkdir(parents=True)
    init = next((m for m in msgs if m.get("subtype") == "init"), {}).get("data", {})
    tool_uses = [b for m in msgs if m["type"] == "AssistantMessage" for b in m.get("content", []) if "input" in b]
    thinking = [b["thinking"] for m in msgs if m["type"] == "AssistantMessage" for b in m.get("content", [])
                if "thinking" in b and b["thinking"]]
    # With no tools, a user message here was injected by Claude Code after a reply hit the output cap
    # ("Output token limit hit. Resume directly ..."). Keep the reply as it was when cut (docs/protocol.md, deviation 5).
    injected = ["".join(b.get("text", "") for b in m["content"]) for m in msgs if m["type"] == "UserMessage"]
    if injected:
        first = msgs[:next(i for i, m in enumerate(msgs) if m["type"] == "UserMessage")]
        cut = "".join(b.get("text", "") for m in first if m["type"] == "AssistantMessage" for b in m["content"])
        (out / "answer_cut.md").write_text(cut, encoding="utf-8")
    record.update({
        "resumed_after_output_cap": bool(injected), "injected_messages": injected,
        "session_id": result.get("session_id"), "wall_s": wall, "refused": refused,
        "usage": result.get("usage"), "model_usage": result.get("model_usage"),
        "total_cost_usd": result.get("total_cost_usd"), "num_turns": result.get("num_turns"),
        "audit": {"tool_free": not tool_uses and init.get("tools") == [], "init_tools": init.get("tools"),
                  "init_mcp_servers": init.get("mcp_servers"), "api_key_source": init.get("apiKeySource"),
                  "model": init.get("model")},
    })
    (out / "answer.md").write_text(answer or "", encoding="utf-8")
    (out / "thinking.md").write_text("\n\n---\n\n".join(thinking), encoding="utf-8")
    with open(out / "messages.jsonl", "w", encoding="utf-8") as f:
        for m in msgs:
            if m.get("subtype") in NOISE:
                continue
            for b in m.get("content") or []:
                if isinstance(b, dict):
                    b.pop("signature", None)  # opaque blob the API uses to verify thinking; no content
            f.write(json.dumps(m, default=str) + "\n")
    (out / "run.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("experiment", "item", "sample", "wall_s", "usage", "total_cost_usd",
                                             "refused", "prompt_tokens", "audit")}, indent=1, default=str))
    return out


def _score_text(exp: str, item: str, answer: str, out: Path | None) -> dict:
    if exp in ("exp1", "exp2"):
        from wsp.scoring import score_exp1, score_exp2
        return (score_exp1 if exp == "exp1" else score_exp2)(answer, item)
    if exp == "exp5":
        from wsp.exp5 import evaluate, read_reply, save_npz
        w, rep = read_reply(answer)
        if out:
            save_npz(str(out / "network.npz"), w)
            (out / "apply.json").write_text(json.dumps(rep, indent=1))
        return {"apply": {k: rep[k] for k in ("statements", "applied", "nonzero")}, "skipped": len(rep["skipped"]),
                "eval": evaluate(w, item, 1000, 10, seed=0)}
    raise SystemExit(f"no scorer for {exp} yet")


def score(exp: str, item: str, sample: int) -> dict:
    out = RUNS / exp / item / f"s{sample}"
    res = _score_text(exp, item, (out / "answer.md").read_text(encoding="utf-8"), out)
    if (out / "answer_cut.md").exists():
        res["cut_at_output_cap"] = _score_text(exp, item, (out / "answer_cut.md").read_text(encoding="utf-8"), None)
    (out / "score.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("exp", choices=list(ITEMS))
    ap.add_argument("item")
    ap.add_argument("--sample", type=int, default=1)
    a = ap.parse_args()
    if a.item not in ITEMS[a.exp]:
        sys.exit(f"unknown item {a.item!r}; choose from {list(ITEMS[a.exp])}")
    if a.cmd == "run":
        scrub_inherited_env()
        run(a.exp, a.item, a.sample)
    else:
        score(a.exp, a.item, a.sample)
