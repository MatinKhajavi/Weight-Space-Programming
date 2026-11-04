"""Experiment 8: edit Qwen3.5-0.8B's weights to make it better at math (runs p10c, p25c, p50c: 10, 25, 50 passes).

  python -m wsp.exp8 prepare                 the kernel's /model Volume (after python -m wsp.exp8_eval prepare)
  python -m wsp.conversation exp8 p10c

The harness, rebuilt from the upstream logs (runs/<run>/turns, tool_calls.jsonl, requests/): the first message is
upstream turns/t001/prompt.txt; replies hold one ```python block (run in a persistent kernel) or one ```forward
block (a forward pass on a Modal GPU); output templates, the 40-turn handoff with /work/notes.md, the EXPLAIN EVERY
STEP switch (turn 52 / 16 / 29) and the two notices (at the turns upstream delivered them) are copied from the logs.
The kernel runs in a Modal sandbox (CPU, 32 GB, no network) with /work on a Modal Volume.
Our choices where the logs show nothing: the texts sent for a reply without a block before the passes are used,
after 300 Python runs, for a forward block after the last pass, and for a kernel timeout or crash.
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

from wsp.runner import UP

EXP = UP / "exp8_math"
BUDGETS = {"p10c": 10, "p25c": 25, "p50c": 50}
EXPLAIN_AT = {"p10c": 52, "p25c": 16, "p50c": 29}
NOTICES = {"p10c": {59: "001_pass_size", 65: "002_tool_facts"}, "p25c": {25: "001_pass_size", 31: "002_tool_facts"},
           "p50c": {46: "001_pass_size", 51: "002_tool_facts"}}
MAX_RUNS, RUN_TIMEOUT, MAX_OUT, THREAD_TURNS, SANDBOX_HOURS = 300, 900, 10_000, 40, 23.5
MARK = "\x00wsp-kernel\x00"

PARA = ("EXPLAIN EVERY STEP. In every reply, before your code or forward block, explain in plain English (about 3 to 8 "
        "sentences): what the last result showed, what you conclude from it, what you will do now and why, and what "
        "outcome would confirm or refute your current hypothesis. In your final answer, also summarize the path that led "
        "to your edits. These explanations are read by the experimenters.")
SWITCH = f"[note from the experimenters: a new instruction for the rest of the session] {PARA}\n\n"
SENT_BACK = (f"Your reply had no explanation, so its block was NOT run (no Python run or forward pass was used). {PARA} "
             "Send the block again, preceded by your explanation.")
REMINDER = "\n(Reminder: before your next block, explain what this result shows, what you will do next and why.)"
RESTARTED = "[note: the harness was restarted, so your kernel restarted too and its variables are gone]\n"
LAST_PASS = ("\nAll forward passes are used. Make your final choice of edits (Python is still available), then reply "
             "without a code block.")
NUDGE = ("Forward passes left: {left} of {b}. The session continues until all forward passes are used; reply with a "
         "```python or ```forward block.")
NO_PASSES = "All forward passes are used; this block was not run. Make your final choice of edits, then reply without a code block."
NO_RUNS = f"You have used all {MAX_RUNS} Python runs; this block was not run."
TIMED_OUT = f"[timed out after {RUN_TIMEOUT} s; the kernel was restarted and its variables are gone]"
DIED = "[the kernel stopped (out of memory or crashed) and was restarted; its variables are gone]"
EVAL_PROMPT_FORMAT = (  # /work/eval_prompt_format.txt, which came with notice 002 (as printed in p50c)
    "EXACT EVALUATION PROMPT FORMAT\nThe evaluation prompt, exactly as the model sees it (thinking on), for a problem P:\n"
    "PREFIX + P + SUFFIX, with\nPREFIX = '<|im_start|>user\\n'\nSUFFIX = '\\nPlease reason step by step, and put your "
    "final answer within \\\\boxed{}.<|im_end|>\\n<|im_start|>assistant\\n<think>\\n'\nThe model's answer follows SUFFIX "
    "directly. To score a text with a teacher-forced pass, send PREFIX + P + SUFFIX + answer as one raw text in \"texts\" "
    "(\"chats\" re-renders assistant turns and drops their reasoning).\n")


def message(run: str) -> str:
    return (EXP / "runs" / run / "turns" / "t001" / "prompt.txt").read_text(encoding="utf-8")


def shown(raw: str) -> str:
    """As upstream showed outputs: ASCII only, cut at 10,000 characters, then each line at most twice (short lines
    exempt)."""
    out = raw.encode("ascii", "replace").decode("ascii")
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + f"\n[output cut at {MAX_OUT:,} characters]"
    seen, keep, dropped = {}, [], 0
    for line in out.split("\n"):
        key = line.strip()
        seen[key] = seen.get(key, 0) + 1
        if len(key) >= 10 and seen[key] > 2:
            dropped += 1
            continue
        keep.append(line)
    out = "\n".join(keep)
    if dropped:
        out += f"\n[harness: {dropped} repeated lines not shown (a line is shown at most 2 times per output)]"
    return out


def _modal():
    from wsp.exp7 import _modal as m   # the MODAL_PROFILE guard
    return m()


def kernel_image(modal):
    return (modal.Image.debian_slim(python_version="3.12")
            .pip_install("numpy==2.5.4", "scipy==1.18.1", "scikit-learn==1.9.1", "pandas==3.1.0", "sympy==1.14.0",
                         "networkx==3.7", "threadpoolctl==3.7.0", "safetensors==0.8.0", "tokenizers==0.23.3")
            .pip_install("torch==2.15.0", index_url="https://download.pytorch.org/whl/cpu")
            .env({"HOME": "/tmp", "OMP_NUM_THREADS": "8"})
            .add_local_dir(str(EXP / "practice"), "/practice")
            .add_local_file(str(Path(__file__).with_name("exp8_kernel.py")), "/k/kernel.py"))


class Kernel:
    """The persistent kernel: one Python process in a Modal sandbox; /work is a Modal Volume, /model read-only."""

    def __init__(self, work_volume: str):
        self.modal = _modal()
        m = self.modal
        self.app = m.App.lookup("wsp-exp8", create_if_missing=True)
        self.image = kernel_image(m)
        self.volumes = {"/model": m.Volume.from_name("wsp-exp8-model").read_only(),
                        "/work": m.Volume.from_name(work_volume, create_if_missing=True, version=2)}
        self.sb = None
        self.new_sandbox()

    def new_sandbox(self):
        if self.sb is not None:
            self.sb.terminate()
        self.sb = self.modal.Sandbox.create(app=self.app, image=self.image, volumes=self.volumes, cpu=(1, 8), memory=(4096, 32768),
                                            timeout=24 * 3600, block_network=True, workdir="/work")
        self.started = time.time()
        self.start()

    def start(self):
        self.p = self.sb.exec("python", "-u", "/k/kernel.py", bufsize=1)
        self.q = queue.Queue()
        threading.Thread(target=self._read, args=(self.p, self.q), daemon=True).start()

    @staticmethod
    def _read(p, q):
        for line in p.stdout:
            q.put(line)
        q.put(None)

    def restart(self):
        self.sb.exec("pkill", "-f", "/k/kernel.py").wait()
        self.start()

    def fresh(self) -> bool:
        """New sandbox if this one is near its 24 h limit or gone; True if the kernel was restarted."""
        if time.time() - self.started > SANDBOX_HOURS * 3600 or self.sb.poll() is not None:
            self.new_sandbox()
            return True
        return False

    def run(self, code: str) -> str:
        self.p.stdin.write(json.dumps({"code": code}) + "\n")
        self.p.stdin.drain()
        deadline, extra = time.time() + RUN_TIMEOUT, ""
        while True:
            try:
                line = self.q.get(timeout=max(1.0, deadline - time.time()))
            except queue.Empty:
                self.restart()
                return extra + TIMED_OUT
            if line is None:
                self.restart()
                return extra + DIED
            if line.startswith(MARK):
                return extra + json.loads(line[len(MARK):])["out"]
            extra += line   # output written below Python (e.g. by a C library)

    def text(self, path: str) -> str:
        try:
            return self.sb.filesystem.read_text(path)
        except Exception:   # noqa: BLE001 - missing file
            return ""

    def ls(self, path: str) -> list[str]:
        try:
            return sorted(f.name for f in self.sb.filesystem.list_files(path))
        except Exception:   # noqa: BLE001 - missing folder
            return []


def step(run: str, out: Path, sample: int):
    b, explain_at, notices = BUDGETS[run], EXPLAIN_AT[run], NOTICES[run]
    first = message(run)
    with_explain = (EXP / "runs" / run / "message.txt").read_text(encoding="utf-8")
    work_volume = f"wsp-exp8-{run}-s{sample}-{int(time.time())}"
    (out / "work_volume.txt").write_text(work_volume + "\n")
    k = Kernel(work_volume)
    st = {"turn": 1, "thread_turns": 1, "runs": 0, "passes": 0, "requests": 0, "streak": 0, "handoffs": 0}

    def log(rec):
        with open(out / "tool_calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": round(time.time(), 1), "turn": st["turn"], **rec}) + "\n")

    def python(code: str) -> str:
        if st["runs"] >= MAX_RUNS:
            return NO_RUNS
        st["runs"] += 1
        t0 = time.time()
        raw = k.run(code)
        log({"kind": "python", "run": st["runs"], "s": round(time.time() - t0, 1), "code": code, "out": raw[:20000]})
        return f"OUTPUT of Python run {st['runs']} of at most {MAX_RUNS}:\n```\n{shown(raw) or '(no output)'}\n```"

    def forward(body: str) -> str:
        if st["passes"] >= b:
            return NO_PASSES
        st["requests"] += 1
        n, t0 = st["requests"], time.time()
        try:
            req = json.loads(body)
            done = run_forward(k.sb.object_id, n, req)
        except Exception as e:   # noqa: BLE001 - a failed request is reported and not counted
            done = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        log({"kind": "forward", "k": n, "s": round(time.time() - t0, 1), "request": body, "result": done})
        if not done.get("ok"):
            return f"The forward pass failed and was not counted: {done.get('error')}"
        st["passes"] += 1
        where = {"teacher_forced": f"/work/fwd_{n}/"}.get(req["type"]) or (
            f"/work/gen_{n}_original.jsonl and /work/gen_{n}_edited.jsonl" if req.get("model") == "both"
            else f"/work/gen_{n}.jsonl")
        msg = (f"Forward pass {st['passes']} of {b} done ({req['type']}); results in {where}. Summary:\n```\n"
               f"{json.dumps(done, indent=1)}\n```\nForward passes left: {b - st['passes']}.")
        return msg + (LAST_PASS if st["passes"] == b else "")

    def handoff(msg: str, nxt: int) -> str:
        ed = [f[:-4] for f in k.ls("/work/edited") if f.endswith(".npy")]
        notes = k.text("/work/notes.md")
        return (f"{with_explain if nxt > explain_at else first}\n\nTHIS IS A RESTARTED CONVERSATION (to keep it short): "
                f"{nxt - 1} replies happened before. Forward passes used: {st['passes']} of {b}; Python runs used: "
                f"{st['runs']} of {MAX_RUNS}. Your kernel is still running with its variables. Tensors currently edited: "
                f"{repr(ed) if ed else 'none'}. Files in /work: {k.ls('/work')!r}.\nYOUR NOTEBOOK (/work/notes.md, last "
                f"8,000 characters):\n```\n{notes[-8000:]}\n```\nContinue from here.\n\nLast result:\n{msg}")

    def handle(answer: str):
        t, nxt = st["turn"], st["turn"] + 1
        m = re.search(r"^[ \t]*```(python|forward)[ \t]*\n(.*?)^[ \t]*```", answer, re.S | re.M)
        unexplained = m is not None and t >= explain_at and not answer[:m.start()].strip()
        if unexplained and st["streak"] < 2:
            st["streak"] += 1
            log({"kind": "sent_back", "streak": st["streak"]})
            msg = SENT_BACK
        else:
            st["streak"] = 0
            if m is None:
                if st["passes"] >= b:
                    log({"kind": "final", **{x: st[x] for x in ("runs", "passes", "requests", "handoffs")}})
                    return None
                log({"kind": "nudge"})
                msg = NUDGE.format(left=b - st["passes"], b=b)
            else:
                msg = python(m.group(2)) if m.group(1) == "python" else forward(m.group(2))
                msg += REMINDER if unexplained else ""
        if k.fresh():
            msg = RESTARTED + msg
        if nxt in notices:
            if notices[nxt] == "002_tool_facts":
                k.sb.filesystem.write_text(EVAL_PROMPT_FORMAT, "/work/eval_prompt_format.txt")
            msg = f"[note from the experimenters] {(EXP / 'notices' / (notices[nxt] + '.txt')).read_text().strip()}\n\n" + msg
        if nxt == explain_at:
            msg = SWITCH + msg
        fresh = st["thread_turns"] == THREAD_TURNS
        if fresh:
            msg = handoff(msg, nxt)
            st["handoffs"] += 1
            st["thread_turns"] = 1
            log({"kind": "handoff"})
        else:
            st["thread_turns"] += 1
        st["turn"] = nxt
        return msg, fresh

    def close():
        """Stop the sandbox (its final commit saves /work), then keep the submitted edits and the notebook locally."""
        k.sb.terminate()
        dst = out / "work"
        dst.mkdir(exist_ok=True)
        for path in ["edited", "notes.md"]:
            subprocess.run(["modal", "volume", "get", "--force", work_volume, path, str(dst)], env=os.environ)

    handle.close = close
    return handle


GPU, FORWARD_TIMEOUT = "B200", 2 * 60 * 60
LIMITS = {"gen_max": 512, "tf_max_texts": 1024, "tf_max_tokens": 32768, "tf_total_tokens": 524288, "act_max_tokens": 8192,
          "attn_max_texts": 2, "attn_max_tokens": 2048, "max_bytes": 6 << 30}
FULL_ATTENTION = [3, 7, 11, 15, 19, 23]


def _items(req: dict, practice: dict) -> list[dict]:
    rows = []
    for pid in req.get("problems", []):
        if pid not in practice:
            raise ValueError(f"unknown practice id: {pid}")
        rows.append({"id": pid, **practice[pid]})
    rows += [{"id": f"prompt_{i}", "source": "prompt", "user": p} for i, p in enumerate(req.get("prompts", []))]
    return rows


def _generate(llm_path: str, rows: list[dict], n: int, max_tokens: int, tok, Q) -> tuple[list[dict], dict]:
    """vLLM generation with upstream evaluate.py's settings (protocol think35), seeded by item; graded by mathqa."""
    import zlib
    from vllm import LLM, SamplingParams
    llm = LLM(model=llm_path, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=max_tokens + 2048, seed=0)
    prompts = [tok.apply_chat_template([{"role": "user", "content": r.get("user") or Q.build_prompt(r["problem"])}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in rows]
    sp = [SamplingParams(n=n, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=max_tokens,
                         seed=zlib.crc32(r["id"].encode())) for r in rows]
    outs = llm.generate(prompts, sp)
    del llm
    res, fail = [], {"wrong_answer": 0, "no_answer_finished": 0, "cut_at_max_tokens": 0}
    for r, prompt, o in zip(rows, prompts, outs):
        samples = []
        for c in o.outputs:
            cut = c.finish_reason == "length"
            s = {"text": c.text, "pred": None, "correct": None, "tokens": len(c.token_ids), "cut": cut,
                 "_ids": list(c.token_ids)}
            if "answer" in r:
                g = Q.score({"gold": r["answer"]}, c.text)
                s["pred"], s["correct"] = g["pred"], g["correct"]
                if not g["correct"]:
                    fail["cut_at_max_tokens" if cut else "wrong_answer" if g["boxed"] else "no_answer_finished"] += 1
            samples.append(s)
        res.append({"id": r["id"], "source": r["source"], "answer": r.get("answer"), "samples": samples,
                    "_prompt_ids": list(o.prompt_token_ids)})
    graded = [s for x in res for s in x["samples"] if s["correct"] is not None]
    by = {}
    for x in res:
        for s in x["samples"]:
            if s["correct"] is not None:
                by.setdefault(x["source"], []).append(s["correct"])
    allg = [s for x in res for s in x["samples"]]
    summary = {"generations": len(allg), "graded": len(graded),
               "accuracy": round(sum(s["correct"] for s in graded) / max(1, len(graded)), 4),
               "accuracy_by_source": {k: round(sum(v) / len(v), 4) for k, v in sorted(by.items())},
               "mean_tokens": int(sum(s["tokens"] for s in allg) / max(1, len(allg))),
               "cut_share": round(sum(s["cut"] for s in allg) / max(1, len(allg)), 4), "failures": fail}
    return res, summary


def _hf(path: str):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()


def _capture(model, layers: list[int], mlp: list[int], attn: list[int]):
    """Hooks: residual stream after layer l (0 = embeddings), down_proj inputs, attention probabilities."""
    got, hooks = {}, []
    base = model.model
    if 0 in layers:
        hooks.append(base.embed_tokens.register_forward_hook(lambda m, i, o: got.__setitem__("hidden_0", o[0])))
    for l in layers:
        if l > 0:
            hooks.append(base.layers[l - 1].register_forward_hook(
                lambda m, i, o, l=l: got.__setitem__(f"hidden_{l}", (o[0] if isinstance(o, tuple) else o)[0])))
    for l in mlp:
        hooks.append(base.layers[l].mlp.down_proj.register_forward_pre_hook(
            lambda m, i, l=l: got.__setitem__(f"mlp_act_{l}", i[0][0])))
    for l in attn:
        hooks.append(base.layers[l].self_attn.register_forward_hook(
            lambda m, i, o, l=l: got.__setitem__(f"attn_{l}", o[1][0])))
    return got, hooks


def _activations(path: str, res: list[dict], layers: list[int], mlp: bool, dst: str):
    """Per generated answer: residual stream (and down_proj inputs) at the last prompt token and averaged over the
    answer's first 8,192 tokens."""
    import numpy as np
    import torch
    model = _hf(path)
    mlp_layers = [l for l in layers if l <= 23] if mlp else []
    got, hooks = _capture(model, layers, mlp_layers, [])
    hl, ha, ml, ma, index = [], [], [], [], []
    with torch.no_grad():
        for x in res:
            p = x["_prompt_ids"]
            for j, s in enumerate(x["samples"]):
                a = s["_ids"][:LIMITS["act_max_tokens"]]
                got.clear()
                model.model(torch.tensor([p + a], device="cuda"))
                cut = len(p) - 1
                hl.append(torch.stack([got[f"hidden_{l}"][cut] for l in layers]).float().cpu())
                ha.append(torch.stack([got[f"hidden_{l}"][len(p):].float().mean(0) for l in layers]).cpu())
                if mlp_layers:
                    ml.append(torch.stack([got[f"mlp_act_{l}"][cut] for l in mlp_layers]).float().cpu())
                    ma.append(torch.stack([got[f"mlp_act_{l}"][len(p):].float().mean(0) for l in mlp_layers]).cpu())
                index.append({"item": x["id"], "sample": j, "correct": s["correct"], "tokens": len(a)})
    for h in hooks:
        h.remove()
    os.makedirs(dst, exist_ok=True)
    np.save(f"{dst}/hidden_last_prompt.npy", torch.stack(hl).numpy().astype(np.float16))
    np.save(f"{dst}/hidden_mean_answer.npy", torch.stack(ha).numpy().astype(np.float16))
    if mlp_layers:
        np.save(f"{dst}/mlp_last_prompt.npy", torch.stack(ml).numpy().astype(np.float16))
        np.save(f"{dst}/mlp_mean_answer.npy", torch.stack(ma).numpy().astype(np.float16))
    json.dump({"layers": layers, "mlp_layers": mlp_layers, "samples": index}, open(f"{dst}/index.json", "w"))
    del model, got
    torch.cuda.empty_cache()   # the next vLLM engine ("both") needs the GPU memory back


def _teacher_forced(path: str, req: dict, dst: str, edited: list[str]) -> dict:
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if "chats" in req:
        texts = [tok.apply_chat_template(c, tokenize=False, enable_thinking=bool(req.get("enable_thinking")))
                 for c in req["chats"]]
    else:
        texts = list(req["texts"])
    ret = req.get("return", {})
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    lengths = [len(i) for i in ids]
    if not 0 < len(ids) <= LIMITS["tf_max_texts"]:
        raise ValueError(f"teacher_forced takes 1 to {LIMITS['tf_max_texts']} texts")
    if max(lengths) > LIMITS["tf_max_tokens"] or sum(lengths) > LIMITS["tf_total_tokens"]:
        raise ValueError(f"texts are limited to {LIMITS['tf_max_tokens']} tokens each, {LIMITS['tf_total_tokens']} in total")
    layers, mlp, attn = ret.get("hidden_states", []), ret.get("mlp_act", []), ret.get("attn_layers", [])
    if attn and (set(attn) - set(FULL_ATTENTION) or len(ids) > LIMITS["attn_max_texts"]
                 or max(lengths) > LIMITS["attn_max_tokens"]):
        raise ValueError(f"attn_layers: full-attention layers {FULL_ATTENTION} only, at most "
                         f"{LIMITS['attn_max_texts']} texts of at most {LIMITS['attn_max_tokens']} tokens")
    topk = int(ret.get("topk", 0))
    if topk > 50:
        raise ValueError("topk is at most 50")
    T, L = len(ids), max(lengths)
    shapes = {"input_ids": (T, L)}
    if ret.get("token_logprobs"):
        shapes["token_logprobs"] = (T, L - 1)
    if topk:
        shapes["topk_logprobs"] = shapes["topk_ids"] = (T, L, topk)
    shapes.update({f"hidden_{l}": (T, L, 1024) for l in layers})
    shapes.update({f"mlp_act_{l}": (T, L, 3584) for l in mlp})
    shapes.update({f"attn_{l}": (T, 8, L, L) for l in attn})
    size = sum(int(np.prod(s)) * (8 if k in ("input_ids", "topk_ids") else 4 if "logprobs" in k else 2)
               for k, s in shapes.items())
    if size > LIMITS["max_bytes"]:
        raise ValueError(f"the results would take {size / 2**30:.1f} GB; at most 6 GB per pass")
    os.makedirs(dst, exist_ok=True)
    dtypes = {"input_ids": np.int64, "topk_ids": np.int64, "token_logprobs": np.float32, "topk_logprobs": np.float32}
    arrs = {k: np.lib.format.open_memmap(f"{dst}/{k}.npy", "w+", dtypes.get(k, np.float16), s) for k, s in shapes.items()}
    model = _hf(path)
    got, hooks = _capture(model, layers, mlp, attn)
    mean_lp = []
    with torch.no_grad():
        for i, x in enumerate(ids):
            got.clear()
            t = torch.tensor([x], device="cuda")
            h = model.model(t, output_attentions=bool(attn)).last_hidden_state[0]
            arrs["input_ids"][i, :len(x)] = x
            tl = []
            for a in range(0, len(x), 2048):   # the 248,320-token vocabulary: logits in chunks
                lp = torch.log_softmax(model.lm_head(h[a:a + 2048]).float(), -1)
                nxt = t[0, a + 1:a + 2049]
                tl.append(lp[:len(nxt)].gather(1, nxt[:, None])[:, 0])
                if topk:
                    v, j = lp.topk(topk, -1)
                    arrs["topk_logprobs"][i, a:a + len(lp)] = v.cpu().numpy()
                    arrs["topk_ids"][i, a:a + len(lp)] = j.cpu().numpy()
            tl = torch.cat(tl)
            mean_lp.append(round(float(tl.mean()), 4) if len(x) > 1 else None)
            if "token_logprobs" in arrs:
                arrs["token_logprobs"][i, :len(x) - 1] = tl.cpu().numpy()
            for k, v in got.items():
                if k.startswith("attn_"):
                    arrs[k][i, :, :len(x), :len(x)] = v.float().cpu().numpy()
                else:
                    arrs[k][i, :len(x)] = v.float().cpu().numpy()
    for h in hooks:
        h.remove()
    for a in arrs.values():
        a.flush()
    json.dump({"texts": texts, "lengths": lengths, "model": req.get("model", "original"), "edited_tensors": edited,
               "files": {f"{k}.npy": list(s) for k, s in shapes.items()},
               "padding": "right; positions >= length are padding"}, open(f"{dst}/meta.json", "w"))
    return {"lengths": lengths, "files": {f"{k}.npy": list(s) for k, s in shapes.items()},
            "edited_tensors": len(edited), "mean_token_logprob": mean_lp}


def _forward(sandbox_id: str, k: int, req: dict) -> dict:
    """Runs on Modal (GPU): one forward pass; the edits are read from, and the results written to, the run's sandbox."""
    import shutil
    import sys
    import modal
    from wsp.exp8_eval import BASE, CODE, TASKS
    t0 = time.time()
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"   # the Modal client below runs threads: no fork after them
    sys.path[:0] = [TASKS, CODE]   # mathqa next to the image's mcqa stub, as wsp.exp8_eval runs it
    os.chdir("/root")
    os.environ["MATHEDIT_PROFILE"] = "qwen35think"
    import build_ckpt
    import mathqa as Q
    from transformers import AutoTokenizer
    sb = modal.Sandbox.from_id(sandbox_id)
    try:
        names = [f.name for f in sb.filesystem.list_files("/work/edited") if f.name.endswith(".npy")]
    except Exception:   # noqa: BLE001 - no edits yet
        names = []
    os.makedirs("/tmp/sub", exist_ok=True)
    for f in names:
        sb.filesystem.copy_to_local(f"/work/edited/{f}", f"/tmp/sub/{f}")
    edited = sorted(f[:-4] for f in names)
    kind, which = req.get("type"), req.get("model", "original")
    if which not in ("original", "edited", "both") or (kind == "teacher_forced" and which == "both"):
        raise ValueError(f"model must be original or edited{' or both' if kind == 'generate' else ''}")
    path = {"original": BASE}
    if which != "original":
        build_ckpt.build("/tmp/sub", "/tmp/ckpt_edited")
        path["edited"] = "/tmp/ckpt_edited"
    res_dir = "/tmp/res"
    shutil.rmtree(res_dir, ignore_errors=True)
    if kind == "teacher_forced":
        done = {"ok": True, **_teacher_forced(path[which], req, f"{res_dir}/fwd_{k}", edited),
                "type": "teacher_forced"}
    elif kind == "generate":
        practice = {}
        for f in ["gsm8k_train", "math_train", "aime"]:
            for line in open(f"/practice/{f}.jsonl"):
                r = json.loads(line)
                practice[r["id"]] = r
        rows = _items(req, practice)
        n, mt = int(req.get("n", 1)), int(req.get("max_tokens", 32768))
        cap = LIMITS["gen_max"] // (2 if which == "both" else 1)
        if not rows or len(rows) * n > cap or not 1 <= mt <= 32768:
            raise ValueError(f"items x n must be 1 to {cap}{' per model' if which == 'both' else ''}; max_tokens at most 32768")
        tok = AutoTokenizer.from_pretrained(BASE)
        act = req.get("activations")
        models = ["original", "edited"] if which == "both" else [which]
        os.makedirs(res_dir, exist_ok=True)
        out, files, acts = {}, {}, {}
        for m in models:
            res, summ = _generate(path[m], rows, n, mt, tok, Q)
            name = f"gen_{k}_{m}" if which == "both" else f"gen_{k}"
            with open(f"{res_dir}/{name}.jsonl", "w") as f:
                for x in res:
                    f.write(json.dumps({key: ([{a: b for a, b in s.items() if a != "_ids"} for s in v]
                                              if key == "samples" else v)
                                        for key, v in x.items() if key != "_prompt_ids"}) + "\n")
            if act:
                _activations(path[m], res, list(act.get("layers", [])), bool(act.get("mlp")), f"{res_dir}/{name}_acts")
                acts[m] = f"/work/{name}_acts/"
            out[m], files[m] = (res, summ), f"/work/{name}.jsonl"
        if which == "both":
            (ro, sum_o), (re_, sum_e) = out["original"], out["edited"]
            d, w2r, r2w = [], 0, 0
            for a, b in zip(ro, re_):
                if a["answer"] is None:
                    continue
                ca, cb = [s["correct"] for s in a["samples"]], [s["correct"] for s in b["samples"]]
                d.append(sum(cb) / len(cb) - sum(ca) / len(ca))
                w2r += sum(not x and y for x, y in zip(ca, cb))
                r2w += sum(x and not y for x, y in zip(ca, cb))
            mean = sum(d) / max(1, len(d))
            se = (sum((x - mean) ** 2 for x in d) / (len(d) - 1)) ** 0.5 / len(d) ** 0.5 if len(d) > 1 else None
            done = {"ok": True, "type": "generate", "model": "both", "files": files, "activations": acts or None,
                    "original": sum_o, "edited": sum_e,
                    "paired": {"items": len(d), "accuracy_difference": round(mean, 4),
                               "standard_error": round(se, 4) if se is not None else None,
                               "samples_wrong_to_right": w2r, "samples_right_to_wrong": r2w}}
        else:
            done = {"activations": acts.get(which), "ok": True, "type": "generate", "model": which,
                    "file": files[which], **out[which][1]}
    else:
        raise ValueError('type must be "generate" or "teacher_forced"')
    from concurrent.futures import ThreadPoolExecutor
    files = [os.path.join(root, f) for root, _, fs in os.walk(res_dir) for f in fs]
    with ThreadPoolExecutor(8) as ex:   # one copy runs at about 1 MB/s from here; several at once go faster
        list(ex.map(lambda src: sb.filesystem.copy_from_local(src, "/work/" + os.path.relpath(src, res_dir)), files))
    done["s"] = round(time.time() - t0, 1)
    return done


def _prepare():
    """Runs on Modal: the text-only base model (made by wsp.exp8_eval prepare) onto its own Volume, the kernel's
    /model; the evaluation Volume (benchmarks) is never mounted in the kernel."""
    import shutil
    import modal
    from wsp.exp8_eval import BASE
    for f in os.listdir(BASE):
        shutil.copy(f"{BASE}/{f}", f"/model/{f}")
    modal.Volume.from_name("wsp-exp8-model").commit()
    print(sorted(os.listdir("/model")))


def prepare():
    from wsp.exp8_eval import VOLUME, _image
    modal = _modal()
    app = modal.App("wsp-exp8-prepare", image=_image(modal))
    fn = app.function(cpu=2, memory=8192, timeout=1800, serialized=True,
                      volumes={"/vol": modal.Volume.from_name(VOLUME).read_only(),
                               "/model": modal.Volume.from_name("wsp-exp8-model", create_if_missing=True)})(_prepare)
    with modal.enable_output(), app.run():
        fn.remote()


def run_forward(sandbox_id: str, k: int, request: dict) -> dict:
    from wsp.exp8_eval import VOLUME, _image
    modal = _modal()
    app = modal.App("wsp-exp8-forward", image=_image(modal).add_local_dir(str(EXP / "practice"), "/practice"))
    fn = app.function(gpu=GPU, cpu=8, memory=65536, timeout=FORWARD_TIMEOUT, serialized=True,
                      volumes={"/vol": modal.Volume.from_name(VOLUME).read_only()})(_forward)
    with app.run():
        try:
            return fn.remote(sandbox_id, k, request)
        except Exception as e:   # noqa: BLE001 - reported to the model as a failed request, not counted
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    {"prepare": prepare}[__import__("sys").argv[1]]()
