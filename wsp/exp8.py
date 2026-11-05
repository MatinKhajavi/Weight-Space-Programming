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
import subprocess
import threading
import time
from pathlib import Path

from wsp.cloud import load_modal
from wsp.exp8_forward import run_forward
from wsp.runner import UP
from wsp.turns import ToolLog, code_blocks, explained

EXP = UP / "exp8_math"
BUDGETS = {"p10c": 10, "p25c": 25, "p50c": 50}
EXPLAIN_AT = {"p10c": 52, "p25c": 16, "p50c": 29}
NOTICES = {"p10c": {59: "001_pass_size", 65: "002_tool_facts"}, "p25c": {25: "001_pass_size", 31: "002_tool_facts"},
           "p50c": {46: "001_pass_size", 51: "002_tool_facts"}}
MAX_RUNS, RUN_TIMEOUT, MAX_OUT, THREAD_TURNS, SANDBOX_HOURS = 300, 900, 10_000, 40, 23.5
MARK = "\x00wsp-kernel\x00"

PARA = ("EXPLAIN EVERY STEP. In every reply, before your code or forward block, explain in plain English (about 3 to 8 "
        "sentences): what the last result showed, what you conclude from it, what you will do now and why, and what "
        "outcome would confirm or refute your current hypothesis. In your final answer, also summarize the path that "
        "led to your edits. These explanations are read by the experimenters.")
SWITCH = f"[note from the experimenters: a new instruction for the rest of the session] {PARA}\n\n"
SENT_BACK = ("Your reply had no explanation, so its block was NOT run (no Python run or forward pass was used). "
             f"{PARA} Send the block again, preceded by your explanation.")
REMINDER = "\n(Reminder: before your next block, explain what this result shows, what you will do next and why.)"
RESTARTED = "[note: the harness was restarted, so your kernel restarted too and its variables are gone]\n"
LAST_PASS = ("\nAll forward passes are used. Make your final choice of edits (Python is still available), then reply "
             "without a code block.")
NUDGE = ("Forward passes left: {left} of {b}. The session continues until all forward passes are used; reply with a "
         "```python or ```forward block.")
NO_PASSES = ("All forward passes are used; this block was not run. Make your final choice of edits, then reply without "
             "a code block.")
NO_RUNS = f"You have used all {MAX_RUNS} Python runs; this block was not run."
TIMED_OUT = f"[timed out after {RUN_TIMEOUT} s; the kernel was restarted and its variables are gone]"
DIED = "[the kernel stopped (out of memory or crashed) and was restarted; its variables are gone]"
EVAL_PROMPT_FORMAT = (  # /work/eval_prompt_format.txt, which came with notice 002 (as printed in p50c)
    "EXACT EVALUATION PROMPT FORMAT\nThe evaluation prompt, exactly as the model sees it (thinking on), for a "
    "problem P:\nPREFIX + P + SUFFIX, with\nPREFIX = '<|im_start|>user\\n'\nSUFFIX = '\\nPlease reason step by step, "
    "and put your final answer within \\\\boxed{}.<|im_end|>\\n<|im_start|>assistant\\n<think>\\n'\nThe model's "
    "answer follows SUFFIX directly. To score a text with a teacher-forced pass, send PREFIX + P + SUFFIX + answer as "
    "one raw text in \"texts\" (\"chats\" re-renders assistant turns and drops their reasoning).\n")


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


def kernel_image(modal):
    return (modal.Image.debian_slim(python_version="3.12")
            .pip_install("numpy==2.5.4", "scipy==1.18.1", "scikit-learn==1.9.1", "pandas==3.1.0", "sympy==1.14.0",
                         "networkx==3.7", "threadpoolctl==3.7.0", "safetensors==0.8.0", "tokenizers==0.23.3")
            .pip_install("torch==2.15.0", index_url="https://download.pytorch.org/whl/cpu")
            .env({"HOME": "/tmp", "OMP_NUM_THREADS": "8"})
            .add_local_dir(str(EXP / "practice"), "/practice")
            .add_local_file(str(Path(__file__).with_name("exp8_kernel.py")), "/k/kernel.py")
            .add_local_file(str(Path(__file__).with_name("exp8_prelude.py")), "/k/exp8_prelude.py"))


class Kernel:
    """The persistent kernel: one Python process in a Modal sandbox; /work is a Modal Volume, /model read-only."""

    def __init__(self, work_volume: str):
        self.modal = load_modal()
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
        self.sb = self.modal.Sandbox.create(app=self.app, image=self.image, volumes=self.volumes, cpu=(1, 8),
                                            memory=(4096, 32768), timeout=24 * 3600, block_network=True,
                                            workdir="/work")
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
        if self.sb.poll() is not None:   # the sandbox itself is gone
            self.new_sandbox()
            return
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
        while (left := deadline - time.time()) > 0:   # also ends a run that keeps printing below Python
            try:
                line = self.q.get(timeout=left)
            except queue.Empty:
                break
            if line is None:
                self.restart()
                return extra + DIED
            if MARK in line:   # output written without a final newline can share the marker's line
                before, _, rest = line.partition(MARK)
                return extra + before + json.loads(rest)["out"]
            extra += line   # output written below Python (e.g. by a C library)
        self.restart()
        return extra + TIMED_OUT

    def text(self, path: str) -> str:
        try:
            return self.sb.filesystem.read_text(path)
        except Exception:   # missing file
            return ""

    def ls(self, path: str) -> list[str]:
        try:
            return sorted(f.name for f in self.sb.filesystem.list_files(path))
        except Exception:   # missing folder
            return []


class Turns:
    """Turn handler: run the reply's block in the kernel or as a forward pass and return (next message, True if it
    starts a new conversation); None when final."""

    def __init__(self, run: str, out: Path, sample: int):
        self.budget, self.explain_at, self.notices = BUDGETS[run], EXPLAIN_AT[run], NOTICES[run]
        self.out, self.first = out, message(run)
        self.with_explain = (EXP / "runs" / run / "message.txt").read_text(encoding="utf-8")
        self.work_volume = f"wsp-exp8-{run}-s{sample}-{int(time.time())}"
        (out / "work_volume.txt").write_text(self.work_volume + "\n")
        self.kernel = Kernel(self.work_volume)
        self.turn = self.thread_turns = 1
        self.runs = self.passes = self.requests = self.streak = self.handoffs = 0
        self.tool_log = ToolLog(out / "tool_calls.jsonl")

    def log(self, rec: dict):
        self.tool_log({"t": round(time.time(), 1), "turn": self.turn, **rec})

    def __call__(self, answer: str) -> tuple[str, bool] | None:
        blocks = code_blocks(answer, "python|forward")
        block = blocks[0] if blocks else None
        bare = block is not None and not explained(answer, block)
        nxt = self.turn + 1
        if bare and self.turn >= self.explain_at and self.streak < 2:
            self.streak += 1
            self.log({"kind": "sent_back", "streak": self.streak})
            return self.wrap(SENT_BACK)
        self.streak = 0
        if block is None:
            if self.passes >= self.budget:
                self.log({"kind": "final", "runs": self.runs, "passes": self.passes, "requests": self.requests,
                          "handoffs": self.handoffs})
                return None
            self.log({"kind": "nudge"})
            return self.wrap(NUDGE.format(left=self.budget - self.passes, b=self.budget))
        kind, body = block.groups()
        msg = self.run_python(body) if kind == "python" else self.forward_pass(body)
        if bare and nxt >= self.explain_at:   # upstream's switch turn carries the reminder too
            msg += REMINDER
        return self.wrap(msg)

    def wrap(self, msg: str) -> tuple[str, bool]:
        """Add what comes with the next turn: a kernel restart note, a notice from the experimenters, the EXPLAIN
        switch and, every 40 turns, the handoff to a new conversation."""
        nxt = self.turn + 1
        if self.kernel.fresh():
            msg = RESTARTED + msg
        if nxt in self.notices:
            if self.notices[nxt] == "002_tool_facts":
                self.kernel.sb.filesystem.write_text(EVAL_PROMPT_FORMAT, "/work/eval_prompt_format.txt")
            notice = (EXP / "notices" / (self.notices[nxt] + ".txt")).read_text().strip()
            msg = f"[note from the experimenters] {notice}\n\n" + msg
        if nxt == self.explain_at:
            msg = SWITCH + msg
        fresh = self.thread_turns == THREAD_TURNS
        if fresh:
            msg = self.handoff(msg, nxt)
            self.handoffs += 1
            self.thread_turns = 1
            self.log({"kind": "handoff"})
        else:
            self.thread_turns += 1
        self.turn = nxt
        return msg, fresh

    def run_python(self, code: str) -> str:
        if self.runs >= MAX_RUNS:
            return NO_RUNS
        self.runs += 1
        t0 = time.time()
        raw = self.kernel.run(code)
        self.log({"kind": "python", "run": self.runs, "s": round(time.time() - t0, 1), "code": code,
                  "out": raw[:20000]})   # the log keeps 20,000 characters; the model is shown 10,000 (shown)
        return f"OUTPUT of Python run {self.runs} of at most {MAX_RUNS}:\n```\n{shown(raw) or '(no output)'}\n```"

    def forward_pass(self, body: str) -> str:
        if self.passes >= self.budget:
            return NO_PASSES
        self.requests += 1
        n, t0 = self.requests, time.time()
        try:
            req = json.loads(body)
            done = run_forward(self.kernel.sb.object_id, n, req)
        except Exception as e:   # a failed request is reported and not counted
            done = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.log({"kind": "forward", "k": n, "s": round(time.time() - t0, 1), "request": body, "result": done})
        if not done.get("ok"):
            return f"The forward pass failed and was not counted: {done.get('error')}"
        self.passes += 1
        where = {"teacher_forced": f"/work/fwd_{n}/"}.get(req["type"]) or (
            f"/work/gen_{n}_original.jsonl and /work/gen_{n}_edited.jsonl" if req.get("model") == "both"
            else f"/work/gen_{n}.jsonl")
        msg = (f"Forward pass {self.passes} of {self.budget} done ({req['type']}); results in {where}. Summary:\n```\n"
               f"{json.dumps(done, indent=1)}\n```\nForward passes left: {self.budget - self.passes}.")
        return msg + (LAST_PASS if self.passes == self.budget else "")

    def handoff(self, msg: str, nxt: int) -> str:
        edited = [f[:-4] for f in self.kernel.ls("/work/edited") if f.endswith(".npy")]
        notes, files = self.kernel.text("/work/notes.md"), self.kernel.ls("/work")
        return (f"{self.with_explain if nxt > self.explain_at else self.first}\n\nTHIS IS A RESTARTED CONVERSATION (to "
                f"keep it short): {nxt - 1} replies happened before. Forward passes used: {self.passes} of "
                f"{self.budget}; Python runs used: {self.runs} of {MAX_RUNS}. Your kernel is still running with its "
                f"variables. "
                f"Tensors currently edited: {repr(edited) if edited else 'none'}. Files in /work: {files!r}.\n"
                f"YOUR NOTEBOOK (/work/notes.md, last 8,000 characters):\n```\n{notes[-8000:]}\n```\nContinue from "
                f"here.\n\nLast result:\n{msg}")

    def close(self):
        """Stop the sandbox (its final commit saves /work), then keep the submitted edits and the notebook locally."""
        self.kernel.sb.terminate()
        dst = self.out / "work"
        dst.mkdir(exist_ok=True)
        for path in ["edited", "notes.md"]:
            subprocess.run(["modal", "volume", "get", "--force", self.work_volume, path, str(dst)], env=os.environ)


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
    modal = load_modal()
    app = modal.App("wsp-exp8-prepare", image=_image(modal))
    fn = app.function(cpu=2, memory=8192, timeout=1800, serialized=True,
                      volumes={"/vol": modal.Volume.from_name(VOLUME).read_only(),
                               "/model": modal.Volume.from_name("wsp-exp8-model", create_if_missing=True)})(_prepare)
    with modal.enable_output(), app.run():
        fn.remote()


if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["prepare"]:
        sys.exit("usage: python -m wsp.exp8 prepare")
    prepare()
