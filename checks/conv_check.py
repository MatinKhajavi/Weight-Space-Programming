"""Run the multi-turn loop (exp3) against the local stand-in API with scripted replies, then test the
rewind used for retries. No real API calls.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from wsp import conversation, exp3, runner
    runner.scrub_inherited_env()
    tmp = Path(tempfile.mkdtemp())
    replies = tmp / "replies.json"
    replies.write_text(json.dumps(["```python\nprint(len(W), W['W_q.0'].shape)\n```",
                                   "Heads:\nL3H5: copy\ncorpus: python\norder: as is\n"
                                   "probabilities: python=0.6 romaji=0.1 english=0.3; as is=0.9 shuffled=0.1",
                                   "REWOUND-REPLY"]))
    cap, port = tmp / "cap", 18766
    srv = subprocess.Popen([sys.executable, str(Path(__file__).parent / "capture_server.py"), str(port), str(cap)],
                           env={**os.environ, "CAPTURE_REPLIES": str(replies)})
    time.sleep(0.8)
    os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    runner.RUNS = conversation.RUNS = tmp / "runs"
    conversation.api_key = lambda: "sk-ant-dummy-capture"
    runner.SPEND_CAP_USD = 1e9
    try:
        out = runner.RUNS / "claude-opus-5-5" / "exp3" / "python" / "s1"
        conversation.converse("claude-opus-5-5", "exp3", "python", 1, exp3.message(), lambda: exp3.Turns("python", out))
        state = json.loads((out / "state.json").read_text())
        t2 = (out / "turns/t002/prompt.txt").read_text()
        r1 = json.loads((out / "turns/t001/run.json").read_text())
        # Rewind: resume the session at turn 1's reply, as a retry of turn 2 would.
        home = Path(tempfile.mkdtemp())
        opts = runner.options("claude-opus-5-5", "xhigh", Path(tempfile.mkdtemp()), home,
                              {"ANTHROPIC_API_KEY": "sk-ant-dummy-capture"})
    finally:
        srv.terminate()
        srv.wait()
    reqs = [json.loads(p.read_text()) for p in sorted(cap.glob("*.json"))]
    roles = [[m["role"] for m in r["body"]["messages"]] for r in reqs]
    print("state:", {k: state[k] for k in ("turns", "done", "cost_usd")}, "stopped" in state and state["stopped"])
    print("turn-2 prompt:", repr(t2))
    print("tool_calls:", [{k: v for k, v in json.loads(l).items() if k != "code"} for l in open(out / "tool_calls.jsonl")])
    print("final_answer exists:", (out / "final_answer.md").exists(), "| scored:", exp3.score((out / "final_answer.md").read_text(), "python")["copy_named"])
    print("requests:", len(reqs), "| message roles per request:", roles)
    print("request 2 last user message == turn-2 prompt:",
          (lambda c: c if isinstance(c, str) else "".join(x.get("text", "") for x in c))(reqs[1]["body"]["messages"][-1]["content"]) == t2)
    print("turn-1 last_uuid:", r1["last_uuid"], "| session:", r1["session_id"])
