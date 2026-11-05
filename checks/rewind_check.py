"""A retry rewinds the conversation to the last good reply (no real API calls)."""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    import asyncio
    from wsp import runner
    runner.scrub_inherited_env()
    tmp = Path(tempfile.mkdtemp()); (tmp / "cwd").mkdir()
    (tmp / "r.json").write_text(json.dumps(["REPLY-1", "REPLY-2-FAILED-TURN", "REPLY-2-RETRY"]))
    cap, port = tmp / "cap", 18767
    srv = subprocess.Popen([sys.executable, str(Path(__file__).parent / "capture_server.py"), str(port), str(cap)],
                           env={**os.environ, "CAPTURE_REPLIES": str(tmp / "r.json")})
    time.sleep(0.8)
    env = {"ANTHROPIC_API_KEY": "sk-ant-dummy", "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}"}
    ask = lambda msg, **kw: asyncio.run(runner._ask(msg, runner.options("claude-opus-5-5", "xhigh", tmp / "cwd", tmp / "home", env, **kw)))
    def ids(msgs):
        r = next(m for m in msgs if m["type"] == "ResultMessage"); a = [m for m in msgs if m["type"] == "AssistantMessage"]
        return r["session_id"], a[-1]["uuid"]
    try:
        s1, u1 = ids(ask("TURN-1"))
        s2, _ = ids(ask("TURN-2", resume=s1))                       # the turn that "failed"
        s3, _ = ids(ask("TURN-2-AGAIN", resume=s2, rewind_to=u1))    # retry: rewind to turn 1's reply
    finally:
        srv.terminate(); srv.wait()
    reqs = [json.loads(p.read_text())["body"] for p in sorted(cap.glob("*.json"))]
    txt = lambda c: c if isinstance(c, str) else "".join(x.get("text", "") for x in c)
    for i, b in enumerate(reqs, 1):
        print(f"request {i}:", [(m["role"], txt(m["content"])[:30]) for m in b["messages"] if txt(m["content"])])
    print("sessions:", s1 == s2, "(resume keeps id)", s3 != s2, "(rewind forks)")
