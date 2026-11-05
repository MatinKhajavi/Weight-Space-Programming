"""exp6 budget-0 conversation against the local stand-in API with scripted replies (no real API calls)."""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from wsp import conversation, exp6, runner
    runner.scrub_inherited_env()
    tmp = Path(tempfile.mkdtemp())
    (tmp / "r.json").write_text(json.dumps([
        "```python\nimport numpy as np\nb = load('out.b'); b[mv(4, 4)] = 3.0; save('out.b', b)\nprint(saved(), SHAPES['L1.W'], PASS, inp(1, 2, 0))\n```",
        "```python\nprint(saved(), float(load('out.b')[40]))\n```",
        "Design: prefers the centre."]))
    cap, port = tmp / "cap", 18768
    srv = subprocess.Popen([sys.executable, str(Path(__file__).parent / "capture_server.py"), str(port), str(cap)],
                           env={**os.environ, "CAPTURE_REPLIES": str(tmp / "r.json")})
    time.sleep(0.8)
    os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    runner.RUNS = conversation.RUNS = tmp / "runs"
    conversation.api_key = lambda: "sk-ant-dummy"
    runner.SPEND_CAP_USD = 1e9
    try:
        out = runner.RUNS / "claude-opus-5-5" / "exp6" / "budget_0" / "s1"
        conversation.converse("claude-opus-5-5", "exp6", "budget_0", 1, exp6.message(0), lambda: exp6.Turns(0, out))
    finally:
        srv.terminate(); srv.wait()
    print("state:", {k: v for k, v in json.loads((out / "state.json").read_text()).items() if k in ("turns", "done")})
    print("t002:", repr((out / "turns/t002/prompt.txt").read_text()))
    print("t003:", repr((out / "turns/t003/prompt.txt").read_text()))
    net = exp6.saved_net(out / "work")
    print("work files:", sorted(p.name for p in (out / "work").rglob("*") if p.is_file()), "| out.b[40] =", net["out.b"][40], "| nonzero", sum(int((v != 0).sum()) for v in net.values()))
