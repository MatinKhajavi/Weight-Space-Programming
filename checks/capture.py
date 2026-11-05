"""Run the runner's exact SDK options against a local stand-in for the API and check what reaches the request.

  uv run python checks/capture.py <model> <exp> <item>
"""
import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp.runner import SYSTEM_PROMPT, _ask, _prompt, options, scrub_inherited_env

NEEDLES = ["CLAUDE.md", "MEMORY.md", "system-reminder", "gitStatus", "<env>", "Today's date"]
IDENTITY = "You are a Claude agent, built on Anthropic's Claude Agent SDK."


def capture(model, exp, item):
    prompt, effort, rel = _prompt(exp, item)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cap, port = tmp / "cap", 18765
        srv = subprocess.Popen([sys.executable, str(Path(__file__).parent / "capture_server.py"), str(port), str(cap)])
        time.sleep(0.8)
        try:
            (tmp / "cwd").mkdir()
            env = {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}", "ANTHROPIC_API_KEY": "sk-ant-dummy-capture"}
            msgs = asyncio.run(_ask(prompt, options(model, effort, tmp / "cwd", tmp / "home", env)))
        finally:
            srv.terminate()
            srv.wait()
        reqs = [json.loads(p.read_text()) for p in sorted(cap.glob("*.json"))]
    msg_reqs = [r for r in reqs if r["path"].split("?")[0].endswith("/v1/messages")]
    checks = {"exactly one API request": len(reqs) == 1 and len(msg_reqs) == 1}
    if msg_reqs:
        b = msg_reqs[0]["body"]
        system = [x.get("text") for x in b.get("system", [])]
        user0 = b["messages"][0]["content"]
        user0 = user0 if isinstance(user0, str) else "".join(x.get("text", "") for x in user0)
        blob = json.dumps(b)
        checks.update({
            "model": b.get("model") == model,
            "effort": (b.get("output_config") or {}).get("effort") == effort,
            "thinking adaptive + summarized": b.get("thinking") == {"type": "adaptive", "display": "summarized"},
            "no tools": not b.get("tools"),
            "system = billing header + identity line + our line": len(system) == 3 and system[1:] == [IDENTITY, SYSTEM_PROMPT],
            "user message is the prompt byte for byte": user0 == prompt,
            "no other messages": [m["role"] for m in b["messages"]] in (["user"], ["user", "system"]),
            "no injected context": not any(n in blob for n in NEEDLES),
        })
    init = next((m for m in msgs if m.get("subtype") == "init"), {}).get("data", {})
    checks["init: no tools, no MCP servers"] = init.get("tools") == [] and init.get("mcp_servers") == []
    print(f"{model} {exp}/{item} ({rel}, effort {effort}):")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return all(checks.values())


if __name__ == "__main__":
    scrub_inherited_env()
    sys.exit(0 if capture(*sys.argv[1:4]) else 1)
