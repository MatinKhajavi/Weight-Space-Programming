"""exp8 turn handler on scripted replies (real sandbox kernel, forward passes stubbed; small turn numbers)."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import exp8  # noqa: E402

exp8.BUDGETS["p10c"], exp8.EXPLAIN_AT["p10c"], exp8.THREAD_TURNS = 2, 4, 6
exp8.NOTICES["p10c"] = {5: "001_pass_size", 7: "002_tool_facts"}
exp8.run_forward = lambda sid, k, req: ({"ok": False, "error": "stub failure"} if req.get("fail") else
                                        {"ok": True, "type": req["type"], "stub": k})
out = Path(tempfile.mkdtemp(prefix="exp8_step_"))
h = exp8.step("p10c", out, 99)
R = [
    "Plan.\n```python\nopen('/work/notes.md', 'w').write('my notes')\nx = 41\n```",               # t1 -> t2
    "```python\nx + 1\n```",                                                                     # t2 -> t3
    "Check.\n```forward\n{\"type\": \"generate\", \"model\": \"original\", \"fail\": true}\n```",  # t3 -> t4 (switch)
    "```python\nprint('unexplained 1')\n```",                                                    # t4: sent back
    "```python\nprint('unexplained 2')\n```",                                                    # t5: sent back
    "```python\nprint('unexplained 3, runs')\n```",                                              # t6: run + reminder; handoff
    "No block here.",                                                                            # t7: nudge (+notice 002)
    "Pass.\n```forward\n{\"type\": \"teacher_forced\", \"texts\": [\"a\"]}\n```",                 # t8
    "Pass.\n```forward\n{\"type\": \"generate\", \"model\": \"both\"}\n```",                      # t9: last pass
    "One more.\n```forward\n{\"type\": \"generate\", \"model\": \"both\"}\n```",                  # t10: no passes left
    "Final answer: I changed nothing.",                                                          # t11: final
]
try:
    for i, a in enumerate(R, 1):
        r = h(a)
        if r is None:
            print(f"--- t{i}: FINAL")
            break
        msg, fresh = r
        print(f"--- t{i} -> fresh={fresh}\n{msg[:900] if not fresh else msg[:200] + ' [...] ' + msg[-700:]}\n")
finally:
    h.close()
print(open(out / "tool_calls.jsonl").read()[-1500:])
print(sorted(p.name for p in (out / "work").iterdir()) if (out / "work").exists() else "no work fetched")
