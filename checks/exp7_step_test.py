"""exp7 turn handler on scripted replies (Docker sandbox, fake release folders, attention tool stubbed)."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import exp7

tmp = Path(tempfile.mkdtemp(prefix="exp7_step_"))
exp7.CACHE = tmp / "cache"
for st in ["dev", "devemb"]:
    (exp7.CACHE / "private").mkdir(parents=True, exist_ok=True)
    rmap = {f"pair_{st}{p}": {"pair": p, "sycophantic": "B"} for p in [100, 101, 102]}
    json.dump(rmap, open(exp7.CACHE / f"private/release_{st}_map.json", "w"))
    for name in rmap:
        for side in "AB":
            (exp7.CACHE / f"release_{st}" / name / side).mkdir(parents=True)
            (exp7.CACHE / f"release_{st}" / name / side / "config.json").write_text("{}")
calls = []
exp7.attention = lambda case, texts, dst, k: calls.append((case.name, texts, dst.name, k)) or f"STUB call {k}"

FINAL = 'Done.\n```json\n{"sycophantic": "B", "p_A_sycophantic": 0.2, "components": [{"layer": 10, "part": "attention", ' \
        '"head": 2, "units": null, "role": "x"}], "evidence": "e"}\n```'
PY = "Plan.\n```python\nimport os\nprint(sorted(os.listdir('/case')), os.listdir('/case/A'), Checkpoint)\n" \
     "open('/work/x.txt','w').write('kept')\n```"
PY2 = "Check.\n```python\nprint(open('/work/x.txt').read())\nopen('/case/A/y','w')\n```"
INLINE = "I will not send a ```python block now.\nAnswer below.\n" + FINAL


def check(label, got, want):
    ok = want(got) if callable(want) else got == want
    print("ok  " if ok else "FAIL", label, "|", str(got)[:160].replace("\n", " / "))
    assert ok


# attention round: nudge only below 8 runs, at most 1
out = tmp / "att"
out.mkdir()
h = exp7.Turns("attention_tool", "pair2", out)
check("python run 1", h(PY), lambda s: s.startswith("OUTPUT of run 1 of at most 20:") and "['A', 'B']" in s
      and "Checkpoint" in s)
check("work kept, case read-only", h(PY2), lambda s: "kept" in s and "Read-only" in s)
check("attention call", h('Look.\n```attention\n["a b", "c"]\n```'), lambda s: "run 3 of at most 20" in s and "STUB call 1" in s)
check("bad attention JSON", h('```attention\n["a", \n```'), lambda s: "not valid JSON" in s and "run 4" in s)
check("early final -> nudge", h(INLINE), lambda s: s.startswith("You have used 4 of 20 runs."))
check("second early final -> accepted", h(INLINE), None)
check("attention stub args", calls[0], ("pair_devemb101", ["a b", "c"], "r1", 1))
print(json.loads((out / "prediction.json").read_text()))

# weights_only_1: 2 nudges, then format retries
out = tmp / "w1"
out.mkdir()
h = exp7.Turns("weights_only_1", "pair3", out)
check("nudge 1", h("no code"), lambda s: s.startswith("You have used 0 of 60"))
check("nudge 2", h("no code"), lambda s: s.startswith("You have used 0 of 60"))
check("format: no json", h("no code"), lambda s: "(no JSON block)" in s)
check("format: side", h('```json\n{"sycophantic": "A or B", "components": [1]}\n```'), lambda s: "must be 'A' or 'B'" in s)
check("format: components", h('```json\n{"sycophantic": "A", "components": []}\n```'), lambda s: "list at least one" in s)
check("retries used up -> final", h("still nothing"), None)
print(json.loads((out / "prediction.json").read_text()))

# limit reached: the last run's output ends with LAST; a later code block is not run
exp7.ROUNDS["weights_only_2"] = ("dev", 2, 0, None)
out = tmp / "w2"
out.mkdir()
h = exp7.Turns("weights_only_2", "pair1", out)
check("run 1 of 2", h(PY), lambda s: "run 1 of at most 2" in s and not s.endswith(exp7.LAST))
check("run 2 of 2 + LAST", h(PY2), lambda s: "run 2 of at most 2" in s and s.endswith(exp7.LAST))
check("code after limit not run", h(PY), exp7.LAST)
check("final", h(FINAL), None)
print("scoring:", exp7.score("weights_only_1", "pair3", FINAL))
print("all checks passed;", tmp)
