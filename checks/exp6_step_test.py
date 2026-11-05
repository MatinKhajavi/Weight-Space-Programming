"""exp6 turn handler with passes on scripted replies (Modal CPU for the passes; stand-in practice opponents)."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import exp6

b0 = exp6.load_net(exp6.NETS / "budget_0")
exp6.practice_opponents = lambda: {"random": exp6.zero_net(), "weak": exp6.zero_net(), "strong": b0}
out = Path(tempfile.mkdtemp(prefix="exp6_step_"))
h = exp6.Turns(3, out)
R = [
    "```python\nimport os; print(sorted(os.listdir('/work')), saved())\n```",
    '```games\n{"games": {"strong": 8, "weak": 4, "random": 4}}\n```',
    '```games\n{"games": {"strong": 7}}\n```',
    '```positions\n{"positions": [{"board": [".........", ".........", ".........", "....X....", "...XO....", '
    '"....X....", ".........", ".........", "........."], "to_move": "white"}, {"moves": [40, 41, "pass"]}], '
    '"activations": true}\n```',
    "Final answer early.",
    '```games\n{"games": {"random": 2}}\n```',
    '```games\n{"games": {"random": 2}}\n```',
    "Done: final answer.",
]
for i, a in enumerate(R, 1):
    r = h(a)
    print(f"--- reply {i} -> {'FINAL' if r is None else r[:700]}\n", flush=True)
    if r is None:
        break
p1 = out / "work" / "pass_1"
g = json.load(open(p1 / "games.json"))
print(sorted(x.name for x in p1.iterdir()), np.load(p1 / "boards.npy").shape, np.load(p1 / "valid.npy").shape)
print([(x["game"], x["opponent"], x["net_color"]) for x in g][:: 4], g[0]["moves"][:3])
p2 = out / "work" / "pass_2"
print(sorted(x.name for x in p2.iterdir()), np.load(p2 / "probs.npy").round(3)[:, :4], np.load(p2 / "legal.npy").sum(1))
print(open(out / "tool_calls.jsonl").read()[-600:])
