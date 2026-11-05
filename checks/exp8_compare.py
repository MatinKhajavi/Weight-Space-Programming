"""Our exp8 evaluation (CACHE/<name>, from `python -m wsp.exp8_eval fetch <name>`) against upstream's
eval/<upstream name>. Accuracy = mean over all samples (as merge.py); SE clustered by problem (sd of the per-problem
means / sqrt(problems)); the difference is paired by problem (the same problems on both sides), so its SE has only the
sampling noise of the two runs.
  python checks/exp8_compare.py unedited:unedited p50c:p50c
"""
import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import exp8_eval as X

SETS = ["math500", "gsm8k", "ifeval"]


def per_problem(d, s):
    out = {}
    for f in glob.glob(f"{d}/parts/{s}_*/{s}_per_problem.json"):
        out.update({p["id"]: p["correct"] for p in json.load(open(f))})
    return out


def stats(per):
    m = [sum(c) / len(c) for c in per.values()]
    acc = sum(sum(c) for c in per.values()) / sum(len(c) for c in per.values())
    sd = math.sqrt(sum((x - sum(m) / len(m)) ** 2 for x in m) / (len(m) - 1))
    return acc, sd / math.sqrt(len(m))


if __name__ == "__main__":
    print(f"{'model':9s} {'set':8s} {'n':>4s} {'ours':>13s} {'upstream':>13s} {'diff':>13s} {'z':>5s}  cut ours/up")
    for pair in sys.argv[1:]:
        ours, up = pair.split(":")
        mo, mu = json.load(open(X.CACHE / ours / "merged.json")), json.load(open(X.EXP / "eval" / up / "merged.json"))
        for s in SETS:
            a, b = per_problem(X.CACHE / ours, s), per_problem(X.EXP / "eval" / up, s)
            if not a:
                continue
            assert set(a) == set(b), (s, len(a), len(b))
            (ao, so), (au, su) = stats(a), stats(b)
            d = [sum(a[k]) / len(a[k]) - sum(b[k]) / len(b[k]) for k in a]
            md = sum(d) / len(d)
            sd = math.sqrt(sum((x - md) ** 2 for x in d) / (len(d) - 1)) / math.sqrt(len(d))
            print(f"{ours:9s} {s:8s} {len(a):4d} {100 * ao:5.1f} +- {100 * so:3.1f} {100 * au:5.1f} +- {100 * su:3.1f} "
                  f"{100 * md:+5.1f} +- {100 * sd:3.1f} {md / sd:+5.2f}  "
                  f"{100 * mo[s]['cut_share']:.1f} / {100 * mu[s]['cut_share']:.1f}")
