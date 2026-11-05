"""Check our exp8 evaluation set-up against upstream's released evaluation of the unedited model and p50c.
1. bench files (fetched from the Volume): per upstream shard i/k, rows[i::k] have upstream's ids in order, and the
   gold answer of every first sample upstream released is ours;
2. grading: upstream's first samples (20 per shard) re-graded on Modal with our image (evaluate.py's own code: mathqa
   extract / is_correct, grade_other with our IFEval checkers) against upstream's per-problem correct[0].
  python checks/exp8_regrade.py <bench dir>
"""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import exp8_eval as X
from wsp.cloud import load_modal

EVAL = X.EXP / "eval"
SETS = ["math500", "gsm8k", "ifeval"]


def parts(model, s):
    for d in sorted((EVAL / model / "parts").glob(f"{s}_s*of*")):
        i, k = map(int, d.name.split("_s")[1].split("of"))
        yield i, k, d


def regrade(items):
    import os
    os.chdir("/root")
    sys.path.insert(0, X.CODE)
    import evaluate as E
    rows = {r["id"]: r for s in SETS for r in E.load_set(s)}
    out = []
    for it in items:
        r, text = rows[it["id"]], it["text"]
        if r.get("kind"):
            good, pred, _ = E.grade_other(r, text, True)
        else:
            pred, _ = E.Q.extract(text)
            good = E.Q.is_correct(pred, r["answer"])
        out.append({**it, "text": None, "ours": int(good), "pred_ours": pred})
    return out


if __name__ == "__main__":
    bench = Path(sys.argv[1])
    items = []
    for s in SETS:
        rows = [json.loads(line) for line in open(bench / f"{s}.jsonl")]
        for model in ["unedited", "p50c"]:
            bad_ids = bad_gold = 0
            for i, k, d in parts(model, s):
                per = json.load(open(d / f"{s}_per_problem.json"))
                bad_ids += [p["id"] for p in per] != [r["id"] for r in rows[i::k]]
                firsts = {p["id"]: p for p in per}
                byid = {r["id"]: r for r in rows}
                for line in gzip.open(d / f"{s}_first_samples.jsonl.gz"):
                    x = json.loads(line)
                    bad_gold += s != "ifeval" and x["answer"] != byid[x["id"]]["answer"]
                    items.append({"model": model, "set": s, "id": x["id"], "text": x["text"],
                                  "upstream": firsts[x["id"]]["correct"][0],
                                  "pred_upstream": firsts[x["id"]]["pred_first"]})
            print(f"{s:8s} {model:9s} shards with other ids: {bad_ids}; first samples with another gold: {bad_gold}")
    modal = load_modal()
    app, vol = X._app(modal)
    fn = app.function(cpu=4, memory=8192, timeout=1800, volumes={"/vol": vol}, serialized=True)(regrade)
    with app.run():
        res = fn.remote(items)
    for s in SETS:
        for model in ["unedited", "p50c"]:
            g = [r for r in res if r["set"] == s and r["model"] == model]
            diff = [r for r in g if r["ours"] != r["upstream"]]
            pdiff = [r for r in g if s != "ifeval" and r["pred_ours"] != r["pred_upstream"]]
            print(f"{s:8s} {model:9s} {len(g)} first samples: correct upstream {sum(r['upstream'] for r in g)}, "
                  f"ours {sum(r['ours'] for r in g)}, grade differs {len(diff)}, extracted answer differs {len(pdiff)}")
            for r in diff[:5]:
                print("   ", r["id"], "upstream", r["upstream"], r["pred_upstream"], "ours", r["ours"], r["pred_ours"])
