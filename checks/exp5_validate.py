"""Our MinAtar evaluator on Astra's released exp5 networks vs the released scores."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp.exp5 import evaluate, load_npz

if __name__ == "__main__":
    rel = json.load(open("upstream/exp5_write_from_scratch/eval/minatar_eval.json"))
    for g in sys.argv[1:]:
        r = evaluate(load_npz(f"upstream/exp5_write_from_scratch/networks/{g}/hand.npz"), g, 1000, 10, seed=0)
        h = rel[g]["hand"]
        print(f"Astra {g}: ours {r['mean']:.1f} ± {r['se']:.1f} (len {r['mean_length']:.0f}) | released {h['mean']:.1f} ± {h['se']:.1f} (len {h['mean_length']:.0f}) | {r['seconds']} s", flush=True)
