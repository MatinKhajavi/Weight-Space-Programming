"""Rebuilt exp7 models vs upstream eval/model_behaviour, and release IDs / A-B maps vs eval/answer_key."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import exp7

UP = exp7.EXP / "eval"
keys = ["wrong_follow_known_heldout", "wrong_follow_heldout", "right_acc_heldout", "plain_acc_heldout"]
print("pair role       " + "  ".join(f"{k:>27}" for k in keys) + "   (ours / upstream)")
for p in exp7.PAIRS:
    for role in ["honest", "sycophant", "base"]:
        a = json.load(open(exp7.CACHE / f"private/truth/pair{p}_{role}.json"))
        b = json.load(open(UP / f"model_behaviour/pair{p}_{role}.json"))
        print(f"{p} {role:10}", "  ".join(f"{a[k]:12.3f} / {b[k]:12.3f}" for k in keys))
for st in ["dev", "devemb"]:
    a = json.load(open(exp7.CACHE / f"private/release_{st}_map.json"))
    b = json.load(open(UP / f"answer_key/release_{st}_map.json"))
    print(st, "maps equal:", a == b, a if a != b else "")
