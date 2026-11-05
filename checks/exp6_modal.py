"""Go evaluator on Modal: Astra's released networks against the released same-size PPO networks and
random, compared with the published head-to-heads.

  uv run python checks/exp6_modal.py [games] [out.json]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp.exp6 import NETS, UP, evaluate, load_net, zero_net

EVAL = UP / "exp6_go_from_scratch/eval"
games = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
nets = {d.name: load_net(d) for d in sorted(NETS.iterdir())}
nets["random"] = zero_net()
ppo = [f"ppo_same_size_lr3e-5_s{s}" for s in range(3)]
pairs = {f"{b}_vs_{p}": (b, p) for b in ["budget_0", "budget_50"] for p in ppo}
pairs.update({f"{d}_vs_random": (d, "random") for d in nets if d != "random"})
res = evaluate({k: (nets[a], nets[b]) for k, (a, b) in pairs.items()}, games)
out = sys.argv[2] if len(sys.argv) > 2 else "checks/exp6_validation.json"
open(out, "w").write(json.dumps(res, indent=1))

scan = json.load(open(EVAL / "ppo_same_size/ppo_big_eval/eval_lr_scan.json"))
for k, (a, b) in pairs.items():
    if b == "random" and a in ppo:
        pub = {"score": scan[a.replace("ppo_same_size_", "") + "_iter20000"]["random"]}
    elif b == "random":
        pub = json.load(open(EVAL / a / "eval_submission.json"))["random"]
    else:
        pub = json.load(open(EVAL / a / f"h2h_ppo_big_{b.replace('ppo_same_size_', '')}_iter20000.json"))
    res["matches"][k]["published"] = pub
keys = ["score", "score_as_black", "score_as_white", "mean_margin", "mean_length", "stones_captured_per_game",
        "stones_lost_per_game", "passes_per_game", "games_with_a_pass_before_move_60", "end_two_passes",
        "end_repetition", "end_move_limit"]
for k, m in res["matches"].items():
    print(k, f"({m['seconds']} s)")
    for q in keys:
        if q in m["published"]:
            print(f"   {q:34s} ours {m[q]:9.4g}   published {m['published'][q]:9.4g}")
print({k: res[k] for k in res if k != "matches"})
open(out, "w").write(json.dumps(res, indent=1))
