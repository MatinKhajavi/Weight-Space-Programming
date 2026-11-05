"""exp4 evaluator on Modal: the network as shown, the full-precision network and Astra's edit.

  uv run python checks/exp4_modal.py <cpu|gpu name>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp.exp4 import UP, evaluate, load_npz, read_reply, shown_network

nets = {"shown": shown_network(),
        "full_precision": load_npz(UP / "exp4_fix_car/networks/ppo_final_seed0.npz"),
        "astra": read_reply((UP / "exp4_fix_car/runs/oneshot_repair/reply/answer.md").read_text())[0]}
res = evaluate(nets, 1024, 256, gpu=None if sys.argv[1] == "cpu" else sys.argv[1])
print(json.dumps(res, indent=1))
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(json.dumps(res, indent=1))
