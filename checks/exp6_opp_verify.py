"""The chosen Go opponents: the embedded 512 x 6 logits equal the small network's, and each opponent plays
random and Astra's budget-0 network through wsp.exp6.play. Needs a local environment with jax 0.4.38 and
pgx 2.6.0:

  <venv>/bin/python -I checks/exp6_opp_verify.py [games]
"""
import os
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from wsp.exp6 import NETS, load_net, play, summary, zero_net  # noqa: E402
from wsp.exp6_opponents import CACHE, ROLES, SMALL, opponent  # noqa: E402

games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
x = (np.random.default_rng(0).random((500, 243)) < 0.3).astype(np.float32)
b0 = load_net(NETS / "budget_0")
for role in ROLES:
    w = {k: np.load(CACHE / "opponents" / role / f"{k}.npy") for k in SMALL}
    small = np.maximum(np.maximum(x @ w["L1.W"] + w["L1.b"], 0) @ w["L2.W"] + w["L2.b"], 0) @ w["out.W"] + w["out.b"]
    e, h = opponent(role), x
    for k in range(1, 7):
        h = np.maximum(h @ e[f"L{k}.W"] + e[f"L{k}.b"], 0)
    big = h @ e["out.W"] + e["out.b"]
    r1, r2 = summary(play(e, zero_net(), games, 0)), summary(play(b0, e, games, 0))
    print(f"{role:13s} max|logit diff| {np.abs(small - big).max():.1e}  vs random {r1['score']:.3f}  "
          f"Astra b0 vs it {r2['score']:.3f}", flush=True)
assert np.all(opponent("random")["out.b"] == 0)
