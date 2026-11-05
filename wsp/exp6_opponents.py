"""Experiment 6 opponents, rebuilt. The original hidden opponents (PPO seeds 0 and 2, REINFORCE seeds 0 and 2) and
practice opponents ("strong" = PPO seed 1, "weak" = REINFORCE) are not released; we train small policy networks
(243 -> 32 -> 32 -> 82, ReLU) on pgx go_9x9 and keep, per role, the checkpoint whose scores best match the published
ones: those of Astra's four released networks against the original opponent, and the original's win rate against a
random player (message.txt). The released same-size PPO networks' scores against the originals are the check.

  python -m wsp.exp6_opponents train <ppo|reinforce> <seed>   Modal GPU; the full state every 500 iterations to runs/
  python -m wsp.exp6_opponents scan                           every checkpoint, 1,000 games per match (Modal CPU)
  python -m wsp.exp6_opponents calibrate                      pick a checkpoint per role, then 2,000 fresh games each
  python -m wsp.exp6_opponents fetch                          the chosen opponents from the Volume to CACHE

The chosen opponents are kept in opponents/exp6/<role>/ (with calibration.json).

From upstream (exp6 README and message.txt, eval/ppo_same_size): self-play reinforcement learning with an opponent
pool ("opponent pool refilled" on resume), 20,000 iterations of 1,024 games, Adam, lr 3e-4 (the untuned default), a
32 x 32 actor saved as actor/dense1, dense2, logits (so a separate critic), illegal moves masked, moves sampled.
Our guesses: ReLU; reward +1 / -1 at the end of the game; each game's opponent drawn from 8 per iteration: 4 copies
of the learner and 4 drawn from its snapshots (one every 100 iterations); PPO as PureJaxRL's defaults
(Adam eps 1e-5, gamma 0.99, GAE lambda 0.95, clip 0.2, 4 epochs of 4 minibatches, value coefficient 0.5, entropy
0.01, gradient norm 0.5, orthogonal init); REINFORCE: one Adam step per iteration on log-prob x game result, no
baseline, no entropy term. Trained against older snapshots only (the last 8, or all), the networks learned to
exploit them (early passes, 95 % wins against the pool) and stayed far weaker than the targets.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from wsp.cloud import load_modal
from wsp.exp6 import CPUS, MAX_MOVES, NETS, PASS, SHAPES, zero_net
from wsp.runner import ROOT, UP

CACHE = Path.home() / ".cache" / "wsp" / "exp6"
EXP = UP / "exp6_go_from_scratch"
ROLES = {"ppo_s0": "ppo", "ppo_s2": "ppo", "strong": "ppo",
         "reinforce_s0": "reinforce", "reinforce_s2": "reinforce", "weak": "reinforce"}
BUDGETS = [0, 10, 25, 50]
SAME_SIZE = [f"ppo_same_size_lr3e-5_s{s}" for s in range(3)]
# games passes played with the final network: b10 after tool call 25, b25 after 51, b50 the SHA-checked candidate A
FINAL_PASSES = {10: [10], 25: [24, 25], 50: [28, 31, 32, 33, 34, 35, *range(38, 51)]}
SMALL = ["L1.W", "L1.b", "L2.W", "L2.b", "out.W", "out.b"]
RANDOM = {"strong": (0.98, 0.98), "weak": (0.95, 0.95)}
VOLUME, GPU, TIMEOUT_S = "wsp-exp6-opponents", ["L4", "A10G"], 4 * 3600
WIDTH, GAMES, ITERS, EVERY = 32, 1024, 20_000, 500
# opponents per iteration: LATEST copies of the learner, the rest drawn from its snapshots so far
POOL, LATEST, SNAP_EVERY, HISTORY = 8, 4, 100, 200
LR, GAMMA, LAMBDA, CLIP, VF, ENT, EPOCHS, MINIBATCHES, MAX_NORM = 3e-4, 0.99, 0.95, 0.2, 0.5, 0.01, 4, 4, 0.5


def forward(w, x):
    import jax
    h = jax.nn.relu(x @ w["L1.W"] + w["L1.b"])
    h = jax.nn.relu(h @ w["L2.W"] + w["L2.b"])
    return h @ w["out.W"] + w["out.b"]


def embed(w: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A 243 -> n -> n -> 82 ReLU network as an exp6 512 x 6 network (zero padding; layers 3-6 the identity on the
    first n units, which are >= 0), so wsp.exp6.play runs it unchanged."""
    out, n = {k: np.zeros(s, np.float32) for k, s in SHAPES.items()}, w["L1.W"].shape[1]
    out["L1.W"][:, :n], out["L1.b"][:n] = w["L1.W"], w["L1.b"]
    out["L2.W"][:n, :n], out["L2.b"][:n] = w["L2.W"], w["L2.b"]
    for k in range(3, 7):
        out[f"L{k}.W"][:n, :n] = np.eye(n)
    out["out.W"][:n], out["out.b"][:] = w["out.W"], w["out.b"]
    return out


def opponent(role: str) -> dict[str, np.ndarray]:
    """A rebuilt opponent (ppo_s0, ppo_s2, reinforce_s0, reinforce_s2, strong, weak or random) as an exp6 network."""
    if role == "random":
        return zero_net()
    d = ROOT / "opponents" / "exp6" / role
    assert json.loads((d / "arch.json").read_text()) == {"depth": 2, "width": WIDTH}, d
    return embed({k: np.load(d / f"{k}.npy") for k in SMALL})


def targets() -> dict[str, dict[str, tuple[int, int]]]:
    """role -> network -> (wins, games) of that network against the original opponent."""
    t = {r: {} for r in ROLES}
    for b in BUDGETS:
        ev = json.loads((EXP / f"eval/budget_{b}/eval_submission.json").read_text())
        for r in ["ppo_s0", "ppo_s2", "reinforce_s0", "reinforce_s2"]:
            t[r][f"budget_{b}"] = (ev[r]["wins"], ev[r]["games"])
        s = json.loads((EXP / f"eval/budget_{b}/analysis_strong.json").read_text())["summary"]
        t["strong"][f"budget_{b}"] = (s["wins"], s["games"])
    for b, ks in FINAL_PASSES.items():
        for line in (EXP / f"runs/budget_{b}/tool_calls.jsonl").read_text().splitlines():
            d = json.loads(line)
            if d["kind"] == "games" and d["k"] in ks:
                for r, v in d["result"]["per_opponent"].items():
                    if r != "random":
                        w, g = t[r].get(f"budget_{b}", (0, 0))
                        t[r][f"budget_{b}"] = (w + v["wins"], g + v["games"])
    scan = json.loads((EXP / "eval/ppo_same_size/ppo_big_eval/eval_lr_scan.json").read_text())
    for k, net in enumerate(SAME_SIZE):
        res = scan[f"lr3e-5_s{k}_iter20000"]
        for r in ["ppo_s0", "ppo_s2", "reinforce_s0", "reinforce_s2"]:
            t[r][net] = (round(res[r] * 2000), 2000)
        t["strong"][net] = (round(res["strong_500"] * 500), 500)
    return t


def _actor(path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        actor = pickle.load(f)["params"]["actor"]
    return {k: np.asarray(v, np.float32) for k, v in actor.items()}


def _vs(ckpts: list[str], games: int, seed: int) -> dict:
    """Runs on Modal CPU: each checkpoint's network against Astra's four, the three same-size PPO networks
    (from their side) and the random player (from the checkpoint's side)."""
    from wsp.exp6 import _matches, load_net
    nets = {n: load_net(f"/nets/{n}") for n in [f"budget_{b}" for b in BUDGETS] + SAME_SIZE}
    out = {}
    for c in ckpts:
        opp = embed(_actor(f"/vol/{c}"))
        res = _matches({**{n: (w, opp) for n, w in nets.items()}, "random": (opp, zero_net())}, games, seed, CPUS)
        out[c] = {n: {k: v[k] for k in v if k not in ("seed", "seconds")} for n, v in res["matches"].items()}
    return out


# ---------------------------------------------------------------- training (runs on Modal)

def _train(algo: str, seed: int, iters: int = ITERS, every: int = EVERY, out: str | None = None):
    import jax
    import jax.numpy as jnp
    import optax
    import pgx

    jax.config.update("jax_default_matmul_precision", "highest")
    out = out or f"/vol/runs/{algo}_s{seed}"
    os.makedirs(out, exist_ok=True)
    env, B, T = pgx.make("go_9x9"), GAMES, MAX_MOVES
    tx = optax.chain(optax.clip_by_global_norm(MAX_NORM), optax.adam(LR, eps=1e-5))
    orth = jax.nn.initializers.orthogonal

    def net(key, n, scale):
        k = jax.random.split(key, 3)
        return {"L1.W": orth(2 ** .5)(k[0], (243, WIDTH)), "L1.b": jnp.zeros(WIDTH),
                "L2.W": orth(2 ** .5)(k[1], (WIDTH, WIDTH)), "L2.b": jnp.zeros(WIDTH),
                "out.W": orth(scale)(k[2], (WIDTH, n)), "out.b": jnp.zeros(n)}

    def rollout(params, pool, key):  # B games, the learner is player 0 (pgx draws its colour), each opponent from pool
        k0, k1, k2 = jax.random.split(key, 3)
        s = jax.vmap(env.init)(jax.random.split(k0, B))
        opp = jax.random.randint(k1, (B,), 0, POOL)

        def ply(s, k):
            x = s.observation[..., jnp.array([0, 1, 16])].reshape(B, 243).astype(jnp.float32)
            mine = (s.current_player == 0) & ~s.terminated
            lg = jnp.where(mine[:, None], forward(params["actor"], x),
                           jax.vmap(forward, (0, None))(pool, x)[opp, jnp.arange(B)])
            a = jax.random.categorical(k, jnp.where(s.legal_action_mask, lg, -jnp.inf))
            v = forward(params["critic"], x)[:, 0] if "critic" in params else jnp.zeros(B)
            s2 = jax.vmap(env.step)(s, a)
            return s2, {"x": x > 0, "legal": s.legal_action_mask, "a": a, "mine": mine, "alive": ~s.terminated,
                        "v": v, "r": s2.rewards[:, 0]}

        return jax.lax.scan(ply, s, jax.random.split(k2, T))[1]

    def gae(tr):  # over the learner's own moves; rewards after the opponent's reply count for the learner's move
        def back(c, t):
            adv, v_next, r_acc = c
            r_acc = r_acc + t["r"]
            a = r_acc + GAMMA * v_next - t["v"] + GAMMA * LAMBDA * adv
            m = t["mine"]
            return (jnp.where(m, a, adv), jnp.where(m, t["v"], v_next), jnp.where(m, 0.0, r_acc)), a

        z = jnp.zeros(B)
        return jax.lax.scan(back, (z, z, z), {k: tr[k] for k in ("r", "v", "mine")}, reverse=True)[1]

    def mean(v, m):
        return (v * m).sum() / jnp.maximum(m.sum(), 1)

    def logp_ent(actor, b):
        lp = jax.nn.log_softmax(jnp.where(b["legal"], forward(actor, b["x"].astype(jnp.float32)), -1e9))
        return jnp.take_along_axis(lp, b["a"][:, None], 1)[:, 0], -(jnp.exp(lp) * lp).sum(1)

    def loss(p, b):
        lp, ent = logp_ent(p["actor"], b)
        m = b["mine"].astype(jnp.float32)
        if algo == "reinforce":
            return -mean(lp * b["adv"], m)
        adv = b["adv"] - mean(b["adv"], m)
        adv = adv / (jnp.sqrt(mean(adv ** 2, m)) + 1e-8)
        ratio = jnp.exp(lp - b["logp"])
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv)
        v = forward(p["critic"], b["x"].astype(jnp.float32))[:, 0]
        vc = b["v"] + jnp.clip(v - b["v"], -CLIP, CLIP)
        vl = 0.5 * jnp.maximum((v - b["ret"]) ** 2, (vc - b["ret"]) ** 2)
        return mean(pg, m) + VF * mean(vl, m) - ENT * mean(ent, m)

    def grad_step(c, idx):
        p, o, b = c
        g = jax.grad(loss)(p, jax.tree.map(lambda v: v[idx], b))
        u, o = tx.update(g, o, p)
        return (optax.apply_updates(p, u), o, b), None

    @jax.jit
    def iteration(st):
        key, k1, k2 = jax.random.split(st["key"], 3)
        key, k3 = jax.random.split(key)
        idx = jax.random.randint(k3, (POOL,), 0, jnp.minimum(st["n"], HISTORY))
        p = st["params"]
        pool = jax.tree.map(lambda h, a: h[idx].at[:LATEST].set(a), st["hist"], p["actor"])
        tr = rollout(p, pool, k1)
        result = tr["r"].sum(0)   # the learner's +1 / -1
        if algo == "ppo":
            adv = gae(tr)
            tr = {**tr, "adv": adv, "ret": adv + tr["v"]}
        else:
            tr = {**tr, "adv": jnp.broadcast_to(result, tr["r"].shape)}
        b = jax.tree.map(lambda v: v.reshape(T * B, *v.shape[2:]), tr)
        if algo == "ppo":
            b["logp"] = logp_ent(p["actor"], b)[0]
            idx = jax.vmap(lambda k: jax.random.permutation(k, T * B))(jax.random.split(k2, EPOCHS))
            idx = idx.reshape(EPOCHS * MINIBATCHES, -1)
        else:
            idx = jnp.arange(T * B)[None]
        p, o, _ = jax.lax.scan(grad_step, (p, st["opt"], b), idx)[0]
        stats = {"win": (result > 0).mean(), "plies": tr["alive"].sum(0).mean(),
                 "pass": mean(tr["a"] == PASS, tr["mine"]), "entropy": mean(logp_ent(p["actor"], b)[1], b["mine"])}
        return {**st, "params": p, "opt": o, "key": key}, stats

    @jax.jit
    def vs_random(params, key):   # all-zero opponents: uniform over the legal moves, pass included
        tr = rollout(params, jax.tree.map(lambda v: jnp.zeros((POOL, *v.shape)), params["actor"]), key)
        return (tr["r"].sum(0) > 0).mean()

    done = sorted(glob.glob(f"{out}/iter*.pkl"))
    if done:   # the full state: optimizer leaves stored as a plain list, so numpy alone can read a checkpoint
        with open(done[-1], "rb") as f:
            it0, st = int(done[-1][-9:-4]), pickle.load(f)
        st["opt"] = jax.tree.unflatten(jax.tree.structure(tx.init(st["params"])), st["opt"])
        st = jax.tree.map(jnp.asarray, st)
    else:
        k0, k1, k2 = jax.random.split(jax.random.PRNGKey(seed), 3)
        params = {"actor": net(k1, 82, 0.01), **({"critic": net(k2, 1, 1.0)} if algo == "ppo" else {})}
        st = {"params": params, "opt": tx.init(params), "key": k0,
              "hist": jax.tree.map(lambda v: jnp.zeros((HISTORY, *v.shape)), params["actor"]), "n": jnp.int32(0)}
        it0 = 0
    log, t0, acc = min(100, every), time.time(), []
    for it in range(it0, iters):
        if it % SNAP_EVERY == 0:
            st["hist"] = jax.tree.map(lambda h, a: h.at[st["n"] % HISTORY].set(a),   # noqa: B023 - runs right away
                                      st["hist"], st["params"]["actor"])
            st["n"] = st["n"] + 1
        st, stats = iteration(st)
        acc.append(stats)
        if (it + 1) % log == 0:
            line = {"iter": it + 1, **{k: round(float(np.mean([s[k] for s in acc])), 4) for k in acc[0]},
                    "s_per_iter": round((time.time() - t0) / len(acc), 4)}
            if (it + 1) % every == 0:
                line["vs_random"] = float(vs_random(st["params"], jax.random.PRNGKey(it)))
                with open(f"{out}/iter{it + 1:05d}.pkl", "wb") as f:
                    pickle.dump(jax.device_get({**st, "opt": jax.tree.leaves(st["opt"])}), f)
            print(json.dumps(line), flush=True)
            with open(f"{out}/log.jsonl", "a") as f:
                f.write(json.dumps(line) + "\n")
            t0, acc = time.time(), []


# ---------------------------------------------------------------- Modal

def _image(modal):
    return (modal.Image.debian_slim(python_version="3.12")
            .uv_pip_install("jax[cuda12]==0.4.38", "pgx==2.6.0", "numpy==2.2.6", "optax==0.2.4")
            .add_local_python_source("wsp"))


def _cpu_image(modal):
    return (modal.Image.debian_slim(python_version="3.12")
            .uv_pip_install("jax==0.4.38", "jaxlib==0.4.38", "pgx==2.6.0", "numpy==2.2.6")
            .add_local_dir(str(NETS), "/nets").add_local_python_source("wsp"))


def train(algo: str, seed: int, iters: int = ITERS, gpu=GPU):
    assert algo in ("ppo", "reinforce")
    modal = load_modal()
    app = modal.App(f"wsp-exp6-opp-{algo}-s{seed}", image=_image(modal))
    vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
    fn = app.function(gpu=gpu, timeout=TIMEOUT_S, volumes={"/vol": vol}, serialized=True)(_train)
    with modal.enable_output(), app.run(detach=True):
        fn.remote(algo, seed, iters)


def _volume(*args):
    return subprocess.run(["modal", "volume", *args], check=True, env=os.environ, capture_output=True, text=True).stdout


def vs(ckpts: list[str], games: int, seed: int) -> dict:
    modal = load_modal()
    app = modal.App("wsp-exp6-opp-eval", image=_cpu_image(modal))
    vol = modal.Volume.from_name(VOLUME)
    fn = app.function(cpu=CPUS, memory=8192, timeout=3600, volumes={"/vol": vol}, max_containers=16,
                      serialized=True)(_vs)
    out = {}
    with modal.enable_output(), app.run():
        for r in fn.starmap([([c], games, seed) for c in ckpts]):
            out.update(r)
    return out


def scan(games: int = 1000, seed: int = 0):
    """Every checkpoint not scanned yet, into CACHE/scan.json."""
    path = CACHE / "scan.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    load_modal()
    ckpts = [f for run in [f"runs/{a}_s{s}" for a in ("ppo", "reinforce") for s in range(3)]
             for f in sorted(_volume("ls", VOLUME, run).split()) if f.endswith(".pkl")]
    todo = [c for c in ckpts if c not in done]
    print(f"{len(todo)} checkpoints to play, {len(done)} done", flush=True)
    if todo:
        done.update(vs(todo, games, seed))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(done, indent=1))
    return done


def _chi2(res: dict, tg: dict[str, tuple[int, int]], nets) -> float:
    z2 = 0.0
    for n in nets:
        (w, g), p = tg[n], res[n]["score"]
        q = w / g
        z2 += (p - q) ** 2 / max(q * (1 - q) / g + p * (1 - p) / res[n]["games"], 1e-4)
    return z2


def _random_chi2(res: dict, role: str) -> float:
    """message.txt: "weak" and "strong" beat a random player 95 % and 98 % of the time, the hidden ones 79-98 %."""
    lo, hi = RANDOM.get(role, (0.79, 0.98))
    p, n = res["random"]["score"], res["random"]["games"]
    return (p - min(max(p, lo), hi)) ** 2 / (p * (1 - p) / n + 1e-5)   # 1e-5: the claims are rounded to 1 %


def choose(scanned: dict) -> dict[str, str]:
    """Per role the checkpoint with the smallest chi-square against Astra's networks and the claim about the random
    player; each run used for one role. The same-size PPO networks are left out as a check."""
    from itertools import permutations
    tg, best = targets(), {}
    for algo in ("ppo", "reinforce"):
        roles, runs = [r for r in ROLES if ROLES[r] == algo], [f"runs/{algo}_s{s}" for s in range(3)]
        fit = {(r, run): min((_chi2(v, tg[r], [n for n in tg[r] if n.startswith("budget")]) + _random_chi2(v, r), c)
                             for c, v in scanned.items() if c.startswith(run + "/")) for r in roles for run in runs}
        perm = min(permutations(runs), key=lambda p: sum(fit[r, run][0] for r, run in zip(roles, p)))
        best.update({r: fit[r, run][1] for r, run in zip(roles, perm)})
    return best


def calibrate(games: int = 2000, seed: int = 1000):
    """Chooses the checkpoints, plays them again with fresh games, stores them under CACHE/opponents and on the
    Volume, and writes CACHE/calibration.json."""
    picked = choose(scan())
    final = vs(list(picked.values()), games, seed)
    tg = targets()
    for role, c in picked.items():
        d = CACHE / "opponents" / role
        d.mkdir(parents=True, exist_ok=True)
        _volume("get", "--force", VOLUME, c, str(d / "checkpoint.pkl"))
        for k, v in _actor(d / "checkpoint.pkl").items():
            np.save(d / f"{k}.npy", v)
        (d / "checkpoint.pkl").unlink()
        (d / "arch.json").write_text(json.dumps({"depth": 2, "width": WIDTH}))
        source = {"checkpoint": c, "games": games, "seed": seed, "results": final[c], "targets": tg[role]}
        (d / "source.json").write_text(json.dumps(source, indent=1))
        _volume("put", "--force", VOLUME, str(d), f"opponents/{role}")
    (CACHE / "calibration.json").write_text(json.dumps({"picked": picked, "final": final, "targets": tg}, indent=1))
    table(picked, final, tg)


def table(picked, final, tg):
    total = {"astra": [0.0, 0], "same_size": [0.0, 0]}
    for role, c in picked.items():
        res = final[c]
        lo, hi = RANDOM.get(role, (0.79, 0.98))
        print(f"\n{role} = {c}   vs random {100 * res['random']['score']:.1f} % (target {100 * lo:.0f}-{100 * hi:.0f}, "
              f"chi2 {_random_chi2(res, role):.1f})")
        for n in tg[role]:
            (w, g), p = tg[role][n], res[n]["score"]
            se = 100 * (p * (1 - p) / res[n]["games"]) ** .5
            print(f"  {n:28s} target {100 * w / g:5.1f} ({g:4d})  ours {100 * p:5.1f} +- {se:.1f}  "
                  f"diff {100 * (p - w / g):+5.1f}")
        for part, nets in [("astra", [n for n in tg[role] if n.startswith("budget")]),
                           ("same_size", [n for n in tg[role] if n.startswith("ppo_same")])]:
            if nets:
                sq = sum((100 * (res[n]["score"] - tg[role][n][0] / tg[role][n][1])) ** 2 for n in nets)
                total[part][0] += sq
                total[part][1] += len(nets)
                print(f"  {part}: squared error {sq:.1f} points^2 (RMS {(sq / len(nets)) ** .5:.1f}), "
                      f"chi2 {_chi2(res, tg[role], nets):.1f} on {len(nets)}")
    for part, (sq, n) in total.items():
        print(f"\nall roles, {part}: squared error {sq:.1f} points^2 over {n} targets (RMS {(sq / n) ** .5:.1f})")


def fetch():
    load_modal()
    (CACHE / "opponents").mkdir(parents=True, exist_ok=True)
    _volume("get", "--force", VOLUME, "opponents", str(CACHE))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "train":
        train(args[0], int(args[1]), *(int(a) for a in args[2:3]))
    else:
        {"scan": scan, "calibrate": calibrate, "fetch": fetch}[cmd]()
