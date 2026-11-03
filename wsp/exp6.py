"""Experiment 6: 9x9 Go between two 243 -> 512 x 6 ReLU -> 82 policy networks, as the exp6 task specifies.

Game (upstream exp6_go_from_scratch/runs/budget_50/message.txt): pgx 2.6.0 "go_9x9", whose rules are the task's as
they stand: suicide and simple ko are masked; a stone that repeats an earlier whole-board position ends the game and
its player loses (pgx is_psk; a pass never counts); the game ends after two passes in a row or 162 moves (pgx's
limit, 9 * 9 * 2; passes count as moves); area scoring, komi 7.5. Input: pgx observation planes 0 (mover's stones),
1 (opponent's) and 16 (1 when White is to move), flattened to (row * 9 + col) * 3 + plane, the same point index as the
action. Illegal moves masked, moves sampled from the softmax (temperature 1), no search. Network A is Black in the
first half of the games. Runs on Modal, CPU only, one function call per evaluation; seeds fixed and stored.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from wsp import sandbox
from wsp.runner import UP, read_dotenv

NETS = UP / "exp6_go_from_scratch/networks"
SHAPES = {**{f"L{k}.W": (243 if k == 1 else 512, 512) for k in range(1, 7)},
          **{f"L{k}.b": (512,) for k in range(1, 7)}, "out.W": (512, 82), "out.b": (82,)}
NAMES = list(SHAPES)
KOMI, MAX_MOVES, PASS = 7.5, 162, 81
ENDS = ["two_passes", "repetition", "move_limit"]
CPUS, TIMEOUT_S = 16, 30 * 60
MAX_RUNS, RUN_TIMEOUT, MAX_OUT = 120, 900, 20_000


def load_net(folder) -> dict[str, np.ndarray]:
    assert json.load(open(f"{folder}/arch.json")) == {"depth": 6, "width": 512}, folder
    w = {k: np.load(f"{folder}/{k}.npy").astype(np.float32) for k in NAMES}
    assert all(w[k].shape == SHAPES[k] and np.isfinite(w[k]).all() for k in NAMES), folder
    return w


def saved_net(work) -> dict[str, np.ndarray]:
    """The network a session wrote: the tensors in /work/net, zeros for any never saved (as the task says)."""
    return {k: np.load(f"{work}/net/{k}.npy").astype(np.float32) if os.path.exists(f"{work}/net/{k}.npy")
            else np.zeros(s, np.float32) for k, s in SHAPES.items()}


def zero_net() -> dict[str, np.ndarray]:
    """All logits 0: uniform over the legal moves (pass included), the "random" player."""
    return {k: np.zeros(s, np.float32) for k, s in SHAPES.items()}


# ---------------------------------------------------------------- play

def play(a: dict[str, np.ndarray], b: dict[str, np.ndarray], games: int, seed: int) -> dict[str, np.ndarray]:
    """Per game, from A's side. Games are split over all jax devices; results depend on the seed and device count."""
    import jax
    import jax.numpy as jnp
    import pgx
    from pgx._src.games.go import _count_scores

    assert games % 2 == 0
    env = pgx.make("go_9x9")
    ndev, half = jax.device_count(), games // 2
    per = -(-half // ndev)
    nets = {k: jnp.stack([jnp.asarray(a[k], jnp.float32), jnp.asarray(b[k], jnp.float32)]) for k in NAMES}

    def logits(w, x):
        for k in range(1, 7):
            x = jax.nn.relu(x @ w[f"L{k}.W"] + w[f"L{k}.b"])
        return x @ w["out.W"] + w["out.b"]

    def step(s, act, r, moves, t):
        moves = moves.at[t].set(jnp.where(s.terminated, -1, act))
        s = env.step(s, act)   # a finished game stays as it is, with zero rewards
        return s, r + s.rewards[s._player_order], moves   # rewards by colour (Black, White)

    def device(nets, key):  # games (2, per): [A Black, A White]
        k0, key = jax.random.split(key)
        s = jax.vmap(jax.vmap(env.init))(jax.random.split(k0, 2 * per).reshape(2, per, 2))
        steps = jax.vmap(jax.vmap(step, in_axes=(0, 0, 0, 0, None)), in_axes=(0, 0, 0, 0, None))

        def body(c):
            s, r, moves, key, t = c
            key, k = jax.random.split(key)
            x = s.observation[..., jnp.array([0, 1, 16])].reshape(2, per, 243).astype(jnp.float32)
            w = {n: v[jnp.array([t % 2, (t + 1) % 2])] for n, v in nets.items()}   # the mover's net in each half
            lg = jnp.where(s.legal_action_mask, jax.vmap(logits)(w, x), -jnp.inf)
            s, r, moves = steps(s, jax.random.categorical(k, lg), r, moves, t)
            return s, r, moves, key, t + 1

        c = (s, jnp.zeros((2, per, 2)), jnp.full((2, per, MAX_MOVES), -1, jnp.int32), key, 0)
        s, r, moves, _, _ = jax.lax.while_loop(lambda c: ~c[0].terminated.all(), body, c)
        x = s._x
        area = jax.vmap(jax.vmap(lambda x: _count_scores(x, 9)))(x)
        return r[..., 0], area, x.is_psk, x.consecutive_pass_count >= 2, x.num_captured, moves

    out = jax.pmap(device, in_axes=(None, 0))(nets, jax.random.split(jax.random.PRNGKey(seed), ndev))

    def flat(v):  # (ndev, 2, per, ...) -> (games, ...), A Black first
        v = np.asarray(v).swapaxes(0, 1)
        v = v.reshape(2, ndev * per, *v.shape[3:])[:, :half]
        return v.reshape(games, *v.shape[2:])

    black, area, psk, two, cap, moves = map(flat, out)
    a_black = np.arange(games) < half
    sign = np.where(a_black, 1, -1)
    return {"a_black": a_black, "result": (black * sign).astype(int),
            "margin": (area[:, 0] - area[:, 1] - KOMI) * sign,
            "end": np.where(psk, 1, np.where(two, 0, 2)),
            "captured": np.where(a_black, cap[:, 0], cap[:, 1]), "lost": np.where(a_black, cap[:, 1], cap[:, 0]),
            "moves": moves}


def summary(r: dict[str, np.ndarray]) -> dict:
    win, blk, n = r["result"] > 0, r["a_black"], len(r["result"])
    a_moves = (np.arange(MAX_MOVES) % 2 == 0) == blk[:, None]
    a_pass = (r["moves"] == PASS) & a_moves
    return {"games": n, "score": float(win.mean()), "wins": int(win.sum()), "losses": int(n - win.sum()),
            "score_as_black": float(win[blk].mean()), "score_as_white": float(win[~blk].mean()),
            "mean_margin": float(r["margin"].mean()), "mean_length": float((r["moves"] >= 0).sum(1).mean()),
            "stones_captured_per_game": float(r["captured"].mean()), "stones_lost_per_game": float(r["lost"].mean()),
            "passes_per_game": float(a_pass.sum(1).mean()),
            "games_with_a_pass_before_move_60": int(a_pass[:, :60].any(1).sum()),
            **{f"end_{e}": int((r["end"] == i).sum()) for i, e in enumerate(ENDS)}}


# ---------------------------------------------------------------- Modal

def _matches(matches: dict[str, tuple[dict, dict]], games: int, seed: int, cpus: int) -> dict:
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpus}"  # XLA:CPU runs one core per device
    from importlib.metadata import version

    import jax
    out = {}
    for j, (name, (a, b)) in enumerate(matches.items()):
        t0 = time.time()
        out[name] = {**summary(play(a, b, games, seed + j)), "seed": seed + j, "seconds": round(time.time() - t0, 1)}
    return {"matches": out, "devices": jax.device_count(),
            "versions": {p: version(p) for p in ["jax", "jaxlib", "pgx", "numpy"]}}


def evaluate(matches: dict[str, tuple[dict, dict]], games: int = 2000, seed: int = 0) -> dict:
    """Each match is (A, B); match j uses seed + j."""
    profile = read_dotenv().get("MODAL_PROFILE")
    if not profile:
        sys.exit("No MODAL_PROFILE in .env; refusing to run on whatever Modal profile is active.")
    os.environ["MODAL_PROFILE"] = profile
    import modal
    if modal.config._profile != profile:
        sys.exit(f"modal is on profile {modal.config._profile!r}, not {profile!r}; refusing to run.")
    image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install("jax==0.4.38", "jaxlib==0.4.38", "pgx==2.6.0", "numpy==2.2.6")
             .add_local_python_source("wsp"))
    app = modal.App("wsp-exp6", image=image)
    fn = app.function(gpu=None, cpu=CPUS, timeout=TIMEOUT_S, serialized=True)(_matches)
    t0 = time.time()
    with modal.enable_output(), app.run():
        out = fn.remote(matches, games, seed, CPUS)
    return {**out, "games": games, "cpus": CPUS, "wall_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- conversation

def message(budget: int) -> str:
    return (UP / f"exp6_go_from_scratch/runs/budget_{budget}/message.txt").read_text(encoding="utf-8")


def _prelude() -> str:
    tree = ast.parse((UP / "harness/sandbox.py").read_text())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "PRELUDE")


def step(budget: int, out: Path):
    """Turn handler: run the reply's ```python block in the sandbox, /work kept between runs; None when final.

    Passes (budget > 0) are not implemented yet. Our choices where the original behaviour is not in the logs: a reply
    with several blocks runs the first; a pass block with budget 0 gets a note that there are no pass tools; after
    120 Python runs the model is asked for its final answer."""
    if budget:
        raise NotImplementedError("pass tools are not implemented yet")
    work = out / "work"
    work.mkdir(parents=True)
    (work / "arch.json").write_text(json.dumps({"depth": 6, "width": 512}))
    prelude, runs = _prelude(), 0

    def handle(answer: str) -> str | None:
        nonlocal runs
        blocks = re.findall(r"```(python|games|positions)[^\n]*\n(.*?)```", answer, re.S)
        if not blocks:
            return None
        kind, code = blocks[0]
        if kind != "python":
            return "There are no pass tools in this session (budget 0)."
        if runs >= MAX_RUNS:
            return f"You have used all {MAX_RUNS} Python runs. Give your final answer now, without a code block."
        runs += 1
        t0 = time.time()
        output = sandbox.run(code, prelude, {"/work": (work, "rw")}, workdir="/work", timeout=RUN_TIMEOUT,
                             max_out=MAX_OUT)
        with open(out / "tool_calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "python", "run": runs, "blocks_in_reply": len(blocks),
                                "seconds": round(time.time() - t0, 1), "code": code, "out": output}) + "\n")
        return f"OUTPUT of Python run {runs} of at most {MAX_RUNS}:\n```\n{output}\n```"

    return handle


def score(out: Path) -> dict:
    """The written network against the released same-size PPO networks and a random player, 2,000 games each."""
    net = saved_net(out / "work")
    opponents = {f"ppo_same_size_s{s}": load_net(NETS / f"ppo_same_size_lr3e-5_s{s}") for s in range(3)}
    res = evaluate({**{k: (net, v) for k, v in opponents.items()}, "random": (net, zero_net())})
    np.savez(out / "network.npz", **net)
    return {"nonzero": int(sum(np.count_nonzero(v) for v in net.values())), **res}


if __name__ == "__main__":
    # python -m wsp.exp6 <net A dir | random> <net B dir | random> [games] [seed]
    a, b = [zero_net() if d == "random" else load_net(d) for d in sys.argv[1:3]]
    games, seed = int(sys.argv[3]) if len(sys.argv) > 3 else 2000, int(sys.argv[4]) if len(sys.argv) > 4 else 0
    print(json.dumps(evaluate({f"{sys.argv[1]} vs {sys.argv[2]}": (a, b)}, games, seed), indent=1))
