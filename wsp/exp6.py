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
import sys
import time
from pathlib import Path

import numpy as np

from wsp import sandbox
from wsp.cloud import load_modal
from wsp.runner import UP
from wsp.turns import ToolLog, code_blocks

NETS = UP / "exp6_go_from_scratch/networks"
SHAPES = {**{f"L{k}.W": (243 if k == 1 else 512, 512) for k in range(1, 7)},
          **{f"L{k}.b": (512,) for k in range(1, 7)}, "out.W": (512, 82), "out.b": (82,)}
NAMES = list(SHAPES)
KOMI, MAX_MOVES, PASS = 7.5, 162, 81
ENDS = ["two_passes", "repetition", "move_limit"]
CPUS, TIMEOUT_S = 16, 30 * 60
MAX_RUNS, RUN_TIMEOUT, MAX_OUT = 120, 900, 20_000


def load_net(folder) -> dict[str, np.ndarray]:
    assert json.loads(Path(folder, "arch.json").read_text()) == {"depth": 6, "width": 512}, folder
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
    """Per game, from A's side. Games are split over all jax devices; results depend on the seed and device count.
    Also kept: the board after every ply and the top-5 moves of the mover's distribution."""
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

    def device(key):  # games (2, per): [A Black, A White]
        k0, key = jax.random.split(key)
        s = jax.vmap(jax.vmap(env.init))(jax.random.split(k0, 2 * per).reshape(2, per, 2))
        steps = jax.vmap(jax.vmap(step, in_axes=(0, 0, 0, 0, None)), in_axes=(0, 0, 0, 0, None))

        def body(c):
            s, r, moves, boards, top, key, t = c
            key, k = jax.random.split(key)
            x = s.observation[..., jnp.array([0, 1, 16])].reshape(2, per, 243).astype(jnp.float32)
            w = {n: v[jnp.array([t % 2, (t + 1) % 2])] for n, v in nets.items()}   # the mover's net in each half
            lg = jnp.where(s.legal_action_mask, jax.vmap(logits)(w, x), -jnp.inf)
            p5, m5 = jax.lax.top_k(jax.nn.softmax(lg), 5)
            top = (top[0].at[:, :, t].set(m5), top[1].at[:, :, t].set(p5))
            s, r, moves = steps(s, jax.random.categorical(k, lg), r, moves, t)
            o = s.observation.astype(jnp.int8)
            board = jnp.where(o[..., 16] > 0, o[..., 1] - o[..., 0], o[..., 0] - o[..., 1])   # +1 Black, -1 White
            boards = boards.at[:, :, t].set(board.reshape(2, per, 81).astype(jnp.int8))
            return s, r, moves, boards, top, key, t + 1

        c = (s, jnp.zeros((2, per, 2)), jnp.full((2, per, MAX_MOVES), -1, jnp.int32),
             jnp.zeros((2, per, MAX_MOVES, 81), jnp.int8),
             (jnp.zeros((2, per, MAX_MOVES, 5), jnp.int32), jnp.zeros((2, per, MAX_MOVES, 5))), key, 0)
        s, r, moves, boards, top, _, _ = jax.lax.while_loop(lambda c: ~c[0].terminated.all(), body, c)
        x = s._x
        area = jax.vmap(jax.vmap(lambda x: _count_scores(x, 9)))(x)
        return r[..., 0], area, x.is_psk, x.consecutive_pass_count >= 2, x.num_captured, moves, boards, *top

    out = jax.pmap(device)(jax.random.split(jax.random.PRNGKey(seed), ndev))

    def flat(v):  # (ndev, 2, per, ...) -> (games, ...), A Black first
        v = np.asarray(v).swapaxes(0, 1)
        v = v.reshape(2, ndev * per, *v.shape[3:])[:, :half]
        return v.reshape(games, *v.shape[2:])

    black, area, psk, two, cap, moves, boards, top_m, top_p = map(flat, out)
    a_black = np.arange(games) < half
    sign = np.where(a_black, 1, -1)
    return {"a_black": a_black, "result": (black * sign).astype(int),
            "margin": (area[:, 0] - area[:, 1] - KOMI) * sign,
            "end": np.where(psk, 1, np.where(two, 0, 2)),
            "captured": np.where(a_black, cap[:, 0], cap[:, 1]), "lost": np.where(a_black, cap[:, 1], cap[:, 0]),
            "moves": moves, "boards": boards, "top5_moves": top_m, "top5_probs": top_p}


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


def _remote(fn, *args):
    modal = load_modal()
    image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install("jax==0.4.38", "jaxlib==0.4.38", "pgx==2.6.0", "numpy==2.2.6")
             .add_local_python_source("wsp"))
    app = modal.App("wsp-exp6", image=image)
    f = app.function(gpu=None, cpu=CPUS, timeout=TIMEOUT_S, serialized=True)(fn)
    with modal.enable_output(), app.run():
        return f.remote(*args)


def evaluate(matches: dict[str, tuple[dict, dict]], games: int = 2000, seed: int = 0) -> dict:
    """Each match is (A, B); match j uses seed + j."""
    t0 = time.time()
    out = _remote(_matches, matches, games, seed, CPUS)
    return {**out, "games": games, "cpus": CPUS, "wall_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- passes (budget > 0)

GAMES_README = ("games.json: one entry per game (opponent, net_color, result +1/-1, margin = net's area score minus "
                "the opponent's incl. komi, end, stones captured / lost, moves: ply, by, move (row*9+col, 81 = "
                "pass), and for the network's moves its top-5 [move, probability]). boards.npy: (games, 162, 81) "
                "int8, the board AFTER each ply (1 Black, -1 White, 0 empty; index row*9+col); valid.npy: (games, "
                "162) which plies were played.")
POSITIONS_README = ("inputs.npy (P, 243) the network inputs; legal.npy (P, 82) the legal moves (for 'board' positions: "
                    "the empty points and pass; suicide and ko are not checked); logits.npy, probs.npy (P, 82); ")
END_NAMES = ["two passes", "repetition", "move limit"]


def _games_pass(net: dict, opponents: dict[str, dict], counts: dict[str, int], seed: int, cpus: int) -> dict:
    """Runs on Modal: the games of one pass, opponents in name order, the network Black in the first half of each."""
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpus}"
    per, games, boards, valid = {}, [], [], []
    for j, name in enumerate(sorted(counts)):
        r = play(net, opponents[name], counts[name], seed + j)
        per[name] = summary(r)
        for i in range(counts[name]):
            net_first = bool(r["a_black"][i])
            moves = []
            for ply, m in enumerate(r["moves"][i]):
                if m < 0:
                    break
                rec = {"ply": ply, "by": "net" if (ply % 2 == 0) == net_first else "opponent", "move": int(m)}
                if rec["by"] == "net":
                    rec["top5"] = [[int(a), round(float(b), 4)] for a, b in zip(r["top5_moves"][i, ply],
                                                                                 r["top5_probs"][i, ply])]
                moves.append(rec)
            games.append({"game": len(games), "opponent": name, "net_color": "black" if net_first else "white",
                          "result": int(r["result"][i]), "margin": float(r["margin"][i]),
                          "end": END_NAMES[r["end"][i]], "stones_captured_by_net": int(r["captured"][i]),
                          "stones_lost_by_net": int(r["lost"][i]), "moves": moves})
        boards.append(r["boards"])
        valid.append(r["moves"] >= 0)
    return {"per_opponent": per, "games": games, "boards": np.concatenate(boards), "valid": np.concatenate(valid)}


def _positions_pass(net: dict, positions: list, activations: bool) -> dict:
    """Runs on Modal: the network on the given positions ("board" ones read as they stand, "moves" ones replayed
    with the game's rules)."""
    import jax
    import pgx
    env = pgx.make("go_9x9")
    init, step = jax.jit(env.init), jax.jit(env.step)   # unjitted, one replayed move takes seconds
    xs, legal = [], []
    for p in positions:
        if "board" in p:
            rows = p["board"]
            assert len(rows) == 9 and all(len(r) == 9 and set(r) <= set("XO.") for r in rows), \
                "a board is 9 strings of 9 characters X, O or ."
            assert p.get("to_move") in ("black", "white"), 'to_move is "black" or "white"'
            b = np.array([[{"X": 1, "O": -1, ".": 0}[c] for c in r] for r in rows]).reshape(81)
            me = 1 if p["to_move"] == "black" else -1
            x = np.stack([b == me, b == -me, np.full(81, me == -1)], 1).astype(np.float32).reshape(243)
            lg = np.append(b == 0, True)
        else:
            s = init(jax.random.PRNGKey(0))
            for m in p["moves"]:
                a = PASS if m == "pass" else int(m)
                assert 0 <= a <= PASS and bool(s.legal_action_mask[a]) and not bool(s.terminated), f"illegal move {m}"
                s = step(s, a)
            x = np.asarray(s.observation)[..., [0, 1, 16]].reshape(243).astype(np.float32)
            lg = np.asarray(s.legal_action_mask)
        xs.append(x)
        legal.append(lg)
    h, acts = np.stack(xs), {}
    for k in range(1, 7):
        h = np.maximum(h @ net[f"L{k}.W"] + net[f"L{k}.b"], 0)
        acts[f"act_L{k}"] = h.astype(np.float32)
    logits = (h @ net["out.W"] + net["out.b"]).astype(np.float32)
    legal = np.stack(legal)
    z = np.where(legal, logits, -np.inf)
    probs = np.where(legal, np.exp(z - z.max(1, keepdims=True)), 0)
    probs = (probs / probs.sum(1, keepdims=True)).astype(np.float32)
    return {"inputs": np.stack(xs), "legal": legal, "logits": logits, "probs": probs,
            **(acts if activations else {})}


def _move_name(m: int) -> str:
    return "pass" if m == PASS else f"({m // 9},{m % 9})"


# ---------------------------------------------------------------- conversation

def message(budget: int) -> str:
    return (UP / f"exp6_go_from_scratch/runs/budget_{budget}/message.txt").read_text(encoding="utf-8")


def _prelude() -> str:
    tree = ast.parse((UP / "harness/sandbox.py").read_text())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "PRELUDE")


def practice_opponents() -> dict[str, dict]:
    from wsp.exp6_opponents import opponent
    return {"random": zero_net(), "weak": opponent("weak"), "strong": opponent("strong")}


class Turns:
    """Turn handler: run the reply's ```python block in the sandbox (/work kept between runs) or its ```games /
    ```positions block as a pass; None when final.

    As upstream's logs show: one counter for both passes (/work/pass_<k>/), the result templates and README texts,
    "Passes left: n." after each pass, the final answer once every pass is used. Our choices where the logs show
    nothing: a reply with several blocks runs the first; a final answer with passes left, a pass block after the
    last pass, a failed pass request and a reply after 120 Python runs get the texts below; the seed of games pass k
    is 1000 * k (+ j for the j-th opponent)."""

    def __init__(self, budget: int, out: Path):
        self.budget, self.out, self.work = budget, out, out / "work"
        (self.work / "net").mkdir(parents=True)
        (self.work / "arch.json").write_text(json.dumps({"depth": 6, "width": 512}))
        self.prelude = _prelude()
        self.opponents = practice_opponents() if budget else {}
        self.runs = self.passes = 0
        self.log = ToolLog(out / "tool_calls.jsonl")

    def __call__(self, answer: str) -> str | None:
        blocks = code_blocks(answer, "python|games|positions", info=r"[^\n]*")   # as run: anything may follow the kind
        if not blocks:
            if self.passes < self.budget:
                self.log({"kind": "reminder", "passes": self.passes})
                return (f"Passes left: {self.budget - self.passes} of {self.budget}. The session continues until every "
                        "pass is used; reply with a ```python, ```games or ```positions block.")
            return None
        kind, code = blocks[0].groups()
        if kind == "python":
            return self.run_python(code, len(blocks))
        return self.run_pass(kind, code)

    def run_python(self, code: str, n_blocks: int) -> str:
        if self.runs >= MAX_RUNS:
            return f"You have used all {MAX_RUNS} Python runs. Give your final answer now, without a code block."
        self.runs += 1
        t0 = time.time()
        output = sandbox.run(code, self.prelude, {"/work": (self.work, "rw")}, workdir="/work", timeout=RUN_TIMEOUT,
                             max_out=MAX_OUT)
        self.log({"kind": "python", "run": self.runs, "blocks_in_reply": n_blocks,
                  "seconds": round(time.time() - t0, 1), "code": code, "out": output})
        return f"OUTPUT of Python run {self.runs} of at most {MAX_RUNS}:\n```\n{output}\n```"

    def run_pass(self, kind: str, code: str) -> str:
        if not self.budget:
            return "There are no pass tools in this session (budget 0)."
        if self.passes >= self.budget:
            return f"All {self.budget} passes are used; this block was not run."
        k, t0 = self.passes + 1, time.time()
        try:
            req = json.loads(code)
            res = (self.games if kind == "games" else self.positions)(req, k)
        except Exception as e:   # a failed request is reported and not counted
            self.log({"kind": kind, "k": k, "request": code, "error": f"{type(e).__name__}: {e}",
                      "seconds": round(time.time() - t0, 1)})
            return f"The {kind} pass failed and was not counted: {type(e).__name__}: {e}"
        self.passes = k
        (self.out / "requests").mkdir(exist_ok=True)
        (self.out / "requests" / f"{kind}_{k}.json").write_text(json.dumps(req, indent=1))
        self.log({"kind": kind, "k": k, "request": req, "result": res, "seconds": round(time.time() - t0, 1)})
        return (f"{kind.capitalize()} pass {k} of {self.budget} done.\n```\n{json.dumps(res)}\n```\n"
                f"Passes left: {self.budget - k}.")

    def games(self, req: dict, k: int) -> dict:
        # asserts on purpose: the model sees "AssertionError: ..." in the failed-pass message, as in the runs
        counts = req["games"]
        assert isinstance(counts, dict) and counts and set(counts) <= set(self.opponents), \
            f"games is a dict of opponent -> count; opponents: {sorted(self.opponents)}"
        assert all(isinstance(c, int) and c > 0 and c % 2 == 0 for c in counts.values()), "counts are positive and even"
        assert sum(counts.values()) <= 128, "at most 128 games in total"
        res = _remote(_games_pass, saved_net(self.work), self.opponents, counts, 1000 * k, CPUS)
        d = self.work / f"pass_{k}"
        d.mkdir()
        (d / "summary.json").write_text(json.dumps(res["per_opponent"], indent=1))
        (d / "games.json").write_text(json.dumps(res["games"]))
        np.save(d / "boards.npy", res["boards"])
        np.save(d / "valid.npy", res["valid"])
        (d / "README.txt").write_text(GAMES_README)
        return {"ok": True, "pass": k, "kind": "games", "per_opponent": res["per_opponent"],
                "files": f"/work/pass_{k}/", "readme": GAMES_README}

    def positions(self, req: dict, k: int) -> dict:
        pos, act = req["positions"], bool(req.get("activations", False))
        assert isinstance(pos, list) and 0 < len(pos) <= 256, "positions is a list of 1 to 256 positions"
        res = _remote(_positions_pass, saved_net(self.work), pos, act)
        d = self.work / f"pass_{k}"
        d.mkdir()
        for name, v in res.items():
            np.save(d / f"{name}.npy", v)
        readme = POSITIONS_README + ("act_L1.npy .. act_L6.npy (P, 512) the hidden activations after ReLU." if act
                                     else 'no activations (ask with "activations": true).')
        (d / "README.txt").write_text(readme)
        top3 = [[[_move_name(int(m)), round(float(p[m]), 4)] for m in np.argsort(-p, kind="stable")[:3]]
                for p in res["probs"]]
        return {"ok": True, "pass": k, "kind": "positions", "positions": len(pos), "top3_per_position": top3,
                "files": f"/work/pass_{k}/", "readme": readme}


def score(out: Path) -> dict:
    """The written network, 2,000 games each: against the hidden opponents (rebuilt, wsp.exp6_opponents; seeds 777 + j
    in upstream's order) and against the released same-size PPO networks and a random player (seeds 0 + j)."""
    from wsp.exp6_opponents import opponent
    net = saved_net(out / "work")
    hidden = evaluate({r: (net, opponent(r)) for r in ["ppo_s0", "ppo_s2", "reinforce_s0", "reinforce_s2", "random"]},
                      seed=777)
    same = {f"ppo_same_size_s{s}": load_net(NETS / f"ppo_same_size_lr3e-5_s{s}") for s in range(3)}
    released = evaluate({**{k: (net, v) for k, v in same.items()}, "random": (net, zero_net())})
    return {"nonzero": int(sum(np.count_nonzero(v) for v in net.values())), "hidden": hidden, "released": released}


if __name__ == "__main__":
    # python -m wsp.exp6 <net A dir | random> <net B dir | random> [games] [seed]
    a, b = [zero_net() if d == "random" else load_net(d) for d in sys.argv[1:3]]
    games, seed = int(sys.argv[3]) if len(sys.argv) > 3 else 2000, int(sys.argv[4]) if len(sys.argv) > 4 else 0
    print(json.dumps(evaluate({f"{sys.argv[1]} vs {sys.argv[2]}": (a, b)}, games, seed), indent=1))
