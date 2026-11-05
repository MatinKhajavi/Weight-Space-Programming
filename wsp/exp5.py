"""Experiment 5: read an `edits` reply into a 600-32-32-6 network, and score it on MinAtar 1.0.15.

Edit language and scoring follow the exp5 prompt (upstream exp5_write_from_scratch/runs/<game>/message.txt):
only the LAST ```edits block is read; one statement per line, '#' starts a comment, a list may continue over
lines; ranges a:b, a:b:s, ':'; ops = += -= *=; an unreadable statement is skipped; |v| <= 1e6.
Scoring: mean score over 1,000 episodes of the official MinAtar 1.0.15 Environment (sticky actions 0.1,
difficulty ramping on), actions sampled from the softmax of the logits, episodes cut at 10,000 steps.
The original seeds are not recorded; ours are fixed and stored with every result.
"""
from __future__ import annotations

import ast
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}
NPZ_KEYS = {"W1": "actor/dense1/kernel", "b1": "actor/dense1/bias", "W2": "actor/dense2/kernel",
            "b2": "actor/dense2/bias", "W3": "actor/logits/kernel", "b3": "actor/logits/bias"}

# The 6 input channels <- the game's own channels ("How the 6 channels are built" in each prompt).
CHANNEL_MAP = {
    "breakout": [["paddle"], [], ["brick"], ["ball"], ["trail"], []],
    "seaquest": [["sub_front", "sub_back"], ["enemy_fish", "enemy_sub", "enemy_bullet"],
                 ["enemy_fish", "enemy_sub", "diver"], ["friendly_bullet"], ["trail", "sub_back"],
                 ["oxygen_guage", "diver_guage"]],
    "asterix": [["player"], ["enemy"], ["gold"], [], ["trail"], []],
}

MAX_STEPS, STICKY = 10_000, 0.1


# ---------------------------------------------------------------- parsing

def edit_statements(answer: str) -> list[str]:
    blocks = re.findall(r"```edits[^\n]*\n(.*?)```", answer, re.S)
    if not blocks:
        return []
    stmts, cur = [], ""
    for line in blocks[-1].split("\n"):
        line = line.split("#")[0].rstrip()
        if not line.strip() and not cur:
            continue
        cur = f"{cur} {line.strip()}" if cur else line.strip()
        if cur.count("[") - cur.count("]") <= 0 and not re.search(r"=\s*$", cur):
            stmts.append(cur)
            cur = ""
    if cur:
        stmts.append(cur)
    return stmts


def index(part: str):
    part = part.strip()
    if part == ":":
        return slice(None)
    if ":" in part:
        return slice(*[int(x) if x.strip() else None for x in part.split(":")])
    return int(part)


def apply_edits(stmts: list[str]) -> tuple[dict[str, np.ndarray], dict]:
    w = {k: np.zeros(s, np.float64) for k, s in SHAPES.items()}
    skipped = []
    for s in stmts:
        try:
            m = re.match(r"(W[123]|b[123])\[(.*?)\]\s*([+\-*]?=)\s*(.*)$", s)
            if not m:
                raise ValueError("not a statement")
            name, ix, op, val = m.groups()
            idx = tuple(index(p) for p in ix.split(","))
            v = np.array(ast.literal_eval(val.strip()), dtype=np.float64)
            if np.any(np.abs(v) > 1e6):
                raise ValueError("|v| > 1e6")
            cur = w[name][idx]
            if v.ndim and v.shape != np.shape(cur):
                raise ValueError(f"shape {v.shape} != {np.shape(cur)}")
            w[name][idx] = {"=": v, "+=": cur + v, "-=": cur - v, "*=": cur * v}[op]
        except Exception as e:  # unreadable statements are skipped, as the prompt says
            skipped.append({"statement": s[:200], "error": str(e)[:200]})
    w32 = {k: v.astype(np.float32) for k, v in w.items()}
    report = {"statements": len(stmts), "applied": len(stmts) - len(skipped), "skipped": skipped,
              "nonzero": int(sum(np.count_nonzero(v) for v in w32.values()))}
    return w32, report


def read_reply(answer: str) -> tuple[dict[str, np.ndarray], dict]:
    return apply_edits(edit_statements(answer))


def load_npz(path: str) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[NPZ_KEYS[k]].astype(np.float32) for k in SHAPES}


def save_npz(path: str, w: dict[str, np.ndarray]) -> None:
    np.savez(path, **{NPZ_KEYS[k]: w[k] for k in SHAPES})


# ---------------------------------------------------------------- scoring

def _episodes(args):
    game, w, seeds = args
    from minatar import Environment
    W = {k: v.astype(np.float64) for k, v in w.items()}
    env = Environment(game, sticky_action_prob=STICKY, difficulty_ramping=True)
    env.seed(seeds[0])                     # one official env per worker; episodes run back to back in it
    rng = np.random.default_rng(seeds[0] + 1_000_003)
    chans = [[env.env.channels[c] for c in cs] for cs in CHANNEL_MAP[game]]
    out = []
    for _ in seeds:
        env.reset()
        ret, t, done = 0.0, 0, False
        while not done and t < MAX_STEPS:
            s = env.state()
            x = np.zeros((10, 10, 6))
            for j, cs in enumerate(chans):
                if cs:
                    x[:, :, j] = s[:, :, cs].any(axis=2)
            x = x.reshape(-1)
            lg = np.tanh(np.tanh(x @ W["W1"] + W["b1"]) @ W["W2"] + W["b2"]) @ W["W3"] + W["b3"]
            p = np.exp(lg - lg.max())
            a = int(rng.choice(6, p=p / p.sum()))
            r, done = env.act(a)
            ret += r
            t += 1
        out.append((ret, t))
    return out


def evaluate(w: dict[str, np.ndarray], game: str, episodes: int = 1000, workers: int = 10, seed: int = 0) -> dict:
    seeds = list(range(seed, seed + episodes))
    chunks = [seeds[i::workers] for i in range(workers)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = [r for part in ex.map(_episodes, [(game, w, c) for c in chunks]) for r in part]
    rets = np.array([r for r, _ in res])
    lens = np.array([t for _, t in res])
    return {"game": game, "episodes": episodes, "mean": float(rets.mean()),
            "se": float(rets.std(ddof=1) / np.sqrt(episodes)), "max": float(rets.max()),
            "mean_length": float(lens.mean()), "capped_at_10000": int((lens >= MAX_STEPS).sum()),
            "seed": seed, "workers": workers, "minatar": "1.0.15", "sticky": STICKY,
            "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    # python -m wsp.exp5 <game> <hand.npz | answer.md> [episodes] [workers]
    game, src = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    weights = load_npz(src) if src.endswith(".npz") else read_reply(Path(src).read_text(encoding="utf-8"))[0]
    print(json.dumps(evaluate(weights, game, n, k)))
