"""How does a written exp5 Breakout network lose the ball? Plays games and classifies each loss."""
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minatar import Environment

from wsp.exp5 import CHANNEL_MAP, load_npz

DIRS = {0: "up-left", 1: "up-right", 2: "down-right", 3: "down-left"}


def run(path, games=300, seed=0):
    W = {k: v.astype(np.float64) for k, v in load_npz(path).items()}
    env = Environment("breakout", sticky_action_prob=0.1)
    env.seed(seed)
    rng = np.random.default_rng(seed + 1)
    chans = [[env.env.channels[c] for c in cs] for cs in CHANNEL_MAP["breakout"]]
    scores, lengths, losses = [], [], collections.Counter()
    gap_at_loss, examples = collections.Counter(), []
    for _g in range(games):
        env.reset()
        e, ret, t, hist = env.env, 0, 0, []
        while not e.terminal and t < 10000:
            s = env.state()
            x = np.zeros((10, 10, 6))
            for j, cs in enumerate(chans):
                if cs:
                    x[:, :, j] = s[:, :, cs].any(axis=2)
            lg = np.tanh(np.tanh(x.reshape(-1) @ W["W1"] + W["b1"]) @ W["W2"] + W["b2"]) @ W["W3"] + W["b3"]
            p = np.exp(lg - lg.max()); p /= p.sum()
            a = int(rng.choice(6, p=p))
            hist.append((e.ball_y, e.ball_x, e.ball_dir, e.last_y, e.last_x, e.pos, a))
            r, _ = env.act(a)
            ret += r; t += 1
        scores.append(ret); lengths.append(t)
        if e.terminal:
            y, bx, d, ly, lx, pos, a = hist[-1]
            # ball came down to row 9 between ball (row 8) and the next column; how far was the paddle?
            land = bx + (1 if d == 2 else -1)
            gap_at_loss[min(abs(e.pos - bx), abs(e.pos - land))] += 1
            # what happened just before: was the ball recently reflected off the paddle / a wall / a brick?
            kind = "after wall bounce" if any(h[1] in (0, 9) for h in hist[-6:]) else "open field"
            losses[f"{DIRS[d]} ball, {kind}"] += 1
            if len(examples) < 4:
                examples.append([(h[0], h[1], DIRS[h[2]][:2], h[5], "nlurdf"[h[6]]) for h in hist[-10:]])
    s = np.array(scores)
    print(f"{path}: mean {s.mean():.1f}, median {np.median(s):.0f}, games scoring 0: {(s == 0).sum()}, "
          f"mean length {np.mean(lengths):.0f}, losses {sum(losses.values())}/{games}")
    print("  loss kinds:", dict(losses.most_common()))
    print("  paddle distance from the ball's landing cells at the loss:", dict(sorted(gap_at_loss.items())))
    for ex in examples[:3]:
        print("  last 10 steps (ball row, col, dir, paddle col, action):", ex)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p)
