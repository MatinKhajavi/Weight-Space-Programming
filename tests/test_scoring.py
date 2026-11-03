"""Our exp1/exp2 scorers must reproduce the original scores on Astra's released answers."""
import json
from pathlib import Path

import pytest

from wsp.scoring import map_label, normalize, score_exp1, score_exp2, GAME_KEYWORDS, SHAPE_KEYWORDS

UP = Path(__file__).resolve().parents[1] / "upstream"


def _released(exp_dir):
    scores = json.loads((UP / exp_dir / "eval" / "scores.json").read_text())
    return {r["id"]: r for r in scores}


@pytest.mark.parametrize("game", ["breakout", "pong", "pacman", "freeway"])
def test_exp1_matches_released_scores(game):
    key = json.loads((UP / "exp1_which_game/eval/answer_key.json").read_text())
    net_id = next(i for i, v in key.items() if v["game"] == game)
    released = _released("exp1_which_game")[net_id]
    ours = score_exp1((UP / f"exp1_which_game/runs/{game}/answer.md").read_text(), game)
    assert ours["guess"] == released["guess"]
    assert ours["correct"] == released["correct"]
    assert ours["confidence"] == released["confidence"]


@pytest.mark.parametrize("shape", ["circle", "square", "triangle"])
def test_exp2_matches_released_scores(shape):
    key = json.loads((UP / "exp2_diffusion/eval/answer_key.json").read_text())
    net_id = next(i for i, v in key.items() if (v.get("shape") if isinstance(v, dict) else v) == shape)
    released = _released("exp2_diffusion")[net_id]
    ours = score_exp2((UP / f"exp2_diffusion/runs/{shape}/answer.md").read_text(), shape)
    assert ours["guess"] == released["guess"]
    assert ours["correct"] == released["correct"]
    assert ours["confidence"] == released["confidence"]


@pytest.mark.parametrize("text,label,status", [
    ("pong or breakout", None, "hedged"),
    ("a snake game", None, "unmapped"),
    ("a white blob on a dark background, around the centre", None, "unmapped"),  # "round" inside words
    ("frogger, a traffic-crossing game", "freeway", "mapped"),
])
def test_edge_cases(text, label, status):
    kw = SHAPE_KEYWORDS if "blob" in text else GAME_KEYWORDS
    guess, st, _ = map_label(normalize(text), kw)
    assert (guess, st) == (label, status)


@pytest.mark.parametrize("game", ["breakout", "seaquest", "asterix"])
def test_exp5_parser_rebuilds_released_networks(game):
    import numpy as np
    from wsp.exp5 import load_npz, read_reply
    ours, report = read_reply((UP / f"exp5_write_from_scratch/runs/{game}/reply/answer.md").read_text())
    released = load_npz(str(UP / f"exp5_write_from_scratch/networks/{game}/hand.npz"))
    assert report["skipped"] == []
    assert all(np.array_equal(ours[k], released[k]) for k in released)
