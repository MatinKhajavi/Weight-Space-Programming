"""Our scorers and edit readers (exps 1-5) must reproduce the original results on Astra's released answers."""
import json

import numpy as np
import pytest

from wsp.exp3 import parse, score
from wsp.exp4 import NAMES, apply_edits, shown_network
from wsp.exp4 import read_reply as read_exp4_reply
from wsp.exp5 import edit_statements, load_npz
from wsp.exp5 import read_reply as read_exp5_reply
from wsp.runner import UP
from wsp.scoring import GAME_KEYWORDS, SHAPE_KEYWORDS, answer_field, map_label, normalize, score_exp1, score_exp2


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
    net_id = next(i for i, v in key.items() if v["shape"] == shape)
    released = _released("exp2_diffusion")[net_id]
    ours = score_exp2((UP / f"exp2_diffusion/runs/{shape}/answer.md").read_text(), shape)
    assert ours["guess"] == released["guess"]
    assert ours["correct"] == released["correct"]
    assert ours["confidence"] == released["confidence"]


@pytest.mark.parametrize("text,keywords,label,status", [
    ("pong or breakout", GAME_KEYWORDS, None, "hedged"),
    ("a snake game", GAME_KEYWORDS, None, "unmapped"),
    ("a white blob on a dark background, around the centre", SHAPE_KEYWORDS, None, "unmapped"),  # "round" in words
    ("frogger, a traffic-crossing game", GAME_KEYWORDS, "freeway", "mapped"),
])
def test_edge_cases(text, keywords, label, status):
    guess, st, _ = map_label(normalize(text), keywords)
    assert (guess, st) == (label, status)


def test_exp1_field_needs_colon_or_line_end():
    assert answer_field("Game: Pong\nGames ruled out: Breakout", "game") == "Pong"
    assert answer_field("## Game\n**Pac-Man**", "game") == "**Pac-Man**"
    assert answer_field("**Images:**\na ring", "images") == "a ring"


@pytest.mark.parametrize("game", ["breakout", "seaquest", "asterix"])
def test_exp5_parser_rebuilds_released_networks(game):
    ours, report = read_exp5_reply((UP / f"exp5_write_from_scratch/runs/{game}/reply/answer.md").read_text())
    released = load_npz(str(UP / f"exp5_write_from_scratch/networks/{game}/hand.npz"))
    assert report["skipped"] == []
    assert all(np.array_equal(ours[k], released[k]) for k in released)


def test_exp4_edits_rebuild_released_network():
    ours, report = read_exp4_reply((UP / "exp4_fix_car/runs/oneshot_repair/reply/answer.md").read_text())
    released = np.load(UP / "exp4_fix_car/networks/edited_actor.npz")
    assert report["skipped"] == [] and report["applied"] == 51
    assert all(np.array_equal(ours[k], released[k]) for k in NAMES)
    assert report["changed"] == 6073


def test_exp4_edit_language():
    text = ("```edits\nZERO actor4.W\nactor_head.b[0:3] = [1,\n  2, 3]   # a list over two lines\n"
            "embed.W[0, 0:2] = [1]\nfoo.W[0] = 1\nactor0.b[0] = 2e6\n```")
    w, report = apply_edits(edit_statements(text), shown_network())
    assert (w["actor4.W"] == 0).all() and list(w["actor_head.b"][:3]) == [1, 2, 3]
    assert report["applied"] == 2 and len(report["skipped"]) == 3   # wrong length, unknown tensor, |v| > 1e6


@pytest.mark.parametrize("run", ["python_r0", "python_r1", "romaji_r0", "romaji_r1", "english_r0", "english_r1"])
def test_exp3_matches_released_scores(run):
    corpus, attempt = run.split("_")
    rows = json.loads((UP / "exp3_language/eval/scores.json").read_text())
    released = next(r for r in rows if r["id"].endswith(attempt) and r["corpus"] == corpus)
    ours = score((UP / f"exp3_language/runs/{run}/final_answer.md").read_text(), corpus)
    assert ours["corpus_guess"] == released["corpus_guess"]
    assert ours["head_labels"] == released["head_labels"]
    assert ours["copy_named"] == sorted(released["copy_named"])
    assert ours["probabilities"] == released["probabilities"]
    assert ours["copy_false"] == [] and ours["copy_missed"] == []   # the post: every copying head found, none extra


def test_exp3_head_lines_in_any_grouping():
    text = ("Heads:\n- L0H0: **copy** — x\n- L1H0 – L1H5 local (all six)\n- L2H0, L2H2: broad\n"
            "- L2H3: unclear. L2H4: previous.\n- L3H0 to L3H5: broad\n- L0H1/H2 previous\n")
    h = parse(text)["head_labels"]
    assert h["L0H0"] == "copy" and all(h[f"L1H{i}"] == "local" for i in range(6))
    assert h["L2H0"] == h["L2H2"] == "broad" and h["L2H3"] == "unclear" and h["L2H4"] == "previous"
    assert all(h[f"L3H{i}"] == "broad" for i in range(6)) and h["L0H1"] == h["L0H2"] == "previous"
