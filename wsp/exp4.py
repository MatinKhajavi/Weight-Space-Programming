"""Experiment 4: apply an `edits` reply to the car network as shown in the prompt, and score it in Kinetix.

Edit language (upstream exp4_fix_car/runs/oneshot_repair/reply/prompt.txt, "EDIT FORMAT"): only the LAST ```edits
block is read; one statement per line, '#' starts a comment, a long list may continue over lines;
`tensor[i, j] op value` with indices n, a:b (half-open) or ':', ops = += -= *=, value a number or a list with as many
values as the range has entries; `ZERO tensor`; an unreadable statement is skipped; |v| <= 1e6. Statements apply in
order to the network AS SHOWN (integer x scale per tensor), not to the full-precision weights.

Scoring: Kinetix @ 80ee9c29 with jax 0.4.38, the env the original PPO run trained on (env=symbolic, env_size=l):
level l/h2_car_ramp, symbolic_flat observation (679, clipped to [-10, 10]), multi_discrete actions, frame_skip 2,
dense_reward_scale 0.2, max_timesteps 256. Each action entry is sampled from the softmax of its logits. Outcomes:
success (green touches blue), red, timeout (256 steps), invalid (non-finite state). Progress = 1 - min_t d_t / d_0
(d = distance between the centres of the green wheel and the blue box; 1 on success). Runs on Modal, one function
call per evaluation. The original seeds are not recorded; ours are fixed and stored with every result.
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

from wsp.cloud import load_modal
from wsp.exp5 import edit_statements, index
from wsp.runner import UP

PROMPT = UP / "exp4_fix_car/runs/oneshot_repair/reply/prompt.txt"
KINETIX_COMMIT = "80ee9c292837c825cafaf9889323fdcc212efd1b"
GPU, CPUS, TIMEOUT_S = None, 16, 20 * 60  # GPU None = CPU only; GPUs need a payment method on the Modal workspace
SPLITS = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 14), (14, 16)]   # logit columns of the 6 action entries
OUTCOMES = ["success", "red", "timeout", "invalid"]
NAMES = ["embed.W", "embed.b"] + [f"actor{k}.{p}" for k in range(5) for p in "Wb"] + ["actor_head.W", "actor_head.b"]
_G = "params/GeneralActorCriticRNN_0/"
NPZ_KEYS = {"embed.W": "params/Dense_0/kernel", "embed.b": "params/Dense_0/bias",
            "actor_head.W": _G + "Dense_0/kernel", "actor_head.b": _G + "Dense_0/bias",
            **{f"actor{k}.{p}": _G + f"DenseAndActivation_{k}/Dense_0/{'kernel' if p == 'W' else 'bias'}"
               for k in range(5) for p in "Wb"}}


# ---------------------------------------------------------------- parsing

def shown_network(path=PROMPT) -> dict[str, np.ndarray]:
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    head = re.compile(r"^([a-z_0-9]+\.[Wb]) \((\d+)(?: x (\d+))?\), scale (\S+):")
    w = {}
    for i, line in enumerate(lines):
        m = head.match(line)
        if not m:
            continue
        name, rows, cols, scale = m.group(1), int(m.group(2)), m.group(3), float(m.group(4))
        if cols:
            body = [lines[i + 1 + r].split(":", 1) for r in range(rows)]
            assert [int(r) for r, _ in body] == list(range(rows))
            ints = np.array([[int(v) for v in vals.split()] for _, vals in body], dtype=np.int64)
        else:
            ints = np.array([int(v) for v in lines[i + 1].split()], dtype=np.int64)
        w[name] = ints * scale
    assert list(w) == NAMES
    return w


def apply_edits(stmts: list[str], base: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    w = {k: v.copy() for k, v in base.items()}
    skipped, written, touched = [], 0, set()
    for s in stmts:
        try:
            m = re.match(r"ZERO\s+(\S+)$", s)
            if m:
                w[m.group(1)][...] = 0
                written += w[m.group(1)].size
                touched.add(m.group(1))
                continue
            m = re.match(r"([a-z_0-9]+\.[Wb])\s*\[(.*?)\]\s*([+\-*]?=)\s*(.*)$", s)
            if not m:
                raise ValueError("not a statement")
            name, ix, op, val = m.groups()
            idx = tuple(index(p) for p in ix.split(","))
            v = np.array(ast.literal_eval(val.strip()), dtype=np.float64)
            if not np.all(np.abs(v) <= 1e6):
                raise ValueError("|v| > 1e6")
            cur = w[name][idx]
            if v.ndim and v.shape != np.shape(cur):
                raise ValueError(f"shape {v.shape} != {np.shape(cur)}")
            w[name][idx] = {"=": v, "+=": cur + v, "-=": cur - v, "*=": cur * v}[op]
            written += np.size(cur)
            touched.add(name)
        except Exception as e:  # unreadable statements are skipped, as the prompt says
            skipped.append({"statement": s[:200], "error": f"{type(e).__name__}: {e}"[:200]})
    changed = {k: int((w[k] != base[k]).sum()) for k in NAMES}
    entries = sum(v.size for v in base.values())
    report = {"statements": len(stmts), "applied": len(stmts) - len(skipped), "skipped": skipped,
              "entries_written": written, "tensors_touched": sorted(touched), "changed": sum(changed.values()),
              "entries": entries, "frac_changed": sum(changed.values()) / entries, "changed_per_tensor": changed}
    return w, report


def read_reply(answer: str) -> tuple[dict[str, np.ndarray], dict]:
    w, report = apply_edits(edit_statements(answer), shown_network())
    return w, {"block": "```edits" in answer, **report}


def load_npz(path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k if k in z else NPZ_KEYS[k]].astype(np.float64) for k in NAMES}


def save_npz(path, w: dict[str, np.ndarray]) -> None:
    np.savez(path, **w)


# ---------------------------------------------------------------- scoring

def _rollouts(nets: dict[str, dict[str, np.ndarray]], test: int, search: int, seed: int, cpus: int) -> dict:
    """Runs on Modal: every network on `test` and `search` sampled episodes plus one argmax episode."""
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpus}"  # XLA:CPU runs one core per device
    import jax
    import jax.numpy as jnp
    from kinetix.environment import ActionType, ObservationType, make_kinetix_env
    from kinetix.environment.env_state import EnvParams
    from kinetix.environment.utils import static_env_params_from_size
    from kinetix.util.saving import load_evaluation_levels

    jax.config.update("jax_default_matmul_precision", "highest")  # no TF32, so every GPU type computes the same
    static = static_env_params_from_size("l")             # 12 polygons, 4 circles, 6 joints, 2 thrusters, frame_skip 2
    params = EnvParams().replace(dense_reward_scale=0.2)  # configs/env/symbolic.yaml; max_timesteps 256, dt 1/60
    level = jax.tree.map(lambda x: x[0], load_evaluation_levels(["l/h2_car_ramp"], static)[0])
    env = make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT, None, params, static,
                           auto_reset=False)

    def logits(w, x):
        h = jax.nn.relu(x @ w["embed.W"] + w["embed.b"])
        for k in range(5):
            h = jnp.tanh(h @ w[f"actor{k}.W"] + w[f"actor{k}.b"])
        return h @ w["actor_head.W"] + w["actor_head.b"]

    def episode(w, level, key, greedy):
        obs, state = env.reset(key, params, override_reset_state=level)
        d0 = jnp.linalg.norm(state.circle.position[0] - state.polygon.position[4])  # green wheel - blue box

        def step(c):
            obs, state, key, _, ret, length, _, dmin = c
            key, ka = jax.random.split(key)
            lg = logits(w, obs)
            a = jnp.stack([jnp.where(greedy, jnp.argmax(lg[i:j]), jax.random.categorical(k, lg[i:j]))
                           for k, (i, j) in zip(jax.random.split(ka, 6), SPLITS)])
            obs, state, r, done, info = env.step(key, state, a, params)
            outcome = jnp.select([info["GoalR"], r < -0.5, state.timestep >= params.max_timesteps], [0, 1, 2], 3)
            return obs, state, key, done, ret + r, length + 1, outcome, jnp.fmin(dmin, info["distance"])

        # Kinetix ends every episode by step 256; under vmap, finished episodes are held until all are done.
        c = jax.lax.while_loop(lambda c: ~c[3], step, (obs, state, key, False, 0.0, 0, -1, d0))
        _, _, _, _, ret, length, outcome, dmin = c
        return ret, length, outcome, jnp.where(outcome == 0, 1.0, 1 - dmin / d0)

    # Episodes split over the devices (all CPU cores, or the one GPU); the last episode is the argmax one.
    run = jax.pmap(jax.vmap(episode, in_axes=(None, None, 0, 0)), in_axes=(None, None, 0, 0))
    n, ndev = test + search + 1, jax.device_count()
    per = -(-n // ndev)
    keys = jnp.concatenate([jax.random.split(jax.random.PRNGKey(seed), test),
                            jax.random.split(jax.random.PRNGKey(seed + 1), search), jax.random.PRNGKey(0)[None],
                            jnp.zeros((per * ndev - n, 2), jnp.uint32)]).reshape(ndev, per, 2)
    greedy = (jnp.arange(per * ndev) == n - 1).reshape(ndev, per)

    def summary(ret, length, outcome, progress):
        return {"episodes": len(ret), **{o: float((outcome == i).mean()) for i, o in enumerate(OUTCOMES)},
                "return": float(ret.mean()), "length": float(length.mean()), "progress": float(progress.mean())}

    out = {}
    for name, w in nets.items():
        t0 = time.time()
        res = [np.asarray(x).reshape(-1)[:n]
               for x in run({k: jnp.asarray(v, jnp.float32) for k, v in w.items()}, level, keys, greedy)]
        out[name] = {"test": summary(*[x[:test] for x in res]), "search": summary(*[x[test:-1] for x in res]),
                     "argmax": {"outcome": OUTCOMES[res[2][-1]], "return": float(res[0][-1]),
                                "length": int(res[1][-1])},
                     "seconds": round(time.time() - t0, 1)}
    from importlib.metadata import version
    return {"nets": out, "test_seed": seed, "search_seed": seed + 1, "device": jax.devices()[0].device_kind,
            "devices": ndev, "versions": {p: version(p) for p in ["jax", "jaxlib", "jax2d", "flax", "kinetix-env"]},
            "kinetix_commit": KINETIX_COMMIT}


def evaluate(nets: dict[str, dict[str, np.ndarray]], test: int = 1024, search: int = 256, seed: int = 0,
             gpu: str | None = GPU) -> dict:
    modal = load_modal()
    image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
             .uv_pip_install("jax[cuda12]==0.4.38",
                             f"kinetix-env @ git+https://github.com/FLAIROx/Kinetix.git@{KINETIX_COMMIT}")
             .add_local_python_source("wsp"))
    app = modal.App("wsp-exp4", image=image)
    fn = app.function(gpu=gpu, cpu=CPUS, timeout=TIMEOUT_S, serialized=True)(_rollouts)
    t0 = time.time()
    with modal.enable_output(), app.run():
        out = fn.remote(nets, test, search, seed, CPUS)
    return {**out, "gpu": gpu, "cpus": CPUS, "wall_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    # python -m wsp.exp4 <answer.md | net.npz | shown> [test] [search] [gpu, e.g. T4; default CPU only]
    src = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    m = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    g = sys.argv[4] if len(sys.argv) > 4 else GPU
    w = (shown_network() if src == "shown" else load_npz(src) if src.endswith(".npz")
         else read_reply(Path(src).read_text(encoding="utf-8"))[0])
    print(json.dumps(evaluate({src: w}, n, m, gpu=g), indent=1))
