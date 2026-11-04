"""The persistent Python kernel of exp8, run inside the Modal sandbox (no wsp imports).

Reads one JSON line {"code": ...} per run on stdin, runs it in one persistent namespace (stdout and stderr together;
the value of a last-line expression is shown) and answers with one line: MARK + JSON {"out": ...}.
"""
import ast
import contextlib
import io
import json
import resource
import sys
import traceback

MARK = "\x00wsp-kernel\x00"
MEM = 30 << 30   # below the sandbox's 32 GB, so going over raises MemoryError instead of killing the sandbox

PRELUDE = r'''
import json, os
import numpy as np
import torch
from pathlib import Path
from safetensors import safe_open
from tokenizers import Tokenizer

torch.set_grad_enabled(False)
MODEL, WORK = "/model", "/work"
FROZEN = {"model.language_model.norm.weight", "lm_head.weight", "model.language_model.embed_tokens.weight"}
with safe_open(MODEL + "/model.safetensors", "pt") as _f:
    NAMES = list(_f.keys())
del _f
tok = Tokenizer.from_file(MODEL + "/tokenizer.json")


def load_original(name):
    with safe_open(MODEL + "/model.safetensors", "pt") as f:
        return f.get_tensor(name).float()


def load(name):
    p = f"{WORK}/edited/{name}.npy"
    return torch.from_numpy(np.load(p)) if os.path.exists(p) else load_original(name)


def save_edit(name, tensor):
    if name in FROZEN:
        raise ValueError(f"save_edit refused: {name} may not be edited")
    if name not in NAMES:
        raise ValueError(f"save_edit refused: no tensor named {name}")
    t = torch.as_tensor(tensor).detach().float().cpu()
    o = load_original(name)
    if tuple(t.shape) != tuple(o.shape):
        raise ValueError(f"save_edit refused: shape {tuple(t.shape)} is not {tuple(o.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError("save_edit refused: the tensor has inf or nan values")
    c = float((t * o).sum() / (o * o).sum())
    if abs(c - 1) > 1e-6 and torch.allclose(t, c * o, rtol=1e-5, atol=1e-8):
        raise ValueError(f"save_edit refused: the edit only multiplies {name} by a constant ({c:.6g})")
    os.makedirs(WORK + "/edited", exist_ok=True)
    np.save(f"{WORK}/edited/{name}.npy", t.numpy())


def drop_edit(name):
    p = f"{WORK}/edited/{name}.npy"
    if os.path.exists(p):
        os.remove(p)


def edited():
    d = WORK + "/edited"
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npy")) if os.path.isdir(d) else []
'''


def run(code, g):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            tree = ast.parse(code, "<run>")
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, "<run>", "exec"), g)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), "<run>", "eval"), g)
                if value is not None:
                    print(repr(value))
        except BaseException as e:   # noqa: BLE001 - everything goes back to the model, as in a notebook
            traceback.print_exception(type(e), e, e.__traceback__.tb_next)
    return buf.getvalue()


def main():
    resource.setrlimit(resource.RLIMIT_AS, (MEM, MEM))
    g = {"__name__": "__main__"}
    exec(compile(PRELUDE, "<prelude>", "exec"), g)
    for line in sys.stdin:
        out = run(json.loads(line)["code"], g)
        sys.stdout.write(MARK + json.dumps({"out": out}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
