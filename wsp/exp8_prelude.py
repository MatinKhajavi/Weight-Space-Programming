# exp8 kernel prelude, run by exp8_kernel.py in the model's namespace. A one-line header keeps the line numbers.
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
