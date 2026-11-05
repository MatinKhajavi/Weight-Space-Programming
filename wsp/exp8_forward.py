"""exp8's forward-pass tool: one Modal GPU job per request. Generation runs on vLLM, activations and teacher-forced
reads on transformers; the edits are read from, and the results written to, the run's kernel sandbox."""
from __future__ import annotations

import json
import os
import time

from wsp.cloud import load_modal
from wsp.runner import UP

EXP = UP / "exp8_math"
GPU, FORWARD_TIMEOUT = "B200", 2 * 60 * 60
LIMITS = {"gen_max": 512, "tf_max_texts": 1024, "tf_max_tokens": 32768, "tf_total_tokens": 524288,
          "act_max_tokens": 8192, "attn_max_texts": 2, "attn_max_tokens": 2048, "max_bytes": 6 << 30}
FULL_ATTENTION = [3, 7, 11, 15, 19, 23]


def _items(req: dict, practice: dict) -> list[dict]:
    rows = []
    for pid in req.get("problems", []):
        if pid not in practice:
            raise ValueError(f"unknown practice id: {pid}")
        rows.append({"id": pid, **practice[pid]})
    rows += [{"id": f"prompt_{i}", "source": "prompt", "user": p} for i, p in enumerate(req.get("prompts", []))]
    return rows


def _generate(llm_path: str, rows: list[dict], n: int, max_tokens: int, tok, Q) -> tuple[list[dict], dict]:
    """vLLM generation with upstream evaluate.py's settings (protocol think35), seeded by item; graded by mathqa."""
    import zlib

    from vllm import LLM, SamplingParams
    llm = LLM(model=llm_path, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=max_tokens + 2048, seed=0)
    prompts = [tok.apply_chat_template([{"role": "user", "content": r.get("user") or Q.build_prompt(r["problem"])}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in rows]
    sp = [SamplingParams(n=n, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=max_tokens,
                         seed=zlib.crc32(r["id"].encode())) for r in rows]
    outs = llm.generate(prompts, sp)
    del llm
    res, fail = [], {"wrong_answer": 0, "no_answer_finished": 0, "cut_at_max_tokens": 0}
    for r, o in zip(rows, outs):
        samples = []
        for c in o.outputs:
            cut = c.finish_reason == "length"
            s = {"text": c.text, "pred": None, "correct": None, "tokens": len(c.token_ids), "cut": cut,
                 "_ids": list(c.token_ids)}
            if "answer" in r:
                g = Q.score({"gold": r["answer"]}, c.text)
                s["pred"], s["correct"] = g["pred"], g["correct"]
                if not g["correct"]:
                    fail["cut_at_max_tokens" if cut else "wrong_answer" if g["boxed"] else "no_answer_finished"] += 1
            samples.append(s)
        res.append({"id": r["id"], "source": r["source"], "answer": r.get("answer"), "samples": samples,
                    "_prompt_ids": list(o.prompt_token_ids)})
    graded = [s for x in res for s in x["samples"] if s["correct"] is not None]
    by = {}
    for x in res:
        for s in x["samples"]:
            if s["correct"] is not None:
                by.setdefault(x["source"], []).append(s["correct"])
    allg = [s for x in res for s in x["samples"]]
    summary = {"generations": len(allg), "graded": len(graded),
               "accuracy": round(sum(s["correct"] for s in graded) / max(1, len(graded)), 4),
               "accuracy_by_source": {k: round(sum(v) / len(v), 4) for k, v in sorted(by.items())},
               "mean_tokens": int(sum(s["tokens"] for s in allg) / max(1, len(allg))),
               "cut_share": round(sum(s["cut"] for s in allg) / max(1, len(allg)), 4), "failures": fail}
    return res, summary


def _hf(path: str):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()


def _capture(model, layers: list[int], mlp: list[int], attn: list[int]):
    """Hooks: residual stream after layer l (0 = embeddings), down_proj inputs, attention probabilities."""
    got, hooks = {}, []
    base = model.model
    if 0 in layers:
        hooks.append(base.embed_tokens.register_forward_hook(lambda m, i, o: got.__setitem__("hidden_0", o[0])))
    for layer in layers:
        if layer > 0:
            hooks.append(base.layers[layer - 1].register_forward_hook(
                lambda m, i, o, layer=layer: got.__setitem__(f"hidden_{layer}",
                                                             (o[0] if isinstance(o, tuple) else o)[0])))
    for layer in mlp:
        hooks.append(base.layers[layer].mlp.down_proj.register_forward_pre_hook(
            lambda m, i, layer=layer: got.__setitem__(f"mlp_act_{layer}", i[0][0])))
    for layer in attn:
        hooks.append(base.layers[layer].self_attn.register_forward_hook(
            lambda m, i, o, layer=layer: got.__setitem__(f"attn_{layer}", o[1][0])))
    return got, hooks


def _activations(path: str, res: list[dict], layers: list[int], mlp: bool, dst: str):
    """Per generated answer: residual stream (and down_proj inputs) at the last prompt token and averaged over the
    answer's first 8,192 tokens."""
    import numpy as np
    import torch
    model = _hf(path)
    mlp_layers = [layer for layer in layers if layer <= 23] if mlp else []
    got, hooks = _capture(model, layers, mlp_layers, [])
    hl, ha, ml, ma, index = [], [], [], [], []
    with torch.no_grad():
        for x in res:
            p = x["_prompt_ids"]
            for j, s in enumerate(x["samples"]):
                a = s["_ids"][:LIMITS["act_max_tokens"]]
                got.clear()
                model.model(torch.tensor([p + a], device="cuda"))
                cut = len(p) - 1
                hl.append(torch.stack([got[f"hidden_{layer}"][cut] for layer in layers]).float().cpu())
                ha.append(torch.stack([got[f"hidden_{layer}"][len(p):].float().mean(0) for layer in layers]).cpu())
                if mlp_layers:
                    ml.append(torch.stack([got[f"mlp_act_{layer}"][cut] for layer in mlp_layers]).float().cpu())
                    ma.append(torch.stack([got[f"mlp_act_{layer}"][len(p):].float().mean(0)
                                           for layer in mlp_layers]).cpu())
                index.append({"item": x["id"], "sample": j, "correct": s["correct"], "tokens": len(a)})
    for h in hooks:
        h.remove()
    os.makedirs(dst, exist_ok=True)
    arrays = {"hidden_last_prompt": hl, "hidden_mean_answer": ha}
    if mlp_layers:
        arrays.update({"mlp_last_prompt": ml, "mlp_mean_answer": ma})
    shapes = {}
    for name, rows in arrays.items():
        a = torch.stack(rows).numpy().astype(np.float16)
        np.save(f"{dst}/{name}.npy", a)
        shapes[f"{name}.npy"] = list(a.shape)
    with open(f"{dst}/index.json", "w") as f:
        json.dump({"layers": layers, "mlp_layers": mlp_layers, "samples": index}, f)
    del model, got
    torch.cuda.empty_cache()   # the next vLLM engine ("both") needs the GPU memory back
    return shapes


def _teacher_forced(path: str, req: dict, dst: str, edited: list[str]) -> dict:
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if "chats" in req:
        texts = [tok.apply_chat_template(c, tokenize=False, enable_thinking=bool(req.get("enable_thinking")))
                 for c in req["chats"]]
    else:
        texts = list(req["texts"])
    ret = req.get("return", {})
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    lengths = [len(i) for i in ids]
    if not 0 < len(ids) <= LIMITS["tf_max_texts"]:
        raise ValueError(f"teacher_forced takes 1 to {LIMITS['tf_max_texts']} texts")
    if max(lengths) > LIMITS["tf_max_tokens"] or sum(lengths) > LIMITS["tf_total_tokens"]:
        raise ValueError(f"texts are limited to {LIMITS['tf_max_tokens']} tokens each, "
                         f"{LIMITS['tf_total_tokens']} in total")
    layers, mlp, attn = ret.get("hidden_states", []), ret.get("mlp_act", []), ret.get("attn_layers", [])
    if attn and (set(attn) - set(FULL_ATTENTION) or len(ids) > LIMITS["attn_max_texts"]
                 or max(lengths) > LIMITS["attn_max_tokens"]):
        raise ValueError(f"attn_layers: full-attention layers {FULL_ATTENTION} only, at most "
                         f"{LIMITS['attn_max_texts']} texts of at most {LIMITS['attn_max_tokens']} tokens")
    topk = int(ret.get("topk", 0))
    if topk > 50:
        raise ValueError("topk is at most 50")
    T, L = len(ids), max(lengths)
    shapes = {"input_ids": (T, L)}
    if ret.get("token_logprobs"):
        shapes["token_logprobs"] = (T, L - 1)
    if topk:
        shapes["topk_logprobs"] = shapes["topk_ids"] = (T, L, topk)
    shapes.update({f"hidden_{layer}": (T, L, 1024) for layer in layers})
    shapes.update({f"mlp_act_{layer}": (T, L, 3584) for layer in mlp})
    shapes.update({f"attn_{layer}": (T, 8, L, L) for layer in attn})
    size = sum(int(np.prod(s)) * (8 if k in ("input_ids", "topk_ids") else 4 if "logprobs" in k else 2)
               for k, s in shapes.items())
    if size > LIMITS["max_bytes"]:
        raise ValueError(f"the results would take {size / 2**30:.1f} GB; at most 6 GB per pass")
    os.makedirs(dst, exist_ok=True)
    dtypes = {"input_ids": np.int64, "topk_ids": np.int64, "token_logprobs": np.float32, "topk_logprobs": np.float32}
    arrs = {k: np.lib.format.open_memmap(f"{dst}/{k}.npy", "w+", dtypes.get(k, np.float16), s)
            for k, s in shapes.items()}
    model = _hf(path)
    got, hooks = _capture(model, layers, mlp, attn)
    mean_lp = []
    with torch.no_grad():
        for i, x in enumerate(ids):
            got.clear()
            t = torch.tensor([x], device="cuda")
            h = model.model(t, output_attentions=bool(attn)).last_hidden_state[0]
            arrs["input_ids"][i, :len(x)] = x
            tl = []
            for a in range(0, len(x), 2048):   # the 248,320-token vocabulary: logits in chunks
                lp = torch.log_softmax(model.lm_head(h[a:a + 2048]).float(), -1)
                nxt = t[0, a + 1:a + 2049]
                tl.append(lp[:len(nxt)].gather(1, nxt[:, None])[:, 0])
                if topk:
                    v, j = lp.topk(topk, -1)
                    arrs["topk_logprobs"][i, a:a + len(lp)] = v.cpu().numpy()
                    arrs["topk_ids"][i, a:a + len(lp)] = j.cpu().numpy()
            tl = torch.cat(tl)
            mean_lp.append(round(float(tl.mean()), 4) if len(x) > 1 else None)
            if "token_logprobs" in arrs:
                arrs["token_logprobs"][i, :len(x) - 1] = tl.cpu().numpy()
            for k, v in got.items():
                if k.startswith("attn_"):
                    arrs[k][i, :, :len(x), :len(x)] = v.float().cpu().numpy()
                else:
                    arrs[k][i, :len(x)] = v.float().cpu().numpy()
    for h in hooks:
        h.remove()
    for a in arrs.values():
        a.flush()
    meta = {"texts": texts, "lengths": lengths, "model": req.get("model", "original"), "edited_tensors": edited,
            "files": {f"{k}.npy": list(s) for k, s in shapes.items()},
            "padding": "right; positions >= length are padding"}
    with open(f"{dst}/meta.json", "w") as f:
        json.dump(meta, f)
    return {"lengths": lengths, "files": {f"{k}.npy": list(s) for k, s in shapes.items()},
            "edited_tensors": len(edited), "mean_token_logprob": mean_lp}


def _forward(sandbox_id: str, k: int, req: dict) -> dict:
    """Runs on Modal (GPU): one forward pass; the edits are read from, and the results written to, the run's sandbox."""
    import shutil
    import sys

    import modal

    from wsp.exp8_eval import BASE, CODE, TASKS
    t0 = time.time()
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"   # the Modal client below runs threads: no fork after them
    sys.path[:0] = [TASKS, CODE]   # mathqa next to the image's mcqa stub, as wsp.exp8_eval runs it
    os.chdir("/root")
    os.environ["MATHEDIT_PROFILE"] = "qwen35think"
    import build_ckpt
    import mathqa as Q
    from transformers import AutoTokenizer
    sb = modal.Sandbox.from_id(sandbox_id)
    try:
        names = [f.name for f in sb.filesystem.list_files("/work/edited") if f.name.endswith(".npy")]
    except Exception:   # noqa: BLE001 - no edits yet
        names = []
    os.makedirs("/tmp/sub", exist_ok=True)
    for f in names:
        sb.filesystem.copy_to_local(f"/work/edited/{f}", f"/tmp/sub/{f}")
    edited = sorted(f[:-4] for f in names)
    kind, which = req.get("type"), req.get("model", "original")
    if which not in ("original", "edited", "both") or (kind == "teacher_forced" and which == "both"):
        raise ValueError(f"model must be original or edited{' or both' if kind == 'generate' else ''}")
    path = {"original": BASE}
    if which != "original":
        build_ckpt.build("/tmp/sub", "/tmp/ckpt_edited")
        path["edited"] = "/tmp/ckpt_edited"
    res_dir = "/tmp/res"
    shutil.rmtree(res_dir, ignore_errors=True)
    if kind == "teacher_forced":
        done = {"ok": True, **_teacher_forced(path[which], req, f"{res_dir}/fwd_{k}", edited),
                "type": "teacher_forced"}
    elif kind == "generate":
        practice = {}
        for f in ["gsm8k_train", "math_train", "aime"]:
            with open(f"/practice/{f}.jsonl") as lines:
                for line in lines:
                    r = json.loads(line)
                    practice[r["id"]] = r
        rows = _items(req, practice)
        n, mt = int(req.get("n", 1)), int(req.get("max_tokens", 32768))
        cap = LIMITS["gen_max"] // (2 if which == "both" else 1)
        if not rows or len(rows) * n > cap or not 1 <= mt <= 32768:
            raise ValueError(f"items x n must be 1 to {cap}{' per model' if which == 'both' else ''}; "
                             "max_tokens at most 32768")
        tok = AutoTokenizer.from_pretrained(BASE)
        act = req.get("activations")
        models = ["original", "edited"] if which == "both" else [which]
        os.makedirs(res_dir, exist_ok=True)
        out, files, acts = {}, {}, {}
        for m in models:
            res, summ = _generate(path[m], rows, n, mt, tok, Q)
            name = f"gen_{k}_{m}" if which == "both" else f"gen_{k}"
            with open(f"{res_dir}/{name}.jsonl", "w") as f:
                for x in res:
                    f.write(json.dumps({key: ([{a: b for a, b in s.items() if a != "_ids"} for s in v]
                                              if key == "samples" else v)
                                        for key, v in x.items() if key != "_prompt_ids"}) + "\n")
            if act:
                shapes = _activations(path[m], res, list(act.get("layers", [])), bool(act.get("mlp")),
                                      f"{res_dir}/{name}_acts")
                acts[m] = {"dir": f"/work/{name}_acts/", "files": shapes}   # as upstream's summaries
            out[m], files[m] = (res, summ), f"/work/{name}.jsonl"
        if which == "both":
            (ro, sum_o), (re_, sum_e) = out["original"], out["edited"]
            d, w2r, r2w = [], 0, 0
            for a, b in zip(ro, re_):
                if a["answer"] is None:
                    continue
                ca, cb = [s["correct"] for s in a["samples"]], [s["correct"] for s in b["samples"]]
                d.append(sum(cb) / len(cb) - sum(ca) / len(ca))
                w2r += sum(not x and y for x, y in zip(ca, cb))
                r2w += sum(x and not y for x, y in zip(ca, cb))
            mean = sum(d) / max(1, len(d))
            se = (sum((x - mean) ** 2 for x in d) / (len(d) - 1)) ** 0.5 / len(d) ** 0.5 if len(d) > 1 else None
            done = {"ok": True, "type": "generate", "model": "both", "files": files, "activations": acts or None,
                    "original": sum_o, "edited": sum_e,
                    "paired": {"items": len(d), "accuracy_difference": round(mean, 4),
                               "standard_error": round(se, 4) if se is not None else None,
                               "samples_wrong_to_right": w2r, "samples_right_to_wrong": r2w}}
        else:
            done = {"activations": acts.get(which), "ok": True, "type": "generate", "model": which,
                    "file": files[which], **out[which][1]}
    else:
        raise ValueError('type must be "generate" or "teacher_forced"')
    from concurrent.futures import ThreadPoolExecutor
    files = [os.path.join(root, f) for root, _, fs in os.walk(res_dir) for f in fs]
    with ThreadPoolExecutor(8) as ex:   # one copy runs at about 1 MB/s from here; several at once go faster
        list(ex.map(lambda src: sb.filesystem.copy_from_local(src, "/work/" + os.path.relpath(src, res_dir)), files))
    done["s"] = round(time.time() - t0, 1)
    return done


def run_forward(sandbox_id: str, k: int, request: dict) -> dict:
    from wsp.exp8_eval import VOLUME, _image
    modal = load_modal()
    app = modal.App("wsp-exp8-forward", image=_image(modal).add_local_dir(str(EXP / "practice"), "/practice"))
    fn = app.function(gpu=GPU, cpu=8, memory=65536, timeout=FORWARD_TIMEOUT, serialized=True,
                      volumes={"/vol": modal.Volume.from_name(VOLUME).read_only()})(_forward)
    with app.run():
        try:
            return fn.remote(sandbox_id, k, request)
        except Exception as e:   # noqa: BLE001 - reported to the model as a failed request, not counted
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
