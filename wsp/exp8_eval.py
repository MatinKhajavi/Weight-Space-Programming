"""Experiment 8, evaluation: Qwen3.5-0.8B (the text part of Qwen/Qwen3.5-0.8B), unedited or with a weight edit, on
MATH-500 (4 samples per problem), GSM8K test (2) and IFEval (2, strict prompt-level), with the authors' scripts
(upstream exp8_math/code, run unchanged: evaluate.py protocol think35, build_ckpt.py, apply_edit.py, merge.py) and vLLM
on Modal.

  python -m wsp.exp8_eval prepare                        benchmark files and the text-only base model, on the Volume
  python -m wsp.exp8_eval unedited [--smoke]
  python -m wsp.exp8_eval edit <edit folder> <name> [--smoke]
  python -m wsp.exp8_eval fetch <name>                   download eval/<name> to CACHE and merge it (merge.py)
  python -m wsp.exp8_eval diff <name> <dest>             make_edit_diff.py on an evaluated edit -> <dest>

<edit folder>: the submitted tensors (<tensor name>.npy, cast to bfloat16 by build_ckpt.py) or an upstream edit diff
(manifest.json + edit.npz, turned into those tensors by apply_edit.py). One GPU container per benchmark shard; each
writes eval/<name>/parts/<set>_s<i>of<k>/ as in upstream's eval/. --smoke: 1 % of each benchmark, at most 4,096 new
tokens, on an A10G.

The release's paths are relative placeholders ("<data>/mathedit/..."); the scripts run from /root, where <data> links to
the Volume. Not in the release, so ours: the IFEval benchmark file (google/IFEval, ids ifeval_<key>, in the order of the
upstream shards), its checkers (google-research instruction_following_eval at IFEVAL_COMMIT, placed where evaluate.py
imports them, weightenv/lmo/tasks/ifevalg), a stub for weightenv's mcqa (mathqa.py imports loop_signals from it; unused
here) and the text-only base model (the model.language_model.* tensors of the release, base_model/Qwen3.5-0.8B-text's
files, the release's tokenizer.json).
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from wsp.cloud import load_modal
from wsp.runner import UP, read_dotenv

EXP = UP / "exp8_math"
CACHE = Path.home() / ".cache" / "wsp" / "exp8"
VOLUME = "wsp-exp8"
BASE_REPO, BASE_REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"     # dataset google/IFEval
IFEVAL_COMMIT = "e6890f85757dd84e27ca6df2dd30651dafad28e0"       # google-research/google-research
SHARDS = {"math500": 1, "gsm8k": 1, "ifeval": 1}
# B200: the lowest cost per token here (decode at ~16K context: A10G 1.0K tok/s, A100 4.0K, H100 8.7K, H200 11.1K,
# B200 16.7K; about $0.40, 0.20, 0.14, 0.12 and 0.11 per million tokens)
GPU, TIMEOUT_S = "B200", int(2.5 * 60 * 60)
SMOKE_GPU, SMOKE_SHARD, SMOKE_TOKENS = "A10G", "0/100", 4096

REPO = "/opt/up"                        # evaluate.py puts the folder three levels above itself on sys.path
CODE, TASKS = f"{REPO}/exp8_math/code", f"{REPO}/weightenv/lmo/tasks"
DATA = "/vol/mathedit"                  # "<data>/mathedit" seen from /root
BASE = f"{DATA}/models/Qwen3.5-0.8B-text"


def _image(modal):
    url = f"https://raw.githubusercontent.com/google-research/google-research/{IFEVAL_COMMIT}/instruction_following_eval"
    fetch = " && ".join(f"curl -sfLO {url}/{f}.py"
                        for f in ["instructions", "instructions_registry", "instructions_util"])
    return (modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.12").entrypoint([])
            .apt_install("curl")
            .uv_pip_install("vllm==0.32.0", "transformers==5.17.0", "datasets==5.0.1", "math-verify==0.9.0",
                            "latex2sympy2_extended==1.11.0", "antlr4-python3-runtime==4.13.2", "nltk==3.8.1",
                            "langdetect==1.0.9", "absl-py==2.5.0", "immutabledict==4.3.1")
            .run_commands("python -m nltk.downloader -d /usr/local/share/nltk_data punkt",
                          f"mkdir -p {TASKS}/ifevalg && cd {TASKS}/ifevalg && {fetch}",
                          f"ln -s {TASKS}/ifevalg {REPO}/instruction_following_eval",   # their own imports
                          f"echo 'loop_signals = None  # mcqa is not in the release' > {TASKS}/mcqa.py",
                          "ln -s /vol '/root/<data>'")
            .add_local_dir(str(EXP / "code"), CODE)
            .add_local_file(str(EXP / "code" / "mathqa.py"), f"{TASKS}/mathqa.py")
            .add_local_dir(str(EXP / "base_model"), f"{REPO}/exp8_math/base_model")
            .add_local_python_source("wsp"))


def _prepare():
    """Runs on Modal: evaluate.py prepare, the IFEval file, the text-only base model."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    os.chdir("/root")
    subprocess.run(["python", f"{CODE}/evaluate.py", "prepare"], check=True)
    with open(f"{DATA}/bench/ifeval.jsonl", "w") as f:
        for r in load_dataset("google/IFEval", split="train", revision=IFEVAL_REVISION):
            row = {"id": f"ifeval_{r['key']}", "kind": "ifeval", "prompt": r["prompt"],
                   "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.makedirs(BASE, exist_ok=True)
    for f in ["config.json", "generation_config.json", "tokenizer_config.json", "chat_template.jinja"]:
        shutil.copy(f"{REPO}/exp8_math/base_model/Qwen3.5-0.8B-text/{f}", BASE)
    shutil.copy(hf_hub_download(BASE_REPO, "tokenizer.json", revision=BASE_REVISION), BASE)
    sys.path.insert(0, CODE)
    import st_numpy as S
    full = S.load(hf_hub_download(BASE_REPO, "model.safetensors-00001-of-00001.safetensors", revision=BASE_REVISION))
    text = {k: v for k, v in full.items() if k.startswith("model.language_model.")}
    S.save(f"{BASE}/model.safetensors", text, {"format": "pt"})
    print(f"{len(text)} of {len(full)} tensors -> {BASE}")
    _commit()


def _build(name: str):
    """Runs on Modal: build_ckpt.py on runs/<name> (its submission/ made by apply_edit.py from an edit diff)."""
    os.chdir("/root")
    run = f"{DATA}/runs/{name}"
    if os.path.exists(f"{run}/edit/manifest.json"):
        subprocess.run(["python", f"{CODE}/apply_edit.py", f"{BASE}/model.safetensors", f"{run}/edit",
                        f"{run}/submission"], check=True)
    shutil.rmtree(f"{DATA}/ckpt/{name}", ignore_errors=True)
    subprocess.run(["python", f"{CODE}/build_ckpt.py", run, f"{DATA}/ckpt/{name}"], check=True,
                   env={**os.environ, "MATHEDIT_PROFILE": "qwen35think"})
    _commit()


def _evaluate(ckpt: str, out: str, sets: list[str], shard: str, max_tokens: int | None = None):
    """Runs on Modal: evaluate.run as `evaluate.py run <ckpt> <out> --sets .. --protocol think35 --shard ..`."""
    os.chdir("/root")
    sys.path.insert(0, CODE)
    import evaluate as E
    if max_tokens:      # smoke tests only
        E.SAMPLING_THINK35["max_tokens"] = max_tokens
    E.run(ckpt, out, sets, protocol="think35", shard=shard)
    _commit()


def _commit():
    """Runs on Modal: make the Volume writes visible to the next container now (not only when this one stops)."""
    import modal
    modal.Volume.from_name(VOLUME).commit()


def _app(modal):
    return modal.App("wsp-exp8-eval", image=_image(modal)), modal.Volume.from_name(VOLUME, create_if_missing=True)


def prepare():
    modal = load_modal()
    app, vol = _app(modal)
    secret = modal.Secret.from_dict({"HF_TOKEN": read_dotenv()["HF_TOKEN"]})
    fn = app.function(cpu=4, memory=16384, timeout=1800, volumes={"/vol": vol}, secrets=[secret],
                      serialized=True)(_prepare)
    with modal.enable_output(), app.run():
        fn.remote()


def evaluate(name: str, edit: Path | None = None, smoke: bool = False):
    modal = load_modal()
    app, vol = _app(modal)
    build = app.function(cpu=2, memory=8192, timeout=1800, volumes={"/vol": vol}, serialized=True)(_build)
    gpu = SMOKE_GPU if smoke else GPU
    run = app.function(gpu=gpu, cpu=4, memory=16384, timeout=1800 if smoke else TIMEOUT_S, volumes={"/vol": vol},
                       single_use_containers=True, serialized=True)(_evaluate)
    ckpt = BASE
    if edit:
        kind = "edit" if (edit / "manifest.json").exists() else "submission"
        assert kind == "edit" or any(edit.glob("*.npy")), f"{edit}: no manifest.json and no .npy files"
        with contextlib.suppress(FileNotFoundError, modal.exception.InvalidError):   # not there yet
            vol.remove_file(f"mathedit/runs/{name}", recursive=True)   # no tensors left from an earlier upload
        with vol.batch_upload() as batch:
            batch.put_directory(str(edit), f"mathedit/runs/{name}/{kind}")
        ckpt = f"{DATA}/ckpt/{name}"
    out = f"{DATA}/eval/{'smoke_' if smoke else ''}{name}"
    if smoke:
        jobs = [(ckpt, out, list(SHARDS), SMOKE_SHARD, SMOKE_TOKENS)]
    else:
        jobs = [(ckpt, f"{out}/parts/{s}_s{i}of{k}", [s], f"{i}/{k}") for s, k in SHARDS.items() for i in range(k)]
    print(f"{name}: {len(jobs)} job(s) on {gpu}: {[j[1].split('/')[-1] for j in jobs]}", flush=True)
    with modal.enable_output(), app.run():
        if edit:
            build.remote(name)
        results = list(run.starmap(jobs, return_exceptions=True))     # a failed job stops no other
    for job, res in zip(jobs, results):
        print(job[1].split("/")[-1], "FAILED: " + repr(res) if res else "done", flush=True)


def _diff(name: str):
    """Runs on Modal: make_edit_diff.py, the exact compact record of a submitted edit (as upstream's
    runs/<run>/edit)."""
    os.chdir("/root")
    run = f"{DATA}/runs/{name}"
    subprocess.run(["python", f"{CODE}/make_edit_diff.py", f"{BASE}/model.safetensors",
                    f"{DATA}/ckpt/{name}/model.safetensors", f"{run}/submission", f"{run}/edit_diff"], check=True,
                   env={**os.environ, "MATHEDIT_PROFILE": "qwen35think"})
    _commit()


def diff(name: str, dst: Path):
    modal = load_modal()
    app, vol = _app(modal)
    fn = app.function(cpu=2, memory=8192, timeout=1800, volumes={"/vol": vol}, serialized=True)(_diff)
    with modal.enable_output(), app.run():
        fn.remote(name)
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"mathedit/runs/{name}/edit_diff/", str(dst)],
                   check=True, env=os.environ)


def fetch(name: str):
    load_modal()
    CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"mathedit/eval/{name}", str(CACHE)], check=True,
                   env=os.environ)
    subprocess.run([sys.executable, str(EXP / "code" / "merge.py"), str(CACHE), name], check=True,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--smoke"]
    smoke = "--smoke" in sys.argv
    if args[0] == "prepare":
        prepare()
    elif args[0] == "unedited":
        evaluate("unedited", smoke=smoke)
    elif args[0] == "edit":
        evaluate(args[2], Path(args[1]).resolve(), smoke)
    elif args[0] == "fetch":
        fetch(args[1])
    elif args[0] == "diff":
        diff(args[1], Path(args[2]).resolve())
    else:
        sys.exit(__doc__)
