"""The exp8 kernel sandbox and forward-pass tool on scripted code (no model calls). A few minutes of an H100."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import exp8  # noqa: E402

vol = f"wsp-exp8-test-{int(time.time())}"
k = exp8.Kernel(vol)
print("sandbox", k.sb.object_id, flush=True)
CODES = [
    "sorted(n for n in globals() if not n.startswith('_'))",
    "len(NAMES), NAMES[:2], tuple(load(NAMES[5]).shape), tok.encode('</think>').ids, sorted(os.listdir(MODEL))",
    "x = 5",
    "x + 1",
    "1/0",
    "print('caf\\u00e9 \\u03c0'); [print('this line repeats itself') for _ in range(5)]; None",
    "save_edit('lm_head.weight', load_original('model.language_model.embed_tokens.weight'))",
    "n = 'model.language_model.layers.15.mlp.down_proj.weight'; save_edit(n, 2 * load_original(n))",
    "w = load_original(n); w[:, :8] = 0; save_edit(n, w); edited(), float((load(n) - w).abs().max())",
    "import torch; a = torch.ones((40 << 30) // 4)",
    "open('/model/x', 'w')",
    "import urllib.request; urllib.request.urlopen('http://example.com', timeout=5)",
    "sorted(os.listdir('/practice')), torch.get_num_threads()",
]
for c in CODES:
    t0 = time.time()
    print(f"--- {c[:70]}\n{exp8.shown(k.run(c))[:600]}  [{time.time() - t0:.1f} s]", flush=True)

row = json.loads(open(exp8.EXP / "practice" / "gsm8k_train.jsonl").readline())
prefix, suffix = "<|im_start|>user\n", ("\nPlease reason step by step, and put your final answer within \\boxed{}."
                                        "<|im_end|>\n<|im_start|>assistant\n<think>\n")
good = prefix + row["problem"] + suffix + row["solution"] + "\n</think>\n\n\\boxed{" + row["answer"] + "}"
tf = {"type": "teacher_forced", "model": "original", "texts": [good, "The the the the the the the the."],
      "return": {"token_logprobs": True, "topk": 5, "hidden_states": [0, 12, 24], "mlp_act": [3], "attn_layers": [3]}}
gen = {"type": "generate", "model": "both", "problems": ["gsm8k_train_0", "gsm8k_train_1", "math_train_algebra_0"],
       "n": 2, "max_tokens": 1024, "activations": {"layers": [4, 24], "mlp": True}}
for n, req in [(1, tf), (2, gen), (3, {"type": "generate", "model": "original", "problems": ["nope"]})]:
    t0 = time.time()
    done = exp8.run_forward(k.sb.object_id, n, req)
    print(f"=== forward {n} ({time.time() - t0:.0f} s)\n{json.dumps(done, indent=1)[:2500]}", flush=True)
print(k.run("sorted(os.listdir('/work')), sorted(os.listdir('/work/fwd_1')), "
            "json.load(open('/work/gen_2_original_acts/index.json'))['mlp_layers'], "
            "np.load('/work/gen_2_edited_acts/hidden_mean_answer.npy').shape, "
            "np.load('/work/fwd_1/attn_3.npy').shape, np.load('/work/fwd_1/token_logprobs.npy')[0, :8]"))
print(k.run("r = json.loads(open('/work/gen_2_original.jsonl').readline()); "
            "[(s['pred'], s['correct'], s['tokens'], s['cut'], s['text'][:120]) for s in r['samples']]"))
k.sb.terminate()
print("done; test volume", vol)
