"""The exp8 forward-pass tool alone (a sandbox with one edit, then a paired generation and a teacher-forced pass)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import exp8  # noqa: E402

k = exp8.Kernel(f"wsp-exp8-test-{int(time.time())}")
print(k.run("n = 'model.language_model.layers.15.mlp.down_proj.weight'; w = load_original(n); w[:, :64] = 0; "
            "save_edit(n, w); edited()"), flush=True)
gen = {"type": "generate", "model": "both", "problems": ["gsm8k_train_0", "gsm8k_train_1", "math_train_algebra_0"],
       "n": 2, "max_tokens": 1024, "activations": {"layers": [4, 24], "mlp": True}}
tf = {"type": "teacher_forced", "model": "edited", "chats": [[{"role": "user", "content": "What is 2+2?"},
                                                               {"role": "assistant", "content": "4"}]],
      "enable_thinking": True, "return": {"token_logprobs": True, "hidden_states": [24]}}
try:
    for i, req in [(1, gen), (2, tf), (3, {"type": "generate", "model": "original", "problems": ["nope"]})]:
        t0 = time.time()
        print(f"=== forward {i} ({req['type']})", flush=True)
        print(json.dumps(exp8.run_forward(k.sb.object_id, i, req), indent=1)[:3000], f"\n[{time.time() - t0:.0f} s]",
              flush=True)
    print(k.run("sorted(os.listdir('/work')), json.load(open('/work/gen_1_original_acts/index.json'))['mlp_layers'], "
                "np.load('/work/gen_1_edited_acts/hidden_mean_answer.npy').shape, json.load(open('/work/fwd_2/meta.json'))['texts']"))
    print(k.run("r = json.loads(open('/work/gen_1_original.jsonl').readline()); "
                "[(s['pred'], s['correct'], s['tokens'], s['cut'], s['text'][:100]) for s in r['samples']]"))
finally:
    k.sb.terminate()
    print("sandbox terminated", flush=True)
